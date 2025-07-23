import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import timm
from tqdm import tqdm
import os
import logging
from datetime import datetime
from peft import LoraConfig, get_peft_model, TaskType
import copy
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
from petl import vision_transformer_ssf
from _exp import ContinualLearnerHead, RandomProjectionHead
from util import compute_metrics, accuracy, set_random
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')


def setup_logger(log_file=f'logs/{timestamp}_exp29.log'):
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)
    
    logger.propagate = False

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_formatter = logging.Formatter('%(asctime)s - %(message)s')
    console_formatter = logging.Formatter('%(message)s')
    
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logger()

config = {
}
logger.info(f"{' | '.join('%s: %s' % item for item in config.items())}")

class Program(nn.Module):
    def __init__(self, img_shape=(3, 224, 224), mask_size=50):
        super().__init__()
        if isinstance(mask_size, int):
            mask_size = (mask_size, mask_size)
        
        self.mask_size = mask_size
        self.img_shape = img_shape

        self.W = nn.Parameter(torch.randn(1, *img_shape), requires_grad=True)
        self.M = nn.Parameter(torch.ones(1, *img_shape), requires_grad=False)
        self.M[:, :, 
               (img_shape[1] - mask_size[0])//2:(img_shape[1] + mask_size[0])//2, 
               (img_shape[2] - mask_size[1])//2:(img_shape[2] + mask_size[1])//2] = 0
    
    def reset(self):
        self.W.data = torch.randn(1, *self.img_shape).to(self.W.device)
    
    def forward(self, x):
        B, C, H, W = x.size()
        if C != self.img_shape[0]:
            x = x.repeat(1, self.img_shape[0] // C, 1, 1)
        x_scaled = F.interpolate(x, size=self.mask_size, mode='bicubic', align_corners=False)
        background = torch.zeros(B, *self.img_shape).to(x.device)
        background[:, :, 
                   (self.img_shape[1] - self.mask_size[0])//2:(self.img_shape[1] + self.mask_size[0])//2,
                   (self.img_shape[2] - self.mask_size[1])//2:(self.img_shape[2] + self.mask_size[1])//2] = x_scaled

        x_adv = background + torch.tanh(self.W * self.M)
        return x_adv

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.program = Program(mask_size=200)
        self.backbone = timm.create_model("vit_base_patch16_224_ssf", pretrained=True, num_classes=0)
        self.freeze_backbone()
        self.head = ContinualLearnerHead(768, 20)
    
    def reset(self):
        self.program.reset()
        for name, param in self.backbone.named_parameters():
            if "ssf_shift" in name:
                nn.init.zeros_(param)
            elif "ssf_scale" in name:
                nn.init.ones_(param)
        self.head.reset([0])
    
    def freeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            if "head" not in name and "ssf_" not in name:
                param.requires_grad_(False)
    
    def normalize(self, x):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x = (x - mean) / std
        return x
    
    def forward(self, x):
        # x = self.program(x)
        # x = self.normalize(x)
        
        f = self.backbone(x)
        y = self.head(f)['logits']
        return y
    
    def __repr__(self):
        trainable_params = count_parameters(self, trainable=True)
        total_params = count_parameters(self)
        return f"Model(trainable_params={trainable_params}, total_params={total_params}, percentage={trainable_params * 100 / total_params:.2f})"

def entropy(logits):
    return -F.softmax(logits, dim=0) * F.log_softmax(logits, dim=0)

def energy(logits):
    return -torch.logsumexp(logits, dim=0)

def margin(logits):
    probs = F.softmax(logits, dim=0)
    top2, _ = torch.topk(probs, k=2, dim=0)
    margin = top2[0] - top2[1]
    return 1 - margin

def wasserstein(logits, aug_logits):
    probs = F.softmax(logits, dim=0)
    aug_probs = F.softmax(aug_logits, dim=0)
    dist = torch.sum(torch.abs(torch.cumsum(probs, dim=0) - torch.cumsum(aug_probs, dim=0)), dim=0)
    return dist.mean()

def kl_divergence(p, q):
    epsilon = 1e-10
    return torch.sum(p * (torch.log(p + epsilon) - torch.log(q + epsilon)), dim=0)

def symmetric_kl(logits, aug_logits):
    probs = F.softmax(logits, dim=0)
    aug_logits = F.softmax(aug_logits, dim=0)
    loss = 0.5 * (kl_divergence(probs, aug_logits) + kl_divergence(aug_logits, probs))
    return loss.mean()

class Learner:
    def __init__(self):
        self._known_classes = 0
        self._classes_seen_so_far = 0
        self._class_increments = []
        self.model = Model().to(device)
        self.accuracy_matrix = []
        self.cur_task = -1
        
        self.programs = []
        self.backbones = []
        self.heads = []

    def learn(self, data_manager):
        self.data_manager = data_manager
        num_tasks = data_manager.nb_tasks - 1

        for task in range(num_tasks):
            self.before_task(task, data_manager)

            trainset = data_manager.get_dataset(
                np.arange(self._known_classes, self._classes_seen_so_far), source="train", mode="train")
            train_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
            self.finetune(train_loader)

            test_set = self.data_manager.get_dataset(
                np.arange(0, self._classes_seen_so_far), source="test", mode="test")
            test_loader = DataLoader(test_set, batch_size=128, shuffle=True, num_workers=4)
            self.eval(test_loader)
            self.after_task()

    def before_task(self, task, data_manager):
        print(self.model)
        self._classes_seen_so_far = self._known_classes + data_manager.get_task_size(task)
        self._class_increments.append((self._known_classes, self._classes_seen_so_far - 1))
        self.cur_task = task

    def after_task(self):
        self.model.reset()
        self._known_classes = self._classes_seen_so_far

    def eval(self, test_loader):
        y_pred, y_true = [], []

        task_true = []
        sum_ent_pred = []
        min_ent_pred = []
        sum_ene_pred = []
        min_ene_pred = []
        sum_mar_pred = []
        sum_was_pred = []
        sum_sym_pred = []
        
        with torch.no_grad():
            for _, (_, a, x, y) in tqdm(enumerate(test_loader)):
                a, x, y = a.cuda(), x.cuda(), y.cuda()
                
                # batch_predicts = []
                # for _, (ix, iy) in enumerate(zip(x, y)):
                #     task_index = (iy // 20).item()
                    
                #     self.model.program.load_state_dict(self.programs[task_index], strict=True)
                #     self.model.backbone.load_state_dict(self.backbones[task_index], strict=False)
                #     self.model.head.load_state_dict(self.heads[task_index], strict=True)
                    
                #     logits = self.model(ix.unsqueeze(0))
                #     predicts = torch.topk(logits, k=1, dim=1, largest=True, sorted=True)[1]
                #     clz_predict = predicts.squeeze(0).cpu().numpy()[0] + task_index * 20
                #     batch_predicts.append(clz_predict)
                
                # y_pred.append(batch_predicts)
                # y_true.append(y.cpu().numpy())
                
                for ia, ix, iy in zip(a, x, y):                    
                    sum_entropies = []
                    min_entropies = []
                    sum_energies = []
                    min_energies = []
                    sum_margins = []
                    sum_wassersteins = []
                    sum_symmetric_kls = []
                    
                    for i in range(len(self.backbones)):
                        self.model.program.load_state_dict(self.programs[i], strict=True)
                        self.model.backbone.load_state_dict(self.backbones[i], strict=False)
                        self.model.head.load_state_dict(self.heads[i], strict=True)
                        
                        logits = self.model(ix.unsqueeze(0)).squeeze(0)
                        aug_logits = self.model(ia.unsqueeze(0)).squeeze(0)
                        
                        # print(f"Logits shape: {logits.shape}")
                        
                        entropy_loss = entropy(logits)
                        energy_loss = energy(logits)
                        margin_loss = margin(logits)
                        wasserstein_loss = wasserstein(logits, aug_logits)
                        symmetric_kl_loss = symmetric_kl(logits, aug_logits)
                        
                        sum_entropies.append(entropy_loss.sum().item())
                        min_value, min_index = torch.min(entropy_loss, dim=0)
                        min_entropies.append((min_value.item(), min_index.item()))
                        
                        sum_energies.append(energy_loss.sum().item())
                        min_value, min_index = torch.min(energy_loss, dim=0)
                        min_energies.append((min_value.item(), min_index.item()))
                        
                        sum_margins.append(margin_loss)
                        sum_wassersteins.append(wasserstein_loss.item())
                        sum_symmetric_kls.append(symmetric_kl_loss.item())
                    
                    # print(f"True label: {iy}")
                    # print(f"Sum entropies: {sum_entropies}, Min entropies: {min_entropies}")
                    # print(f"Sum energies: {sum_energies}, Min energies: {min_energies}")
                    # print(f"Sum margins: {sum_margins}")
                    # print(f"Sum wassersteins: {sum_wassersteins}")
                    # print(f"Sum symmetric_kls: {sum_symmetric_kls}")
                    
                    task_idx = (iy // 20).item()
                    sum_entropy_pred_task_idx = sum_entropies.index(min(sum_entropies))
                    min_entropy_pred_task_idx = min_entropies.index(min(min_entropies, key=lambda x: x[0]))
                    sum_energy_pred_task_idx = sum_energies.index(min(sum_energies))
                    min_energy_pred_task_idx = min_energies.index(min(min_energies, key=lambda x: x[0]))
                    margin_pred_task_idx = sum_margins.index(min(sum_margins))
                    wasserstein_pred_task_idx = sum_wassersteins.index(min(sum_wassersteins))
                    symmetric_kl_pred_task_idx = sum_symmetric_kls.index(min(sum_symmetric_kls))
                    # print(f"Task idx: {task_idx} - {sum_entropy_pred_task_idx} - {min_entropy_pred_task_idx} - {sum_energy_pred_task_idx} - {min_energy_pred_task_idx} - {margin_pred_task_idx} - {wasserstein_pred_task_idx} - {symmetric_kl_pred_task_idx}")
                    
                    task_true.append(task_idx)
                    sum_ent_pred.append(sum_entropy_pred_task_idx)
                    min_ent_pred.append(min_entropy_pred_task_idx)
                    sum_ene_pred.append(sum_energy_pred_task_idx)
                    min_ene_pred.append(min_energy_pred_task_idx)
                    sum_mar_pred.append(margin_pred_task_idx)
                    sum_was_pred.append(wasserstein_pred_task_idx)
                    sum_sym_pred.append(symmetric_kl_pred_task_idx)
            
        task_true = np.array(task_true)
        sum_ent_pred = np.array(sum_ent_pred)
        min_ent_pred = np.array(min_ent_pred)
        sum_ene_pred = np.array(sum_ene_pred)
        min_ene_pred = np.array(min_ene_pred)
        sum_mar_pred = np.array(sum_mar_pred)
        sum_was_pred = np.array(sum_was_pred)
        sum_sym_pred = np.array(sum_sym_pred)
        
        acc_sum_ent = (task_true == sum_ent_pred).sum() / len(task_true)
        acc_min_ent = (task_true == min_ent_pred).sum() / len(task_true)
        acc_sum_ene = (task_true == sum_ene_pred).sum() / len(task_true)
        acc_min_ene = (task_true == min_ene_pred).sum() / len(task_true)
        acc_sum_mar = (task_true == sum_mar_pred).sum() / len(task_true)
        acc_sum_was = (task_true == sum_was_pred).sum() / len(task_true)
        acc_sum_sym = (task_true == sum_sym_pred).sum() / len(task_true)
        
        print(f"Sum Entropy Acc: {acc_sum_ent:.2f}")
        print(f"Min Entropy Acc: {acc_min_ent:.2f}")
        print(f"Sum Energy Acc: {acc_sum_ene:.2f}")
        print(f"Min Energy Acc: {acc_min_ene:.2f}")
        print(f"Margin Acc: {acc_sum_mar:.2f}")
        print(f"Wasserstein Acc: {acc_sum_was:.2f}")
        print(f"Symmetric KL Acc: {acc_sum_sym:.2f}")
                    
            
        # y_pred = np.concatenate(y_pred)
        # y_true = np.concatenate(y_true)
        
        # acc_total, grouped = accuracy(y_pred.T, y_true, self._class_increments)
        # logger.info(f"Total Acc: {acc_total:.2f}, Grouped Acc: {grouped}")

        # self.accuracy_matrix.append(grouped)

        # num_tasks = len(self.accuracy_matrix)
        # accuracy_matrix = np.zeros((num_tasks, num_tasks))
        # for i in range(num_tasks):
        #     for j in range(i + 1):
        #         accuracy_matrix[i, j] = self.accuracy_matrix[i][j]

        # faa, ffm = compute_metrics(accuracy_matrix)
        # logger.info(f"Final Average Accuracy (FAA): {faa:.2f}")
        # logger.info(f"Final Forgetting Measure (FFM): {ffm:.2f}")


    def finetune(self, train_loader):
        epochs = config["fine_tune_train_epochs"]
        weight_decay = 5e-4
        min_lr = 0.0
        optimizer = optim.SGD(self.model.parameters(), lr=1e-2, momentum=0.9, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)

        pbar = tqdm(range(epochs))
        for _, epoch in enumerate(pbar):
            self.model.train()
            total_loss, total, correct = 0, 0, 0

            for i, (_, a, x, y) in enumerate(train_loader):
                x, y = x.cuda(), y.cuda()
                y = torch.where(y - self._known_classes >= 0, y - self._known_classes, -100)
                logits = self.model(x)[:, -20:]

                loss = F.cross_entropy(logits, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                correct += (logits.argmax(dim=1) == y).sum().item()
                total += len(y)

                info = f"Task {self.cur_task}, Epoch {epoch}, Samples: {total * 100 / len(train_loader.dataset):.2f}, Loss: {total_loss / total:.4f}, Acc: {correct * 100 / total:.2f}"
                pbar.set_description(info)

            scheduler.step()
            
        logger.info(f"Train accuracy: {correct * 100 / total:.2f}")
        
        params = {}
        for name, param in self.model.backbone.named_parameters():
            if "ssf_" in name:
                params[name] = copy.deepcopy(param)
        self.backbones.append(params)
        self.programs.append(copy.deepcopy(self.model.program.state_dict()))
        self.heads.append(copy.deepcopy(self.model.head.state_dict()))


def set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

for seed in [1993]:
    logger.info(f"Seed: {seed}")
    set_random(1)

    num_classes = 200
    num_init_cls = 20
    data_manager = DataManager("imagenetr", True, seed, 20, 20, False)

    for epoch in [10]:
        config.update(
            {
                "fine_tune_train_batch_size": 64,
                "fine_tune_train_epochs": epoch,
            }
        )
        logger.info(f"{' | '.join('%s: %s' % item for item in config.items())}")

        learner = Learner()
        learner.learn(data_manager)