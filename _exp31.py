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

        self.backbone = timm.create_model("vit_base_patch16_224_ssf", pretrained=True, num_classes=0)
        self.freeze_backbone()
        self.head = ContinualLearnerHead(768, 20)
    
    def reset(self):
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
    
    def forward(self, x):
        f = self.backbone(x)
        y = self.head(f)['logits']
        return y
    
    def __repr__(self):
        trainable_params = count_parameters(self, trainable=True)
        total_params = count_parameters(self)
        return f"Model(trainable_params={trainable_params}, total_params={total_params}, percentage={trainable_params * 100 / total_params:.2f})"

def energy_score(logits, temperature=1.0):
    return -temperature * torch.logsumexp(logits / temperature, dim=0)

def energy_score_each(logits, temperature=1.0):
    return -torch.log(torch.exp(logits / temperature))

def diff_enery(logits, temperature=1.0):
    top2, _ = torch.topk(logits, 2, dim=0)
    # print(f"Diff: {top2}")
    return temperature * (torch.log(torch.exp(top2[0] / temperature) - torch.exp(top2[1] / temperature)))

def normalize(x):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
    x = (x - mean) / std
    return x

def repr_energy(repr):
    return -torch.norm(repr, p=2, dim=0)

class Learner:
    def __init__(self):
        self._known_classes = 0
        self._classes_seen_so_far = 0
        self._class_increments = []
        self.model = Model().to(device)
        self.accuracy_matrix = []
        self.cur_task = -1
        
        self.backbones = []
        self.heads = []
        
        self.program = Program(mask_size=200).cuda()
        
        self.Wrand = torch.randn(768, 10000).cuda()

    def learn(self, data_manager):
        self.data_manager = data_manager
        num_tasks = data_manager.nb_tasks

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
            
            if task == 3:
                break

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
        
        self.model.eval()
        task_true = []
        sum_ene_pred = []
        min_ene_pred = []
        dif_ene_pred = []
        rep_ene_pred = []
        
        with torch.no_grad():
            for _, (_, a, x, y) in tqdm(enumerate(test_loader)):
                a, x, y = a.cuda(), x.cuda(), y.cuda()
                
                for ia, ix, iy in zip(a, x, y):                    
                    sum_energies, min_energies = [], []
                    dif_energies = []
                    rep_energies = []
                    
                    for i in range(len(self.backbones)):
                        self.model.backbone.load_state_dict(self.backbones[i], strict=False)
                        self.model.head.load_state_dict(self.heads[i], strict=True)
                        
                        
                        f = self.model.backbone(ix.unsqueeze(0)).squeeze(0)
                        logits = self.model(ix.unsqueeze(0)).squeeze(0)
                        
                        # print(f"Logits shape: {logits.shape}")
                        
                        energy_loss = energy_score(logits, temperature=10)
                        sum_energies.append(energy_loss.sum().item())
                        
                        energy = energy_score_each(logits, temperature=1)
                        min_energies.append(energy.min().item())
                        
                        dif_ene_loss = diff_enery(logits, temperature=1)
                        dif_energies.append(dif_ene_loss.item())
                        
                        f = f @ self.Wrand
                        rep_ene = repr_energy(f)
                        rep_energies.append(rep_ene.item())
                    
                    # print(f"True label: {iy}")
                    # print(f"Sum: {sum_energies}, Min: {min_energies}, Dif: {dif_energies}, Rep: {rep_energies}")
                    task_idx = (iy // 20).item()
                    sum_value, sum_index = torch.min(torch.tensor(sum_energies), 0)
                    min_value, min_index = torch.min(torch.tensor(min_energies), 0)
                    dif_value, dif_index = torch.max(torch.tensor(dif_energies), 0)
                    rep_value, rep_index = torch.min(torch.tensor(rep_energies), 0)
                    
                    # print(f"Task idx: {task_idx} | {sum_value}, {sum_index} | {min_value}, {min_index} | {dif_value}, {dif_index} | {rep_value}, {rep_index}")
                    
                    task_true.append(task_idx)
                    sum_ene_pred.append(sum_index)
                    min_ene_pred.append(min_index)
                    dif_ene_pred.append(dif_index)
                    rep_ene_pred.append(rep_index)
                
            
        task_true = np.array(task_true)
        sum_ene_pred = np.array(sum_ene_pred)
        min_ene_pred = np.array(min_ene_pred)
        dif_ene_pred = np.array(dif_ene_pred)
        rep_ene_pred = np.array(rep_ene_pred)
        
        acc_sum_ene = (task_true == sum_ene_pred).mean()
        acc_min_ene = (task_true == min_ene_pred).mean()
        acc_dif_ene = (task_true == dif_ene_pred).mean()
        acc_rep_ene = (task_true == rep_ene_pred).mean()
        
        print(f"Sum Energy Acc: {acc_sum_ene:.2f}, Min Energy Acc: {acc_min_ene:.2f}, Dif Energy Acc: {acc_dif_ene:.2f}, Rep Energy Acc: {acc_rep_ene:.2f}")


    def finetune(self, train_loader):
        epochs = config["fine_tune_train_epochs"]
        weight_decay = 5e-4
        min_lr = 0.0
        optimizer = optim.SGD(self.model.parameters(), lr=1e-2, momentum=0.9, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
        
        # task_model = None
        # if self.cur_task > 0:
        #     task_model = Model().cuda()
        #     task_model.eval()

        pbar = tqdm(range(epochs))
        for _, epoch in enumerate(pbar):
            self.model.train()
            total_loss, total, correct = 0, 0, 0
            total_prev_energy_score = [0 for _ in range(len(self.backbones))] if self.cur_task > 0 else [0]
            total_curr_energy_score = 0

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
                
                # if task_model is not None:
                #     for j in range(len(self.backbones)):
                #         task_model.backbone.load_state_dict(self.backbones[j], strict=False)
                #         task_model.head.load_state_dict(self.heads[j], strict=True)
                #         prev_energy_score = energy_score(task_model(x)[:, -20:]).sum().item()
                #         total_prev_energy_score[j] += prev_energy_score
                    
                # curr_energy_score = energy_score(self.model(x)[:, -20:]).sum().item()
                # total_curr_energy_score += curr_energy_score

                info = f"Task {self.cur_task}, Epoch {epoch}, %: {total * 100 / len(train_loader.dataset):.2f}, Loss: {total_loss / total:.4f}, , Acc: {correct * 100 / total:.2f}"
                # info += f", Curr Energy: {total_curr_energy_score / total:.4f}"
                # for e in total_prev_energy_score:
                #     info += f", {e / total:.4f}"
                pbar.set_description(info)

            scheduler.step()
            
        logger.info(f"Train accuracy: {correct * 100 / total:.2f}")
        
        params = {}
        for name, param in self.model.backbone.named_parameters():
            if "ssf_" in name:
                params[name] = copy.deepcopy(param)
        self.backbones.append(params)
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