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
from _exp import ContinualLearnerHead, RandomProjectionHead, plot_heatmap, Buffer
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

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # vit_base_patch16_224_in21k_ssf
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

def sum_energy(logits, temperature=1.0):
    return -torch.logsumexp(logits / temperature, dim=2)

def max_logits(logits):
    return torch.max(logits, dim=2)[0]

def contrastive_loss(logits, labels, temperature=1.0):
    logits = F.normalize(logits, dim=1)
    similarity_matrix = logits @ logits.T
    
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float()
    
    exp_sim = torch.exp(similarity_matrix / temperature)
    pos_sim = exp_sim * mask
    
    sump_exp_sim = exp_sim.sum(dim=1, keepdim=True)
    
    contrastive_loss = -torch.log((pos_sim.sum(dim=1) + 1e-6) / (sump_exp_sim + 1e-6))
    return contrastive_loss.mean()

@torch.enable_grad()
def odin_detection(model, x, temperature=1000, epsilon=0.002, device='cuda'):
    x = x.to(device).requires_grad_(True)

    logits = model(x)
    softmax_probs = F.softmax(logits / temperature, dim=1)
    pred_label = torch.argmax(softmax_probs, dim=1)
    loss = -torch.log(softmax_probs[0, pred_label]).sum()
    loss.backward()

    with torch.no_grad():
        x_adv = x - epsilon * torch.sign(x.grad)
        x_adv = torch.clamp(x_adv, 0, 1)  # Ensure valid pixel range

        logits_adv = model(x_adv)
    
        return logits_adv

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
        self.buffer = Buffer(50)

    def learn(self, data_manager):
        self.data_manager = data_manager
        num_tasks = data_manager.nb_tasks

        for task in range(num_tasks):
            self.before_task(task, data_manager)
            
            trainset = data_manager.get_dataset(
                np.arange(self._known_classes, self._classes_seen_so_far), source="train", mode="train")
            train_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
            self.finetune(train_loader)

            # test_set = self.data_manager.get_dataset(
            #     np.arange(0, self._classes_seen_so_far), source="test", mode="test")
            # test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4)
            # self.eval(test_loader)
            
            self.after_task()
        
        # self.ood_train()
        
        test_set = self.data_manager.get_dataset(
            np.arange(0, 200), source="test", mode="test")
        test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=4)
        self.eval(test_loader)

    def before_task(self, task, data_manager):
        print(self.model)
        self._classes_seen_so_far = self._known_classes + data_manager.get_task_size(task)
        self._class_increments.append((self._known_classes, self._classes_seen_so_far - 1))
        self.cur_task = task

    def after_task(self):
        self.model.reset()
        self._known_classes = self._classes_seen_so_far

    def ood_train(self):
        self.model.train()
        
        for task_idx in range(len(self.backbones)):
            print(f"OOD Train {task_idx}")
            self.model.backbone.load_state_dict(self.backbones[task_idx], strict=False)
            self.model.head.load_state_dict(self.heads[task_idx], strict=True)
            
            train_loader = DataLoader(self.buffer, batch_size=64, shuffle=True, num_workers=4)
            optimizer = optim.SGD(self.model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0)
            epochs = 1
            
            id_margin = -1
            ood_margin = -10
            temperature = 1.0
            
            pbar = tqdm(range(epochs))
            for _, epoch in enumerate(pbar):
                self.model.train()
                total_loss, total, correct = 0, 0, 0

                for _, (x, y) in enumerate(train_loader):
                    x, y = x.cuda(), y.cuda()
                    is_id = (y >= task_idx * 20) & (y < (task_idx + 1) * 20)
                    is_ood = ~is_id
                    
                    logits = self.model(x)

                    energy_scores = -temperature * torch.logsumexp(logits / temperature, dim=1)
                    
                    loss = torch.tensor(0.0).cuda()
                    
                    if is_id.any():
                        energy_loss_id = F.relu(energy_scores[is_id] - id_margin).mean()
                        loss += energy_loss_id
                    if is_ood.any():
                        energy_loss_ood = F.relu(ood_margin - energy_scores[is_ood]).mean()
                        loss += energy_loss_ood
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    total += len(y)

                    info = f"Epoch {epoch}, %: {total * 100 / len(train_loader.dataset):.2f}, Loss: {total_loss / total:.4f}"
                    pbar.set_description(info)

                scheduler.step()
                
            params = {}
            for name, param in self.model.backbone.named_parameters():
                if "ssf_" in name:
                    params[name] = copy.deepcopy(param)
            self.backbones[task_idx] = params
            self.heads[task_idx] = copy.deepcopy(self.model.head.state_dict())

    def eval(self, test_loader):
        self.model.eval()
        task_true = []
        sum_ene_pred = []
        max_log_pred = []
        
        top2_sum_ene = []
        top2_max_log = []
        sum_ene = []
        
        # with torch.no_grad():
        for _, (_, a, x, y) in tqdm(enumerate(test_loader)):
            x, y = x.cuda(), y.cuda()
            
            logits = []
            for i in range(len(self.backbones)):
                self.model.backbone.load_state_dict(self.backbones[i], strict=False)
                self.model.head.load_state_dict(self.heads[i], strict=True)
                # self.model.load_state_dict(torch.load(f"logs/checkpoints/{i}_model.pth"))
                
                # logits.append(self.model(x))
                
                logits.append(odin_detection(self.model, x))
            
            logits = torch.stack(logits, dim=1)
            
            sum_energy_score = sum_energy(logits, temperature=1)
            sum_energy_pred = sum_energy_score.argmin(dim=1)
            
            max_logits_score = max_logits(logits)
            max_logits_pred = max_logits_score.argmax(dim=1)
            
            task_true.append((y // 20).cpu())
            sum_ene_pred.append(sum_energy_pred.cpu())
            max_log_pred.append(max_logits_pred.cpu())
            
            # top2_ene, _ = torch.topk(sum_energy_score, 2, dim=1, largest=False)
            # diff_ene = top2_ene[:, 1] - top2_ene[:, 0]
            # top2_sum_ene.append(diff_ene.cpu())
            
            # top2_log, _ = torch.topk(max_logits_score, 2, dim=1, largest=True)
            # diff_log = top2_log[:, 0] - top2_log[:, 1]
            # top2_max_log.append(diff_log.cpu())
            
            # sum_ene.append(top2_ene[:, 0].cpu())
        
        task_true = np.concatenate(task_true)
        sum_ene_pred = np.concatenate(sum_ene_pred)
        max_log_pred = np.concatenate(max_log_pred)
        
        acc_sum_ene = (task_true == sum_ene_pred).mean()
        acc_max_log = (task_true == max_log_pred).mean()
        
        print(f"Sum Energy Acc: {acc_sum_ene:.2f}, Max Logits Acc: {acc_max_log:.2f}")
            
            # plot_heatmap(task_true, sum_ene_pred)
            # plot_heatmap(task_true, max_log_pred)
            
            # top2_sum_ene = np.concatenate(top2_sum_ene)
            # top2_max_log = np.concatenate(top2_max_log)
            # sum_ene = np.concatenate(sum_ene)
            
            # mis_dist = {}
            # log_mis_dist = {}
            # ene_dist = {}
            # for i, task_id in enumerate(task_true):
            #     if task_id not in mis_dist:
            #         mis_dist[task_id] = []
            #     if task_id not in log_mis_dist:
            #         log_mis_dist[task_id] = []
            #     if task_id not in ene_dist:
            #         ene_dist[task_id] = []
                
            #     if task_id != sum_ene_pred[i]:
            #         mis_dist[task_id].append(top2_sum_ene[i])
                
            #     if task_id != max_log_pred[i]:
            #         log_mis_dist[task_id].append(top2_max_log[i])
                
            #     if task_id == sum_ene_pred[i]:
            #         ene_dist[task_id].append(sum_ene[i])
            
            # mis_avg_dis = {}
            # log_mis_avg_dis = {}
            # ene_avg_dis = {}
            # for task_id in mis_dist:
            #     mis_avg_dis[task_id] = np.mean(mis_dist[task_id])
            # for task_id in log_mis_dist:
            #     log_mis_avg_dis[task_id] = np.mean(log_mis_dist[task_id])
            # for task_id in ene_dist:
            #     ene_avg_dis[task_id] = np.mean(ene_dist[task_id])
            
            # print(mis_avg_dis)
            # print(log_mis_avg_dis)
            # print(ene_avg_dis)

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
                a, x, y = a.cuda(), x.cuda(), y.cuda()
                y = torch.where(y - self._known_classes >= 0, y - self._known_classes, -100)
                
                logits = self.model(x)

                loss = F.cross_entropy(logits, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                correct += (logits.argmax(dim=1) == y).sum().item()
                total += len(y)

                info = f"Task {self.cur_task}, Epoch {epoch}, %: {total * 100 / len(train_loader.dataset):.2f}, Loss: {total_loss / total:.4f}, Acc: {correct * 100 / total:.2f}"
                pbar.set_description(info)

            scheduler.step()
        
        params = {}
        for name, param in self.model.backbone.named_parameters():
            if "ssf_" in name:
                params[name] = copy.deepcopy(param)
        self.backbones.append(params)
        self.heads.append(copy.deepcopy(self.model.head.state_dict()))
        
        for _, (_, _, x, y) in enumerate(train_loader):
            self.buffer.add(x, y)
        
        self.buffer.update()
        
        # os.makedirs("logs/checkpoints", exist_ok=True)
        # torch.save(self.model.state_dict(), f"logs/checkpoints/{self.cur_task}_model.pth")


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