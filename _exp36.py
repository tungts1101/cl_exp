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
from merging.task_vectors import TaskVector, merge_max_abs
from merging.ties import merge_methods, state_dict_to_vector, vector_to_state_dict
import gc
import operator


os.makedirs("logs/checkpoints", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

def setup_logger(log_file=f'logs/{timestamp}_exp36.log'):
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

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # self.backbone = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
        # self.backbone = timm.create_model("resnet50", pretrained=True, num_classes=0)
        # self.backbone = timm.create_model("resnet152", pretrained=True, num_classes=0)
        # self.head = nn.Sequential(
        #     nn.Linear(2048, 512),
        #     nn.LayerNorm(512),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(512, 20)
        # )
        
        self.backbone = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        self.head = nn.Linear(768, 20)
        
        # self.backbone.apply(init_weights)
        self.head.apply(init_weights)
    
    def get_backbone_trainable_params(self):
        params = {}
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                params[name] = param
        return params
    
    def forward(self, x):
        f = self.backbone(x)
        y = self.head(f)
        return y
    
    def __repr__(self):
        trainable_params = count_parameters(self, trainable=True)
        total_params = count_parameters(self)
        return f"Model(trainable_params={trainable_params}, total_params={total_params}, percentage={trainable_params * 100 / total_params:.2f})"
    
os.makedirs("/media/ellen/HardDisk/cl/logs/checkpoints", exist_ok=True) 
   
def backbone_checkpoint(task):
    return f"/media/ellen/HardDisk/cl/logs/checkpoints/ft_imagenetr_backbone_{task}.pt"

def head_checkpoint(task):
    return f"/media/ellen/HardDisk/cl/logs/checkpoints/ft_imagenetr_head_{task}.pt"

def backbone_base():
    return "/media/ellen/HardDisk/cl/logs/checkpoints/ft_imagenetr_backbone_base.pt"

class Learner:
    def __init__(self):
        self._known_classes = 0
        self._classes_seen_so_far = 0
        self._class_increments = []
        self.model = Model()
        
        self.model.cuda()
        
        self.accuracy_matrix = []
        self.cur_task = -1
    
    def learn(self, data_manager):
        self.data_manager = data_manager
        num_tasks = data_manager.nb_tasks

        for task in range(num_tasks):
            print(f"Starting task {task}")
            
            self.before_task(task, data_manager)
            
            trainset = data_manager.get_dataset(
                np.arange(self._known_classes, self._classes_seen_so_far), source="train", mode="train")
            train_loader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
            self.finetune(train_loader)

            test_set = self.data_manager.get_dataset(
                np.arange(0, self._classes_seen_so_far), source="test", mode="test")
            test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4)
            self.eval(test_loader)
            
            self.after_task()
            
    def before_task(self, task, data_manager):
        self._classes_seen_so_far = self._known_classes + data_manager.get_task_size(task)
        self._class_increments.append((self._known_classes, self._classes_seen_so_far - 1))
        self.cur_task = task

    def after_task(self):
        self._known_classes = self._classes_seen_so_far
    
    def eval(self, test_loader):
        self.model.eval()
        
        y_true, y_pred = [], []
        with torch.no_grad():
            for _, (_, a, x, y) in tqdm(enumerate(test_loader)):
                x, y = x.cuda(), y.cuda()
                
                logits = []
                for i in range(self.cur_task + 1):
                    self.model.backbone.load_state_dict(torch.load(backbone_checkpoint(i)), strict=True)
                    self.model.head.load_state_dict(torch.load(head_checkpoint(i)), strict=True)
                        
                    logits.append(self.model(x))
                        
                logits = torch.stack(logits, dim=1)
                max_logit_values, max_logit_indices = torch.max(logits, dim=-1)
                
                temperature = 1
                energy = -torch.logsumexp(logits/ temperature, dim=-1)
                
                task_indices_offset = torch.arange(self.cur_task+1, dtype=torch.long, device=device) * 20
                max_logit_indices += task_indices_offset
                
                top_values, top_indices = torch.topk(energy, 1, dim=-1, largest=False)
                predicts = max_logit_indices[torch.arange(max_logit_indices.shape[0]), top_indices[:, 0]]
                
                y_pred.append(predicts.cpu().numpy())
                y_true.append(y.cpu().numpy())
        
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        acc_total, grouped = accuracy(y_pred.T, y_true, self._class_increments)
        logger.info(f"Total Acc: {acc_total:.2f}, Grouped Acc: {grouped}")

        self.accuracy_matrix.append(grouped)

        num_tasks = len(self.accuracy_matrix)
        accuracy_matrix = np.zeros((num_tasks, num_tasks))
        for i in range(num_tasks):
            for j in range(i + 1):
                accuracy_matrix[i, j] = self.accuracy_matrix[i][j]

        faa, ffm = compute_metrics(accuracy_matrix)
        logger.info(f"Final Average Accuracy (FAA): {faa:.2f}")
        logger.info(f"Final Forgetting Measure (FFM): {ffm:.2f}")

    def finetune(self, train_loader):
        task_model = Model()
        task_model.cuda()
        print(task_model)
        
        epochs = config["fine_tune_train_epochs"]
        weight_decay = 5e-4
        min_lr = 0.0
        # params = [
        #     {"params": task_model.backbone.parameters(), "lr": 1e-2, weight_decay: weight_decay},
        #     {"params": task_model.head.parameters(), "lr": 1e-2, weight_decay: weight_decay},
        # ]
        optimizer = optim.AdamW(task_model.parameters(), lr=3e-4, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)

        pbar = tqdm(range(epochs))
        for _, epoch in enumerate(pbar):
            task_model.train()
            total_loss, total, correct = 0, 0, 0

            for i, (_, a, x, y) in enumerate(train_loader):
                a, x, y = a.cuda(), x.cuda(), y.cuda()
                y = torch.where(y - self._known_classes >= 0, y - self._known_classes, -100)
                
                logits = task_model(x)
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
        
        task_model.cpu()
        torch.save(task_model.backbone.state_dict(), backbone_checkpoint(self.cur_task))
        torch.save(task_model.head.state_dict(), head_checkpoint(self.cur_task))

        del task_model
        gc.collect()
        torch.cuda.empty_cache()

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

    for dataset in ["imagenetr"]:
        data_manager = DataManager(dataset, True, seed, 20, 20, False)

        for epoch in [20]:
            config.update(
                {
                    "fine_tune_train_batch_size": 64,
                    "fine_tune_train_epochs": epoch,
                }
            )
            logger.info(f"{' | '.join('%s: %s' % item for item in config.items())}")

            learner = Learner()
            learner.learn(data_manager)