import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import random
import timm
from tqdm import tqdm
import gc
import operator
import os
import logging

from utils.data_manager import DataManager
from utils.toolkit import count_parameters, accuracy
from petl import vision_transformer_ssf
from experiments.merge_utils import ties_merge
from _exp import ContinualLearnerHead


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1993

def set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random(seed)

num_classes = 200
num_init_cls = 20
data_manager = DataManager("imagenetr", True, seed, 0, num_init_cls, False)

config = {
    "fine_tune_train_batch_size": 64,
    "fine_tune_train_epochs": 20,
    "merge_trim_top_k": 100,
    "merge_lambda": 1.0
}

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.backbone = timm.create_model(
            "vit_base_patch16_224_ssf", pretrained=True, num_classes=0
        )
        self.freeze_backbone()
        self.head = ContinualLearnerHead(self.backbone.num_features, num_init_cls, with_norm=False)
        
        self.init_weights()
    
    def init_weights(self):
        for name, param in self.backbone.named_parameters():
            if "ssf_scale_" in name:
                nn.init.ones_(param)
            elif "ssf_shift_" in name:
                nn.init.zeros_(param)
    
    def forward(self, x):
        f = self.backbone(x)
        y = self.head(f)['logits']
        return y
    
    def freeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            if ("ssf_scale_" not in name) and ("ssf_shift_" not in name):
                param.requires_grad = False
    
    def __repr__(self):
        trainable_params = count_parameters(self.backbone, trainable=True) + count_parameters(self.head.heads, trainable=True)
        total_params = count_parameters(self.backbone) + count_parameters(self.head.heads)
        return f"Model(trainable_params={trainable_params}, total_params={total_params}, percentage={trainable_params * 100 / total_params:.2f})"

class Learner:
    def __init__(self):
        self._known_classes = 0
        self._classes_seen_so_far = 0
        self._class_increments = []
        self.model = Model().to(device)
        self.backbone_params = {}
        self.neck_params = {}
        self.head_params = {}
        
        self.saved_backbone_params = []

    def learn(self, data_manager):
        num_tasks = data_manager.nb_tasks - 1

        for task in range(num_tasks):
            self.before_task(task, data_manager)
            
            testset_CPs = data_manager.get_dataset(
                np.arange(0, self._classes_seen_so_far),
                source="test", mode="test")
            test_loader_CPs = DataLoader(testset_CPs, batch_size=128, shuffle=False, num_workers=4)
            
            backbone_path = f"logs/model_merging/saved_backbone_{self._known_classes}.pt"
            head_path = f"logs/model_merging/saved_head_{self._known_classes}.pt"
            
            # if task == 0:
            if os.path.exists(backbone_path) and os.path.exists(head_path):
                self.model.backbone.load_state_dict(torch.load(backbone_path), strict=False)
                # self.model.head.reset()
                head_params = torch.load(head_path)
                # print(head_params.keys())
                self.model.head.load_state_dict(head_params, strict=True)
                
                self.model.head.reset([0])
                
            
            self.eval(test_loader_CPs)
            self.after_task()
            
            if task >= 1:
                break
    
    def before_task(self, task, data_manager):
        print(f"Before task {task} =====")
        self._classes_seen_so_far = (self._known_classes + data_manager.get_task_size(task + 1))
        self._class_increments.append((self._known_classes, self._classes_seen_so_far - 1))
        print(f"Learn classes: {self._known_classes} - {self._classes_seen_so_far - 1}")
        print(self.model)
                
    def after_task(self):
        self.model.head.update(self._classes_seen_so_far - self._known_classes, freeze_old=True)
        self.model.head.to(device)
        self._known_classes = self._classes_seen_so_far
        print(self.model)
        print("=====================================\n")
    
    def eval(self, test_loader):
        self.model.eval()
        y_pred, y_true = [], []
        with torch.no_grad():
            for _, (_, x, y) in tqdm(enumerate(test_loader)):
                x = x.cuda()
                logits = self.model(x)
                predicts = torch.topk(logits, k=1, dim=1, largest=True, sorted=True)[1]
                y_pred.append(predicts.cpu().numpy())
                y_true.append(y.cpu().numpy())
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        acc_total, grouped = accuracy(y_pred.T[0], y_true, self._known_classes, self._class_increments)
        result = f"Acc total: {acc_total}, Acc grouped: {grouped}"
        print(result)

learner = Learner()
learner.learn(data_manager)