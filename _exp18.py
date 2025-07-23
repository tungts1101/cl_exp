# sequential peft + adv program
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
from timm.models.layers.weight_init import trunc_normal_


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
    "fine_tune_train_epochs": 1,
    "merge_trim_top_k": 20,
    "merge_lambda": 1.0
}

# with open("result.txt", "a") as f:
#     f.write(f"{' | '.join('%s: %s' % item for item in config.items())}\n")

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
    
    def forward(self, x):
        B, C, H, W = x.size()
        if C != self.img_shape[0]:
            x = x.repeat(1, self.img_shape[0] // C, 1, 1)
        x_scaled = F.interpolate(x, size=self.mask_size, mode='bicubic', align_corners=False)
        background = torch.zeros(B, *self.img_shape).to(x.device)
        background[:, :, 
                   (self.img_shape[1] - self.mask_size[0])//2:(self.img_shape[1] + self.mask_size[0])//2,
                   (self.img_shape[2] - self.mask_size[1])//2:(self.img_shape[2] + self.mask_size[1])//2] = x_scaled
        x_adv = background + torch.sigmoid(self.W * self.M)
        return x_adv

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.program = Program(mask_size=150)
        self.backbone = timm.create_model(
            "vit_base_patch16_224_ssf", pretrained=True, num_classes=0
        )
        self.freeze_backbone()
        self._backbone = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        self._backbone.requires_grad_(False)
        self.head = ContinualLearnerHead(self.backbone.num_features * 2, 20).to(device)
        
        self.init_weights()

    def init_weights(self):
        for name, param in self.backbone.named_parameters():
            if "ssf_scale_" in name:
                nn.init.ones_(param)
            elif "ssf_shift_" in name:
                nn.init.zeros_(param)
    
    def freeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            if ("ssf_scale_" not in name) and ("ssf_shift_" not in name):
                param.requires_grad = False
    
    def normalize(self, x_adv):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x_adv.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x_adv.device)
        x_adv = (x_adv - mean) / std
        return x_adv
    
    def forward(self, x):
        # x_adv = self.program(x)
        x_adv = x
        x_adv = self.normalize(x_adv)
        y_adv = torch.cat((self.backbone(x_adv), self._backbone(x_adv)), dim=1)
        y_adv = self.head(y_adv)['logits']
        return y_adv
    
    def __repr__(self):
        trainable_params = count_parameters(self, trainable=True)
        total_params = count_parameters(self)
        return f"Model(trainable_params={trainable_params}, total_params={total_params}, percentage={trainable_params * 100 / total_params:.2f})"


class Learner:
    def __init__(self):
        self._known_classes = 0
        self._classes_seen_so_far = 0
        self._class_increments = []
        self.model = Model().to(device)

    def learn(self, data_manager):
        num_tasks = data_manager.nb_tasks - 1

        for task in range(num_tasks):
            self.before_task(task, data_manager)
            
            trainset = data_manager.get_dataset(
                np.arange(self._known_classes, self._classes_seen_so_far),
                source="train", mode="train_adv")
            train_loader = DataLoader(trainset, batch_size=config["fine_tune_train_batch_size"], shuffle=True, num_workers=4)
            testset = data_manager.get_dataset(
                np.arange(self._known_classes, self._classes_seen_so_far),
                source="test", mode="test_adv")
            test_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)
            self.finetune(train_loader, test_loader, task, epochs=config["fine_tune_train_epochs"])
            
            testset_CPs = data_manager.get_dataset(
                np.arange(0, self._classes_seen_so_far),
                source="test", mode="test")
            test_loader_CPs = DataLoader(testset_CPs, batch_size=128, shuffle=False, num_workers=4)
            
            self.eval(test_loader_CPs)
            self.after_task()
    
    def before_task(self, task, data_manager):
        print(self.model)
        self._classes_seen_so_far = (self._known_classes + data_manager.get_task_size(task + 1))
        self._class_increments.append((self._known_classes, self._classes_seen_so_far - 1))
                
    def after_task(self):
        self.model.head.update(self._classes_seen_so_far - self._known_classes, freeze_old=True)
        self.model.head.to(device)
        self._known_classes = self._classes_seen_so_far
    
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
    
    def compute_loss(self, y_predict, y_true):
        # nll_loss = F.nll_loss(F.log_softmax(y_predict, dim=1), y_true)
        # lambda_reg = 1e-4
        # reg_loss = torch.linalg.vector_norm(self.model.program.W.view(-1), ord=2) ** 2
        # total_loss = nll_loss + lambda_reg * reg_loss
        # return {"nll_loss": nll_loss, "reg_loss": reg_loss, "total_loss": total_loss}
        
        loss = F.cross_entropy(y_predict, y_true)
        return {"nll_loss": loss, "reg_loss": 0, "total_loss": loss}
    
    def finetune(self, train_loader, test_loader, task, epochs=20):
        weight_decay = 5e-4
        min_lr = 0.0
        
        base_lr = 1e-2
        program_lr = 1
        backbone_lr = 1
        head_lr = 1
        params = [
            {"params": self.model.program.parameters(), "lr": base_lr * program_lr},
            {"params": self.model.backbone.parameters(), "lr": base_lr * backbone_lr},
            {"params": self.model.head.parameters(), "lr": base_lr * head_lr}
        ]
        optimizer = optim.SGD(params, momentum=0.9, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
        
        pbar = tqdm(range(epochs))
        for _, epoch in enumerate(pbar):
            self.model.train()
            total_loss, nll_loss, reg_loss = 0, 0, 0
            correct, total = 0, 0
            
            for i, (_, x, y) in enumerate(train_loader):
                x, y = x.cuda(), y.cuda()
                y = torch.where(y - self._known_classes >= 0, y - self._known_classes, -100)
                logits = self.model(x)[:, -20:]
                
                loss = self.compute_loss(logits, y)
                
                optimizer.zero_grad()
                loss["total_loss"].backward()
                optimizer.step()
                
                total_loss += loss["total_loss"].item()
                nll_loss += loss["nll_loss"].item()
                reg_loss += loss["reg_loss"]
                correct += (logits.argmax(dim=1) == y).sum().item()
                total += len(y)
                
                info = f"Task {task}, Epoch {epoch}, Total loss: {total_loss / total:.4f}, NLL loss: {nll_loss / total:.4f}, Reg loss: {reg_loss / total:.4f}, Acc: {correct * 100 / total:.2f}"
                pbar.set_description(info)
                
            scheduler.step()
            

learner = Learner()
learner.learn(data_manager)