import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
import numpy as np
import random
import timm
from tqdm import tqdm
import gc
import operator
import os
import logging
import copy
import matplotlib.pyplot as plt

from petl import vision_transformer_ssf
from utils.data_manager import DataManager
from utils.toolkit import count_parameters

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
        
        # self.G = nn.Sequential(
        #     nn.Conv2d(img_shape[0], 16, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Flatten(),
        #     nn.Linear(16, 1),
        #     nn.Sigmoid()
        # )
    
    def forward(self, x):
        # alpha = self.G(x)
        B, C, H, W = x.size()
        if C != self.img_shape[0]:
            x = x.repeat(1, self.img_shape[0] // C, 1, 1)
        x_scaled = F.interpolate(x, size=self.mask_size, mode='bicubic', align_corners=False)
        background = torch.zeros(B, *self.img_shape).to(x.device)
        background[:, :, 
                   (self.img_shape[1] - self.mask_size[0])//2:(self.img_shape[1] + self.mask_size[0])//2,
                   (self.img_shape[2] - self.mask_size[1])//2:(self.img_shape[2] + self.mask_size[1])//2] = x_scaled

        # x_adv = background + alpha.view(-1, 1, 1, 1) * torch.tanh(self.W * self.M)
        x_adv = background + torch.sigmoid(self.W * self.M)
        return x_adv

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.program = Program(mask_size=150)
        # self.net = timm.create_model("resnet50", pretrained=True, num_classes=0)
        self.net = timm.create_model("vit_base_patch16_224_ssf", pretrained=True, num_classes=0)
        self.net.head = nn.Linear(self.net.num_features, 20)
        self.freeze_backbone()
    
    def freeze_backbone(self):
        for name, param in self.net.named_parameters():
            if "head" not in name and "ssf_" not in name:
                param.requires_grad_(False)
    
    def normalize(self, x_adv):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x_adv.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x_adv.device)
        x_adv = (x_adv - mean) / std
        return x_adv
    
    def label_mapping(self, y_adv):
        # B, C = y_adv.size()
        # y_adv = y_adv.view(B, 10, C // 10)
        # y_adv = y_adv.sum(dim=2)
        # y_adv = y_adv[:, :200]
        return y_adv
    
    def forward(self, x):
        x_adv = self.program(x)
        x_adv = self.normalize(x_adv)
        y_adv = self.net(x_adv)
        # y_adv = self.label_mapping(y_adv)
        return y_adv
    
    def __repr__(self):
        trainable_params = count_parameters(self, trainable=True)
        total_params = count_parameters(self)
        return f"Model(trainable_params={trainable_params}, total_params={total_params}, percentage={trainable_params * 100 / total_params:.2f})"

class Learner(object):
    def __init__(self):
        self.data_manager = DataManager("imagenetr", True, seed, 0, 20, False)
        self.model = Model().to(device)
        
        self.trainset = self.data_manager.get_dataset(np.arange(0, 20), source="train", mode="train_adv")
        self.train_loader = DataLoader(self.trainset, batch_size=64, shuffle=True, num_workers=4)
        self.known_classes = 0
    
    def update(self):
        self.known_classes += 20
        self.trainset = self.data_manager.get_dataset(np.arange(20, 40), source="train", mode="train_adv")
        self.train_loader = DataLoader(self.trainset, batch_size=64, shuffle=True, num_workers=4)
        
        self.model.net.head = nn.Linear(self.model.net.num_features, 20).cuda()
    
    def compute_loss(self, y_adv, y):
        nll_loss = F.nll_loss(F.log_softmax(y_adv, dim=1), y)
        lambda_reg = 1e-2
        # reg_loss = (torch.linalg.vector_norm(self.model.program.W.view(-1), ord=2) - 0.5) ** 2
        # reg_loss = torch.mean((self.model.program.W - 0.5) ** 2)
        reg_loss = lambda_reg * torch.linalg.vector_norm(self.model.program.W.view(-1), ord=2) ** 2
        total_loss = nll_loss + reg_loss
        return {"nll_loss": nll_loss, "reg_loss": reg_loss, "total_loss": total_loss}
    
    def visualize(self):
        self.model.program.eval()
        batch = next(iter(self.train_loader))
        idx, x, y = batch
        x = x.to(device)
        x_adv = self.model.program(x)
        
        # x = (x - x.min()) / (x.max() - x.min() + 1e-6)
        # x_adv = (x_adv - x_adv.min()) / (x_adv.max() - x_adv.min() + 1e-6)

        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        for i in range(4):
            for j in range(4):
                img = x[i * 4 + j] if i < 2 else x_adv[(i - 2) * 4 + j]  # First 2 rows: x, Last 2 rows: x_adv
                img = img.permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)
                axes[i, j].imshow(img)
                axes[i, j].axis("off")

        plt.tight_layout()
        plt.savefig("adv.png")
    
    def train(self):
        self.model.train()
        print(self.model)
        
        epochs = 20
        min_lr = 0.0
        
        # params = [
        #     {"params": self.model.program.parameters(), "lr": 1e-2},
        #     {"params": self.model.net.parameters(), "lr": 1e-2}
        # ]
        optimizer = optim.SGD(self.model.parameters(), momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
        
        pbar = tqdm(range(epochs))
        for epoch in pbar:
            total_loss, total_nll_loss, total_reg_loss = 0, 0, 0
            correct, total = 0, 0
            
            for _, (_, x, y) in enumerate(self.train_loader):
                x, y = x.to(device), y.to(device)
                y_adv = self.model(x)
                # y_task = y // 20
                y_task = torch.where(y - self.known_classes >= 0, y - self.known_classes, -100)
                # print(y_task)
                
                loss = self.compute_loss(y_adv, y_task)
                
                optimizer.zero_grad()
                loss["total_loss"].backward()
                optimizer.step()
                
                total_loss += loss["total_loss"].item()
                total_nll_loss += loss["nll_loss"].item()
                total_reg_loss += loss["reg_loss"].item()
                correct += (y_adv.argmax(dim=1) == y_task).sum().item()
                total += y.size(0)
                
                info = f"Epoch {epoch+1}: Loss {total_loss/total:.4f}, Nll Loss {total_nll_loss/total:.4f}, Reg Loss {total_reg_loss/total:.4f}, Accuracy {100 * correct/total:.2f}%"
                pbar.set_description(info)
            
            scheduler.step()
                
                
learner = Learner()
# learner.visualize()
learner.train()
learner.update()
learner.train()