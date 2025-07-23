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

from utils.data_manager import DataManager
from utils.toolkit import count_parameters, accuracy
from petl import vision_transformer_ssf
from experiments.merge_utils import ties_merge, avg_merge
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
    "fine_tune_train_epochs": 20,
    "merge_trim_top_k": 20,
    "merge_lambda": 1.0,
    "lambda": 0.01
}

class Program(nn.Module):
    def __init__(self, img_shape=(3, 224, 224), mask_size=150):
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
        
        self.G = nn.Sequential(
            nn.Conv2d(img_shape[0], 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        alpha = self.G(x)
        B, C, H, W = x.size()
        if C != self.img_shape[0]:
            x = x.repeat(1, self.img_shape[0] // C, 1, 1)
        x_scaled = F.interpolate(x, size=self.mask_size, mode='bicubic', align_corners=False)
        background = torch.zeros(B, *self.img_shape).to(x.device)
        background[:, :, 
                   (self.img_shape[1] - self.mask_size[0])//2:(self.img_shape[1] + self.mask_size[0])//2,
                   (self.img_shape[2] - self.mask_size[1])//2:(self.img_shape[2] + self.mask_size[1])//2] = x_scaled

        x_perturbed = background + alpha.view(-1, 1, 1, 1) * torch.tanh(self.W * self.M)
        return x_perturbed

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.program = Program()
        self.net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.net.requires_grad_(False)
    
    def normalize(self, x_adv):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x_adv.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x_adv.device)
        x_adv = (x_adv - mean) / std
        return x_adv
    
    def label_mapping(self, y_adv):
        y_adv = y_adv[:, :10]
        y_adv = F.softmax(y_adv, dim=1)
        return y_adv
    
    def forward(self, x):
        x_adv = self.program(x)
        x_adv = self.normalize(x_adv)
        y_adv = self.net(x_adv)
        y_adv = self.label_mapping(y_adv)
        y_adv = F.softmax(y_adv, dim=1)
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
        self.class_to_task = {}
        self.programs = []

    def learn(self, data_manager):
        num_tasks = data_manager.nb_tasks - 1

        for task in range(num_tasks):
            self.before_task(task, data_manager)
            
            trainset = data_manager.get_dataset(
                np.arange(self._known_classes, self._classes_seen_so_far),
                source="train", mode="train")
            train_loader = DataLoader(trainset, batch_size=config["fine_tune_train_batch_size"], shuffle=True, num_workers=4)
            self.finetune(train_loader, task, epochs=config["fine_tune_train_epochs"])
            
            testset_CPs = data_manager.get_dataset(
                np.arange(0, self._classes_seen_so_far),
                source="test", mode="test")
            test_loader_CPs = DataLoader(testset_CPs, batch_size=128, shuffle=False, num_workers=4)
            self.programs.append(copy.deepcopy(self.model.program.state_dict()))
            
            os.makedirs("logs/adv_program", exist_ok=True)
            torch.save(self.model.program.state_dict(), f"logs/adv_program/_program_{task}.pt")
            
            self.eval(test_loader_CPs)
            self.after_task()
            
            self.model.program = Program().cuda()
        
    def eval(self, test_loader):
        self.model.program.eval()
        
        for saved_program in [self.programs[-1]]:
            self.model.program.load_state_dict(saved_program)
            task_correct = {}
            task_total = {}
            with torch.no_grad():
                for _, (_, x, y) in tqdm(enumerate(test_loader)):
                    x, y = x.cuda(), y.cuda()
                    y_adv = self.model(x)

                    task_indices = torch.tensor([self.class_to_task[clz.item()] for clz in y]).to(device)
                    predictions = y_adv.argmax(dim=1)

                    for task in torch.unique(task_indices):
                        mask = task_indices == task
                        correct_count = (predictions[mask] == task_indices[mask]).sum().item()
                        total_count = mask.sum().item()

                        if task.item() not in task_correct:
                            task_correct[task.item()] = 0
                            task_total[task.item()] = 0

                        task_correct[task.item()] += correct_count
                        task_total[task.item()] += total_count

            for task, correct in task_correct.items():
                total = task_total[task]
                accuracy = 100 * correct / total if total > 0 else 0
                print(f"Task {task}: Accuracy = {accuracy:.2f}%")
    
    def before_task(self, task, data_manager):
        print(f"Before task {task} =====")
        self._classes_seen_so_far = (self._known_classes + data_manager.get_task_size(task + 1))
        self._class_increments.append((self._known_classes, self._classes_seen_so_far - 1))
        print(self.model)
                
    def after_task(self):
        self._known_classes = self._classes_seen_so_far
        print(self.model)
        print("=====================================\n")
    
    def compute_loss(self, y_adv, y):
        nll_loss = F.cross_entropy(y_adv, y)
        reg_loss = config["lambda"] * torch.linalg.vector_norm(self.model.program.W.view(-1), ord=2) ** 2
        loss = nll_loss
        return {"loss": loss, "nll_loss": nll_loss, "reg_loss": reg_loss}
    
    def finetune(self, train_loader, task, epochs=20):
        params = [
            {"params": self.model.program.parameters(), "lr": 1e-2},
        ]
        optimizer = optim.Adam(params, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        
        pbar = tqdm(range(epochs))
        for epoch in pbar:
            self.model.program.train()
            total_loss, correct, total = 0, 0, 0
            total_nll_loss, total_reg_loss, total_cosine_loss = 0, 0, 0

            for _, x, y in train_loader:
                # print(self.model.program.G[0].weight.view(-1)[:5])
                x = x.to(device)
                y_adv = self.model(x)
                
                labels = torch.ones_like(y).cuda() * task
                loss = self.compute_loss(y_adv, labels)

                optimizer.zero_grad()
                loss["loss"].backward()
                optimizer.step()

                total_loss += loss["loss"].item()
                correct += (y_adv.argmax(dim=1) == labels).sum().item()
                total_nll_loss += loss["nll_loss"].item()
                total_reg_loss += loss["reg_loss"].item()
                total += y.size(0)

                info = f"Epoch {epoch+1}: Loss {total_loss/total:.4f}, Nll Loss {total_nll_loss/total:.4f}, Reg Loss {total_reg_loss/total:.4f}, Accuracy {100 * correct/total:.2f}%"
                pbar.set_description(info)
                
            scheduler.step()
        
        for _, x, y in train_loader:
            for clz in y.unique():
                self.class_to_task[clz.item()] = task

learner = Learner()
learner.learn(data_manager)