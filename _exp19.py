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
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.freeze_backbone()
    
    def freeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            if ("ssf_scale_" not in name) and ("ssf_shift_" not in name):
                param.requires_grad = False
    
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
        y_adv = self.backbone(x_adv)
        y_adv = self.label_mapping(y_adv)
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
        self.class_to_task = np.load("logs/adv_program/class_to_task.npy", allow_pickle=True).item()

    def learn(self, data_manager):
        testset_CPs = data_manager.get_dataset(np.arange(0, 40), source="test", mode="test")
        test_loader_CPs = DataLoader(testset_CPs, batch_size=4, shuffle=True, num_workers=4)
        self.eval(test_loader_CPs)
        
    def eval(self, test_loader):
        self.model.program.eval()
        with torch.no_grad():
            task_correct = {}
            task_total = {}
            for _, (_, x, y) in tqdm(enumerate(test_loader)):
                x, y = x.cuda(), y.cuda()
                
                task_indices = torch.tensor([self.class_to_task[clz.item()] for clz in y]).to(device)
                candidates = []
                for i_program in range(0, 2):
                    saved_program_path = f"logs/adv_program/_program_{i_program}.pt"
                    self.model.program.load_state_dict(torch.load(saved_program_path))
                    y_adv = self.model(x)
                    candidates.append(y_adv)
                
                candidates = torch.stack(candidates).to(device)
                print(y)
                print(task_indices)
                print(candidates)
                values, predictions = torch.topk(candidates, k=1, dim=2, largest=True, sorted=True)
                print(values)
                print(predictions)
                break
                
                # # List of checkpoint paths
                # checkpoint_paths = ["logs/adv_program/_program_0.pt", "logs/adv_program/_program_1.pt"]

                # # Load the first state_dict to initialize the averaging
                # avg_state_dict = torch.load(checkpoint_paths[0])

                # # Set all values to zero to prepare for averaging
                # for key in avg_state_dict.keys():
                #     avg_state_dict[key] = torch.zeros_like(avg_state_dict[key])

                # # Accumulate parameters from each checkpoint
                # num_checkpoints = len(checkpoint_paths)

                # for path in checkpoint_paths:
                #     state_dict = torch.load(path)
                #     for key in avg_state_dict.keys():
                #         avg_state_dict[key] += state_dict[key] / num_checkpoints  # Compute the mean

                # # Load the averaged state_dict into the model
                # self.model.program.load_state_dict(avg_state_dict)
                # y_adv = self.model(x)
                # values, predictions = torch.topk(y_adv, k=1, dim=1, largest=True, sorted=True)
                # print(values)
                # print(predictions)
                # print(task_indices)
                # break
                
                
            
            # for task, correct in task_correct.items():
            #     total = task_total[task]
            #     accuracy = 100 * correct / total if total > 0 else 0
            #     print(f"Task {task}: Accuracy = {accuracy:.2f}%")
        
learner = Learner()
learner.learn(data_manager)