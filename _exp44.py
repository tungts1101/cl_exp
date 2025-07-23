import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import random
import timm
from tqdm import tqdm
import os
import logging
from datetime import datetime
from peft import LoraConfig, get_peft_model
import copy
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
from petl import vision_transformer_ssf
from petl import vision_transformer_adapter
from petl.vpt import build_promptmodel
from easydict import EasyDict
from _exp import ContinualLearnerHead
from util import compute_metrics, accuracy, set_random
import gc
import time
import math
from merging.task_vectors import TaskVector, merge_max_abs
from merging.ties import merge_methods, state_dict_to_vector, vector_to_state_dict
from inc_net import CosineLinear

timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

def setup_logger(log_file=f'logs/{timestamp}_exp43.log'):
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

class DyT(nn.Module):
    def __init__(self, C, alpha):
        super().__init__()
        self.a = nn.Parameter(torch.ones(1) * alpha)
        self.g = nn.Parameter(torch.ones(C))
        self.b = nn.Parameter(torch.zeros(C))
        
        nn.init.normal_(self.g, mean=1, std=0.02)
        nn.init.normal_(self.b, std=0.02)
    
    def forward(self, x):
        x = torch.tanh(self.a * x)
        return self.g * x + self.b
    
class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        
        self.backbone = timm.create_model(config["model_backbone"], pretrained=True, num_classes=0)
        self.backbone.requires_grad_(False)
        self.norm = DyT(self.backbone.num_features, 1.0)
        self.head = None
        self.fc = None
    
    @property
    def feature_dim(self):
        return 768
    
    def update_head(self, num_classes):
        if self.head == None:
            self.head = ContinualLearnerHead(
                self.backbone.num_features, 
                num_classes)
        else:
            self.head.update(num_classes)
        self.head.cuda()
    
    def update_fc(self, nb_classes):
        fc = CosineLinear(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc
    
    def get_norm_params(self):
        return {"a": self.norm.a.cpu().item(), "g": self.norm.g.mean().cpu().item(), "b": self.norm.b.mean().cpu().item()}
    
    def get_features(self, x):
        f = self.backbone(x)
        f = self.norm(f)
        return f
    
    def train_forward(self, x):
        f = self.get_features(x)
        y = self.head(f)
        return y
    
    def forward(self, x):
        f = self.get_features(x)
        y = self.fc(f)
        return y
    
    def __repr__(self):
        trainable_params = count_parameters(self, trainable=True)
        total_params = count_parameters(self)
        return f"Model(trainable_params={trainable_params}, total_params={total_params}, percentage={trainable_params * 100 / total_params:.2f})"
    
os.makedirs("/media/ellen/HardDisk/cl/logs/checkpoints", exist_ok=True) 

def backbone_base():
    return "/media/ellen/HardDisk/cl/logs/checkpoints/imagenetr_backbone_base.pt"
   
def backbone_checkpoint(task):
    return f"/media/ellen/HardDisk/cl/logs/checkpoints/imagenetr_backbone_{task}.pt"

class Learner:
    def __init__(self, config):
        self._config = config
        self._known_classes = 0
        self._total_classes = 0
        self._class_increments = []
        self._cur_task = -1
        self.accuracy_matrix = []
        
        self.model = Model(config).cuda()
        self.model.eval()
        
        self._faa, self._ffm = 0, 0
    
    def learn(self, data_manager):
        self.data_manager = data_manager
        num_tasks = data_manager.nb_tasks
        self.total_classnum = data_manager.get_total_classnum()

        for task in range(num_tasks):
            self.before_task(task, data_manager)
            self.train()
            self.eval()
            self.after_task()
    
    def before_task(self, task, data_manager):
        self._total_classes = self._known_classes + data_manager.get_task_size(task)
        self._class_increments.append((self._known_classes, self._total_classes - 1))
        self._cur_task = task

    def after_task(self):
        self._known_classes = self._total_classes
    
    def eval(self):
        print(f"Model use RP: {self.model.fc.use_RP}")
        
        test_set = self.data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4)
        
        self.model.eval()
        
        y_true, y_pred = [], []
        with torch.no_grad():
            for _, (_, _, x, y) in tqdm(enumerate(test_loader)):
                x, y = x.cuda(), y.cuda()
                
                logits = self.model(x)['logits']
                predicts = logits.argmax(dim=1)
                
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
        
        self._faa, self._ffm = faa, ffm
    
    def train(self):
        trainset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="train")
        train_loader = DataLoader(trainset, batch_size=self._config["train_batch_size"], shuffle=True, num_workers=4)
        
        task_model = Model(self._config).cuda()
        task_model.update_head(self._total_classes - self._known_classes)
        task_model.norm.g.data.copy_(self.model.norm.g.data)
        task_model.norm.b.data.copy_(self.model.norm.b.data)
                
        epochs = self._config["train_epochs"]
        optimizer = optim.SGD(task_model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
        
        pbar = tqdm(range(epochs))
        for _, epoch in enumerate(pbar):
            task_model.train()
            total_loss, total, total_acc = 0, 0, 0

            for _, (_, _, x, y) in enumerate(train_loader):
                x, y = x.cuda(), y.cuda()
                y = torch.where(y - self._known_classes >= 0, y - self._known_classes, -100)
                
                logits = task_model.train_forward(x)['logits']
                loss = F.cross_entropy(logits, y)
                                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total += len(y)
                total_acc += (logits.argmax(dim=1) == y).sum().item()
                
                info = f"Task {self._cur_task + 1}, Epoch {epoch + 1}, Loss: {total_loss / total:.4f}, Acc: {total_acc * 100 / total:.2f}"
                pbar.set_description(info)
            scheduler.step()
                
        self.model.norm.a.data.copy_((self.model.norm.a.data * self._cur_task + task_model.norm.a.data) / (self._cur_task + 1))
        # self.model.norm.g.data.copy_(torch.max(self.model.norm.g.data, task_model.norm.g.data))
        # self.model.norm.b.data.copy_(torch.max(self.model.norm.b.data, task_model.norm.b.data))
        self.model.norm.g.data.copy_(task_model.norm.g.data)
        self.model.norm.b.data.copy_(task_model.norm.b.data)
        print(self.model.get_norm_params())
        
        # trainset = data_manager.get_dataset(np.arange(0, self._total_classes), source="train", mode="train")
        # train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
        
        trainset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="test")
        train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
        
        del self.model.fc
        self.model.fc = None
        self.model.update_fc(self._total_classes)
        self.model.fc.use_RP = True
        if self._cur_task == 0:
            M = 10000
            self.model.fc.W_rand = torch.randn(self.model.feature_dim, M).cuda()
            self.W_rand = copy.deepcopy(self.model.fc.W_rand)
            self.Q = torch.zeros(M, self.total_classnum)
            self.G = torch.zeros(M, M)
        else:
            self.model.fc.W_rand = self.W_rand
        
        self.model.eval()
        fs = []
        ys = []
        with torch.no_grad():
            for _, (_, _, x, y) in tqdm(enumerate(train_loader)):
                x, y = x.cuda(), y.cuda()
                embedding = self.model.get_features(x)
                fs.append(embedding.cpu())
                ys.append(y.cpu())
                
        fs = torch.cat(fs, dim=0)
        ys = torch.cat(ys, dim=0)
        # for class_index in np.unique(trainset.labels):
        #     data_index = (ys == class_index).nonzero().squeeze(-1)
        #     class_prototype = fs[data_index].mean(0)
        #     self.model.fc.weight.data[class_index] = class_prototype
        
        Y = F.one_hot(ys, self.total_classnum).float()
        hs = F.relu(fs @ self.model.fc.W_rand.cpu())
        self.Q += hs.T @ Y
        self.G += hs.T @ hs
        ridge = self.optimise_ridge_parameter(hs, Y)
        Wo = torch.linalg.solve(self.G + ridge * torch.eye(self.G.size(dim=0)), self.Q).T
        self.model.fc.weight.data = Wo[0 : self.model.fc.weight.shape[0], :].cuda()
            
        del task_model
        gc.collect()
        torch.cuda.empty_cache()

    def optimise_ridge_parameter(self, Features, Y):
        ridges = 10.0 ** np.arange(-8, 9)
        num_val_samples = int(Features.shape[0] * 0.8)
        losses = []
        Q_val = Features[0:num_val_samples, :].T @ Y[0:num_val_samples, :]
        G_val = Features[0:num_val_samples, :].T @ Features[0:num_val_samples, :]
        for ridge in ridges:
            Wo = torch.linalg.solve(
                G_val + ridge * torch.eye(G_val.size(dim=0)), Q_val
            ).T  # better nmerical stability than .inv
            Y_train_pred = Features[num_val_samples::, :] @ Wo.T
            losses.append(F.mse_loss(Y_train_pred, Y[num_val_samples::, :]))
        ridge = ridges[np.argmin(np.array(losses))]
        logger.info("Optimal lambda: " + str(ridge))
        return ridge
    
def set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

data_table = {
    "cifar224": (10, 10),
    "cars": (16, 20),
    "vtab": (10, 10),
    "omnibenchmark": (30, 30),
    "cub": (20, 20),
    "imagenetr": (20, 20),
    "imageneta": (20, 20)
}

for dataset_name in ["imagenetr"]:
    start_time = time.time()
    faa, ffm = [], []
    for seed in [1993]:
        set_random(1)
        
        config = {
            "seed": seed,
        }
        
        dataset_init_cls = data_table[dataset_name][0]
        dataset_increment = data_table[dataset_name][1]
        dataset_config = {
            "dataset_name": dataset_name,
            "dataset_init_cls": dataset_init_cls,
            "dataset_increment": dataset_increment
        }
        config.update(dataset_config)
        
        data_manager = DataManager(config["dataset_name"], True, config["seed"], config["dataset_init_cls"], config["dataset_increment"], False)
        
        model_config = {
            "model_backbone": "vit_base_patch16_224",
        }
        config.update(model_config)
        
        train_config = {
            "train_epochs": 5,
            "train_batch_size": 48,
        }
        
        config.update(train_config)
        
        for item in config.items():
            logger.info(f"{item[0]}: {item[1]}")
        
        learner = Learner(config)
        learner.learn(data_manager)
    
    logger.info(f"End experiment in {time.time() - start_time}\n")