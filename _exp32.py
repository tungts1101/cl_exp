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

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model("vit_base_patch16_224_ssf", pretrained=True, num_classes=0)
        self.freeze_backbone()
        self.fc = ContinualLearnerHead(768, 20).cuda()
        self.Wrand = None
    
    def setup_RP(self):
        self.Wrand = torch.randn(768, 10000).cuda()
        self.fc = nn.Linear(10000, 20).cuda()
        nn.init.trunc_normal_(self.fc.weight, std=0.02)
    
    def freeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            if "head" not in name and "ssf_" not in name:
                param.requires_grad_(False)
    
    def forward(self, x):
        f = self.backbone(x)
        if self.Wrand is None:
            y = self.fc(f)['logits']
        else:
            f = F.relu(f @ self.Wrand)
            y = self.fc(f)
        return y
    
    def __repr__(self):
        trainable_params = count_parameters(self, trainable=True)
        total_params = count_parameters(self)
        return f"Model(trainable_params={trainable_params}, total_params={total_params}, percentage={trainable_params * 100 / total_params:.2f})"

def energy_score(logits, temperature=1.0):
    return -temperature * torch.logsumexp(logits / temperature, dim=0)

class Learner:
    def __init__(self):
        self._known_classes = 0
        self._classes_seen_so_far = 0
        self._class_increments = []
        self.model = Model().to(device)
        self.accuracy_matrix = []
        self.cur_task = -1
        
        self.G = torch.zeros(10000, 10000)
        self.Ws = []
        self.Qs = []

    def learn(self, data_manager):
        self.data_manager = data_manager
        num_tasks = data_manager.nb_tasks

        for task in range(num_tasks):
            self.before_task(task, data_manager)
            
            if task == 0:
                trainset = data_manager.get_dataset(
                    np.arange(self._known_classes, self._classes_seen_so_far), source="train", mode="train")
                train_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
                self.finetune(train_loader)
                self.model.setup_RP()
                self.model.backbone.requires_grad_(False)
            
            self.Ws.append(torch.zeros(10000, 20))
                            
            trainset_CPs = data_manager.get_dataset(
                np.arange(self._known_classes, self._classes_seen_so_far), source="train", mode="test")
            train_loader_CPs = DataLoader(trainset_CPs, batch_size=64, shuffle=True, num_workers=4)
            testset_CPs = self.data_manager.get_dataset(
                np.arange(0, self._classes_seen_so_far), source="test", mode="test")
            test_loader_CPs = DataLoader(testset_CPs, batch_size=128, shuffle=True, num_workers=4)
            self.fit(train_loader_CPs, test_loader_CPs)
            
            self.after_task()

    def before_task(self, task, data_manager):
        print(self.model)
        self._classes_seen_so_far = self._known_classes + data_manager.get_task_size(task)
        self._class_increments.append((self._known_classes, self._classes_seen_so_far - 1))
        self.cur_task = task

    def after_task(self):
        self._known_classes = self._classes_seen_so_far

    def fit(self, train_loader, test_loader):
        self.model.eval()
        FS = []
        YS = []
        with torch.no_grad():
            for _, (_, a, x, y) in tqdm(enumerate(train_loader)):
                x, y = x.cuda(), y.cuda()
                f = self.model.backbone(x)
                FS.append(f.cpu())
                YS.append((y // 20).cpu())
        
        FS = torch.cat(FS, dim=0)
        YS = torch.cat(YS, dim=0)
        YS = F.one_hot(YS, 20).float()
        FS = F.relu(FS @ self.model.Wrand.cpu())
        
        self.G = self.G + FS.T @ FS
        Q = FS.T @ YS
        self.Qs.append(Q)
        
        for i, Q in enumerate(self.Qs):
            ridge = self.optimise_ridge_parameter(FS, YS)
            Wo = torch.linalg.solve(self.G + ridge * torch.eye(self.G.size(dim=0)), Q).T
            self.Ws[i] = copy.deepcopy(Wo)
        
        task_true = []
        sum_ene_pred = []
        min_ene_pred = []
        dif_ene_pred = []
        
        with torch.no_grad():
            for _, (_, a, x, y) in tqdm(enumerate(test_loader)):
                a, x, y = a.cuda(), x.cuda(), y.cuda()
                
                for ia, ix, iy in zip(a, x, y):                    
                    sum_energies, min_energies = [], []
                    dif_energies = []
                    rep_energies = []
                    
                    for i, W in enumerate(self.Ws):
                        self.model.fc.weight.data = W.cuda()
                        
                        logits = self.model(ix.unsqueeze(0)).squeeze(0)
                        # print(f"Logits shape: {logits.shape}")
                        
                        energy_loss = energy_score(logits, temperature=10)
                        sum_energies.append(energy_loss.sum().item())
                    
                    # print(f"True label: {iy}")
                    # print(f"Sum: {sum_energies}, Min: {min_energies}, Dif: {dif_energies}")
                    task_idx = (iy // 20).item()
                    sum_value, sum_index = torch.min(torch.tensor(sum_energies), 0)
                    min_value, min_index = torch.min(torch.tensor(min_energies), 0)
                    dif_value, dif_index = torch.max(torch.tensor(dif_energies), 0)
                    
                    # print(f"Task idx: {task_idx} | {sum_value}, {sum_index} | {min_value}, {min_index} | {dif_value}, {dif_index}")
                    task_true.append(task_idx)
                    sum_ene_pred.append(sum_index)
                    min_ene_pred.append(min_index)
                    dif_ene_pred.append(dif_index)
            
        task_true = np.array(task_true)
        sum_ene_pred = np.array(sum_ene_pred)
        min_ene_pred = np.array(min_ene_pred)
        dif_ene_pred = np.array(dif_ene_pred)
        
        acc_sum_ene = (task_true == sum_ene_pred).mean()
        acc_min_ene = (task_true == min_ene_pred).mean()
        acc_dif_ene = (task_true == dif_ene_pred).mean()
        
        print(f"Sum Energy Acc: {acc_sum_ene:.2f}, Min Energy Acc: {acc_min_ene:.2f}, Dif Energy Acc: {acc_dif_ene:.2f}")

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
        print("Optimal lambda: " + str(ridge))
        return ridge

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

                info = f"Task {self.cur_task}, Epoch {epoch}, %: {total * 100 / len(train_loader.dataset):.2f}, Loss: {total_loss / total:.4f}, , Acc: {correct * 100 / total:.2f}"
                pbar.set_description(info)

            scheduler.step()
            
        logger.info(f"Train accuracy: {correct * 100 / total:.2f}")

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