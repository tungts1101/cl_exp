import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
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
    "fine_tune_train_epochs": 50,
    "merge_trim_top_k": 20,
    "merge_lambda": 1.0
}

# with open("result.txt", "a") as f:
#     f.write(f"{' | '.join('%s: %s' % item for item in config.items())}\n")

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            "vit_base_patch16_224_ssf", pretrained=True, num_classes=0
        )
        self.freeze_backbone()
        self.head = ContinualLearnerHead(self.backbone.num_features, 20).to(device)
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
    
    def forward(self, x):
        f = self.backbone(x)
        y = self.head(f)['logits']
        return y
    
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
        self.backbone_params = {}

    def learn(self, data_manager):
        num_tasks = data_manager.nb_tasks - 1

        for task in range(num_tasks):
            self.before_task(task, data_manager)
            
            trainset = data_manager.get_dataset(
                np.arange(self._known_classes, self._classes_seen_so_far),
                source="train", mode="train")
            train_loader = DataLoader(trainset, batch_size=config["fine_tune_train_batch_size"], shuffle=True, num_workers=4)
            testset = data_manager.get_dataset(
                np.arange(self._known_classes, self._classes_seen_so_far),
                source="test", mode="test")
            test_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)
            self.finetune(train_loader, test_loader, task, epochs=config["fine_tune_train_epochs"])
            
            # ties_merge(self.model.backbone, self.backbone_params, config["merge_trim_top_k"], config["merge_lambda"])
            avg_merge(self.model.backbone, self.backbone_params)
            
            testset_CPs = data_manager.get_dataset(
                np.arange(0, self._classes_seen_so_far),
                source="test", mode="test")
            test_loader_CPs = DataLoader(testset_CPs, batch_size=128, shuffle=False, num_workers=4)
            
            self.eval(test_loader_CPs)
            self.after_task()
    
    def before_task(self, task, data_manager):
        print(f"Before task {task} =====")
        self._classes_seen_so_far = (self._known_classes + data_manager.get_task_size(task + 1))
        self._class_increments.append((self._known_classes, self._classes_seen_so_far - 1))
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
    
    def finetune(self, train_loader, test_loader, task, epochs=20):
        task_model = Model().to(device)
        task_model.backbone.load_state_dict(self.model.backbone.state_dict())
        
        weight_decay = 5e-4
        min_lr = 0.0
        
        base_lr = 1e-2
        backbone_lr = 1e-2
        head_lr = 1.
        params = [
            {"params": task_model.backbone.parameters(), "lr": base_lr * backbone_lr, "weight_decay": weight_decay },
            {"params": task_model.head.parameters(), "lr": base_lr * head_lr, "weight_decay": weight_decay }
        ]
        optimizer = optim.Adam(params, lr=base_lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
        
        pbar = tqdm(range(epochs))
        
        train_size = int(0.8 * len(train_loader.dataset))
        val_size = len(train_loader.dataset) - train_size
        train_dataset, val_dataset = random_split(train_loader.dataset, [train_size, val_size])

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_loader.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=train_loader.batch_size, shuffle=False)

        val_acc, val_samples = 0, 0
        for _, epoch in enumerate(pbar):
            task_model.train()
            losses, train_acc, num_samples = 0, 0, 0
            
            for i, (_, x, y) in enumerate(train_loader):
                x, y = x.cuda(), y.cuda()
                y = torch.where(y - self._known_classes >= 0, y - self._known_classes, -100)
                logits = task_model(x)

                loss = F.cross_entropy(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()
                correct = (logits.argmax(dim=1) == y).sum().item()
                train_acc += correct
                num_samples += len(y)

                info = f"Loss {losses / num_samples:.3f}, Train_acc {train_acc * 100 / num_samples:.2f}, Val_acc {val_acc * 100 / max(val_samples, 1):.2f}"
                pbar.set_description(info)

            scheduler.step()
            
            task_model.eval()
            val_acc, val_samples = 0, 0
            
            with torch.no_grad():
                for _, x, y in val_loader:
                    x, y = x.cuda(), y.cuda()
                    y = torch.where(y - self._known_classes >= 0, y - self._known_classes, -100)
                    logits = task_model(x)
                    
                    correct = (logits.argmax(dim=1) == y).sum().item()
                    val_acc += correct
                    val_samples += len(y)

                    pbar.set_description(f"Loss {losses / num_samples:.3f}, Train_acc {train_acc * 100 / num_samples:.2f}, Val_acc {val_acc * 100 / val_samples:.2f}")

            if val_acc * 100 / val_samples >= 90:
                break
            
        self.model.backbone.load_state_dict(task_model.backbone.state_dict())
        self.model.head.heads[-1][-1].weight.data.copy_(task_model.head.heads[-1][-1].weight)
        
        for name, param in task_model.backbone.named_parameters():
            if "ssf_scale_" in name:
                tv = operator.attrgetter(name)(task_model.backbone).detach().clone()
                if name not in self.backbone_params:
                    self.backbone_params[name] = (1.0, [])
                self.backbone_params[name][1].append(tv)
            elif "ssf_shift_" in name:
                tv = operator.attrgetter(name)(task_model.backbone).detach().clone()
                if name not in self.backbone_params:
                    self.backbone_params[name] = (0.0, [])
                self.backbone_params[name][1].append(tv)
        
        
        # clean up
        del task_model
        torch.cuda.empty_cache()
        gc.collect()

learner = Learner()
learner.learn(data_manager)