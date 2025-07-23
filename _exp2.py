import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import timm
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import operator
import gc
import random
from petl import vision_transformer_ssf
from experiments.merge_utils import ties_merge, init_weights
from utils.data_manager import DataManager
from utils.toolkit import accuracy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

num_classes = 200
num_init_cls = 20
data_manager = DataManager("imagenetr", True, seed, 0, num_init_cls, False)

config = {
    "fine_tune_train_batch_size": 64,
    "fine_tune_train_epochs": 1,
    "merge_trim_top_k": 20,
    "merge_lambda": 50.0
}

with open("result.txt", "a") as f:
    f.write(f"{' | '.join('%s: %s' % item for item in config.items())}\n")

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.backbone = timm.create_model(
            "vit_base_patch16_224_ssf", pretrained=True, num_classes=0
        )
        self.freeze_backbone()
        self.head = nn.Linear(self.backbone.num_features, num_classes, bias=False)
        
        self.init_weights()
    
    def init_weights(self):
        for name, param in self.backbone.named_parameters():
            if "ssf_scale_" in name:
                nn.init.ones_(param)
            elif "ssf_shift_" in name:
                nn.init.zeros_(param)
        self.head.apply(init_weights)
    
    def forward(self, x):
        f = self.backbone(x)
        y = self.head(f)
        return y
    
    def freeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            if ("ssf_scale_" not in name) and ("ssf_shift_" not in name):
                param.requires_grad = False
    
    def __repr__(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"Total params: {total_params:,} | Trainable params: {trainable_params:,}"


class Learner:
    def __init__(self):
        self._known_classes = 0
        self._classes_seen_so_far = 0
        self._class_increments = []
        self.model = Model().to(device)
        self.backbone_params = {}
        self.neck_params = {}
        self.head_params = {}

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
            self.finetune(train_loader, test_loader, epochs=config["fine_tune_train_epochs"])
            
            ties_merge(self.model.backbone, self.backbone_params, lamb=config["merge_lambda"] / num_tasks, trim_top_k=config["merge_trim_top_k"])
            # ties_merge(self.model.neck, self.neck_params, lamb=config["merge_lambda"], trim_top_k=config["merge_trim_top_k"])
            # ties_merge(self.model.head, self.head_params, lamb=1.0, trim_top_k=100)
            
            testset_CPs = data_manager.get_dataset(
                np.arange(0, self._classes_seen_so_far),
                source="test", mode="test")
            test_loader_CPs = DataLoader(testset_CPs, batch_size=128, shuffle=False, num_workers=4)
            self.eval(test_loader_CPs)
            
            self.after_task()
    
    def before_task(self, task, data_manager):
        self._classes_seen_so_far = (self._known_classes + data_manager.get_task_size(task + 1))
        self._class_increments.append((self._known_classes, self._classes_seen_so_far - 1))
        print(f"Learn classes: {self._known_classes} - {self._classes_seen_so_far - 1}")
                
    def after_task(self):
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
        with open("result.txt", "a") as f:
            f.write(result + "\n")
    
    def finetune(self, train_loader, test_loader, epochs=20):
        task_model = Model().to(device)
        task_model.head = nn.Linear(task_model.backbone.num_features, num_init_cls, bias=False).to(device)
        task_model.head.apply(init_weights)
        
        body_lr = 1e-2
        weight_decay = 5e-4
        min_lr = 0.0
        
        optimizer = optim.SGD(task_model.parameters(), lr=body_lr, momentum=0.9, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
        
        pbar = tqdm(range(epochs))
        for _, epoch in enumerate(pbar):
            task_model.train()
            losses, train_acc = 0.0, 0
            for i, (_, x, y) in enumerate(train_loader):
                x, y = x.cuda(), y.cuda()
                y -= self._known_classes
                logits = task_model(x)
                loss = F.cross_entropy(F.softmax(logits, dim=1), y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses += loss.item()
                correct = (logits.argmax(dim=1) == y).sum().item()
                train_acc += correct
            scheduler.step()
            train_acc = train_acc * 100 / len(train_loader.dataset)
            
            task_model.eval()
            test_acc = 0
            with torch.no_grad():
                for i, (_, x, y) in enumerate(test_loader):
                    x, y = x.cuda(), y.cuda()
                    y -= self._known_classes
                    logits = task_model(x)
                    correct = (logits.argmax(dim=1) == y).sum().item()
                    test_acc += correct
            test_acc = test_acc * 100 / len(test_loader.dataset)
            
            info = "Loss {:.3f}, Train_acc {:.2f}, Test_acc {:.2f}".format(
                losses / len(train_loader.dataset), train_acc, test_acc)
            pbar.set_description(info)
        with open("result.txt", "a") as f:
            f.write(info + "\n")

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
        
        # for name, param in task_model.head.named_parameters():
        #     tv = operator.attrgetter(name)(task_model.head).detach().clone()
        #     if name not in self.head_params:
        #         self.head_params[name] = (0.0, [])
        #     self.head_params[name][1].append(tv)
        
        # for name, param in task_model.neck.named_parameters():
        #     tv = operator.attrgetter(name)(task_model.neck).detach().clone()
        #     if name not in self.neck_params:
        #         self.neck_params[name] = (1.0, [])
        #     self.neck_params[name][1].append(tv)
        
        self.model.head.weight.data[self._known_classes:self._classes_seen_so_far].copy_(task_model.head.weight.data)
        if self.model.head.bias and task_model.head.bias:
            self.model.head.bias.data[self._known_classes:self._classes_seen_so_far].copy_(task_model.head.bias.data)
        
        del task_model
        torch.cuda.empty_cache()
        gc.collect()

learner = Learner()
learner.learn(data_manager)