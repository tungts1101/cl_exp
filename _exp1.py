import torch
from torch import nn, optim
import torch.nn.functional as F
from utils.data_manager import DataManager
import timm
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.toolkit import accuracy


seed = 42

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        self.freeze_backbone()
        self.head = nn.Linear(self.backbone.num_features, 20, bias=False)    
    
    def forward(self, x):
        f = self.backbone(x)
        y = self.head(f)
        return y
    
    def freeze_backbone(self):
        self.backbone.requires_grad_(False)
    
    def show_num_params(self,verbose=False):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Toal number of parameters: {total_params:,}")
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total number of trainable parameters: {trainable_params:,}")

class Learner:
    def __init__(self):
        self._known_classes = 0
        self._classes_seen_so_far = 0
        self._class_increments = []
        
        self.net = Net()
        self.net = self.net.cuda()

    def learn(self, data_manager):
        num_tasks = data_manager.nb_tasks - 1

        for task in range(num_tasks):
            self.before_task(task)
            
            if task == 0:
                trainset = data_manager.get_dataset(
                    np.arange(self._known_classes, self._classes_seen_so_far),
                    source="train", mode="train")
                train_loader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
                testset = data_manager.get_dataset(
                    np.arange(self._known_classes, self._classes_seen_so_far),
                    source="test", mode="test")
                test_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)
                self.finetune(train_loader, test_loader)
            
            trainset_CPs = data_manager.get_dataset(
                np.arange(self._known_classes, self._classes_seen_so_far),
                source="train", mode="test")
            train_loader_CPs = DataLoader(trainset_CPs, batch_size=128, shuffle=False, num_workers=4)
            testset_CPs = data_manager.get_dataset(
                np.arange(0, self._classes_seen_so_far),
                source="test", mode="test")
            test_loader_CPs = DataLoader(testset_CPs, batch_size=128, shuffle=False, num_workers=4)
            self.train(train_loader_CPs)
            self.eval(test_loader_CPs)
            
            self.after_task()
    
    def before_task(self, task):
        self._classes_seen_so_far = (self._known_classes + data_manager.get_task_size(task + 1))
        self._class_increments.append((self._known_classes, self._classes_seen_so_far - 1))
        print(f"Learn classes: {self._known_classes} - {self._classes_seen_so_far - 1}")
                
    def after_task(self):
        self._known_classes = self._classes_seen_so_far
    
    def train(self, train_loader):
        pass
    
    def eval(self, test_loader):
        self.net.eval()
        y_pred, y_true = [], []
        with torch.no_grad():
            for _, (_, x, y) in tqdm(enumerate(test_loader)):
                x = x.cuda()
                logits = self.net(x)
                predicts = torch.topk(logits, k=1, dim=1, largest=True, sorted=True)[1]
                y_pred.append(predicts.cpu().numpy())
                y_true.append(y.cpu().numpy())
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        acc_total, grouped = accuracy(y_pred.T[0], y_true, self._known_classes, self._class_increments)
        print(f"Acc total: {acc_total}, Acc grouped: {grouped}")
    
    def finetune(self, train_loader, test_loader):
        epochs = 1
        body_lr = 1e-2
        weight_decay = 5e-4
        min_lr = 0.0
        
        optimizer = optim.SGD(self.net.parameters(), lr=body_lr, momentum=0.9, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
        
        pbar = tqdm(range(epochs))
        for _, epoch in enumerate(pbar):
            self.net.train()
            losses, train_acc = 0.0, 0
            for i, (_, x, y) in enumerate(train_loader):
                x, y = x.cuda(), y.cuda()
                logits = self.net(x)
                
                loss = F.cross_entropy(F.softmax(logits, dim=1), y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses += loss.item()
                correct = (logits.argmax(dim=1) == y).sum().item()
                train_acc += correct
            scheduler.step()
            train_acc = train_acc * 100 / len(train_loader.dataset)
            
            self.net.eval()
            test_acc = 0
            with torch.no_grad():
                for i, (_, x, y) in enumerate(test_loader):
                    x, y = x.cuda(), y.cuda()
                    logits = self.net(x)
                    correct = (logits.argmax(dim=1) == y).sum().item()
                    test_acc += correct
            test_acc = test_acc * 100 / len(test_loader.dataset)
            
            info = "Loss {:.3f}, Train_acc {:.2f}, Test_acc {:.2f}".format(
                losses / len(train_loader.dataset), train_acc, test_acc)
            pbar.set_description(info)


learner = Learner()
data_manager = DataManager("imagenetr", True, seed, 0, 20, False)
learner.learn(data_manager)
