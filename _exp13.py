import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm
from timm.models.layers.weight_init import trunc_normal_
from petl import vision_transformer_ssf # register vision_transformer_ssf
import numpy as np
import random
from tqdm import tqdm
import gc
import operator
from utils.data_manager import DataManager
from utils.toolkit import accuracy
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


class Neck(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(out_features))
        self.shift = nn.Parameter(torch.zeros(out_features))
        
        trunc_normal_(self.scale, mean=1., std=0.02)
        trunc_normal_(self.shift, std=0.02)
    
    def forward(self, x):
        return x * self.scale + self.shift
        

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.backbone = timm.create_model("vit_base_patch16_224_ssf", pretrained=True, num_classes=0)
        self.freeze_backbone()
        self.neck = Neck(self.backbone.num_features, self.backbone.num_features)
        # self.neck = nn.Identity()
    
    def freeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            if "head." not in name and "ssf_" not in name: 
                param.requires_grad = False
        
        for name, param in self.backbone.named_parameters():
            if "ssf_scale" in name:
                trunc_normal_(param, mean=1., std=0.02)
            if "ssf_shift" in name:
                trunc_normal_(param, std=0.02)
    
    def configure_finetune(self, num_classes):
        # self.head = nn.Linear(self.backbone.num_features, num_classes, bias=False)
        # trunc_normal_(self.head.weight, std=0.02)
        self.head = ContinualLearnerHead(self.backbone.num_features, num_classes)
    
    def configure(self):
        self.Win = torch.randn(self.backbone.num_features, 10000).cuda()
        self.Wout = torch.zeros(10000, 200).cuda()
        
    def forward(self, x):
        f = self.backbone(x)
        f = self.neck(f)
        h = F.relu(f @ self.Win)
        y = h @ self.Wout
        return y
    
    def tune_forward(self, x):
        f = self.backbone(x)
        f = self.neck(f)
        y = self.head(f)['logits']
        return y
    
class Learner:
    def __init__(self):
        self.model = Model()
        self.model = self.model.cuda()
        self.model.configure()
        
        self._known_classes = 0
        self._classes_seen_so_far = 0
        self.class_increments = []
        self.backbone_params = {}
        
        self.C = torch.zeros(10000, 200)
        self.Q = torch.zeros(10000, 10000)
        self.I = torch.eye(10000)
    
    def train(self, datamanager):
        num_task = datamanager.nb_tasks - 1
        
        for task in range(num_task):
            self._classes_seen_so_far = self._known_classes + datamanager.get_task_size(task+1)
            self.class_increments.append((self._known_classes, self._classes_seen_so_far - 1))
            
            print(f"Learn classes: {self._known_classes} - {self._classes_seen_so_far - 1}")
            print(f"Neck: {self.model.neck.scale.data[:5]} | {self.model.neck.shift.data[:5]}")
            
            # if task == 0:
            trainset = datamanager.get_dataset(
                np.arange(self._known_classes, self._classes_seen_so_far),
                source="train", mode="train")
            trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
            testset = datamanager.get_dataset(
                np.arange(self._known_classes, self._classes_seen_so_far),
                source="test", mode="test")
            testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)
            self.finetune(trainloader, testloader, task, 20)
                
            trainset_CPs = datamanager.get_dataset(
                np.arange(self._known_classes, self._classes_seen_so_far),
                source="train", mode="test")
            trainloader_CPs = DataLoader(trainset_CPs, batch_size=128, shuffle=False, num_workers=4)
            testset_CPs = datamanager.get_dataset(
                np.arange(0, self._classes_seen_so_far),
                source="test", mode="test")
            testloader_CPs = DataLoader(testset_CPs, batch_size=128, shuffle=False, num_workers=4)
            
            self.fit(trainloader_CPs, testloader_CPs)
            self._known_classes = self._classes_seen_so_far
    
    def fit(self, trainloader, testloader):
        fs = []
        ys = []
        for i, (_, x, y) in tqdm(enumerate(trainloader)):
            x, y = x.cuda(), y.cuda()
            with torch.no_grad():
                fx = self.model.backbone(x)
                fx = self.model.neck(fx)
                
            fs.append(fx.cpu())
            ys.append(y.cpu())
        
        ys = torch.cat(ys, dim=0)
        fs = torch.cat(fs, dim=0)
        
        H = F.relu(fs @ self.model.Win.cpu())
        Y = F.one_hot(ys, num_classes=200).float()
        
        self.Q = self.Q + H.T @ H
        self.C = self.C + H.T @ Y
        ridge = self.optimise_ridge_parameter(H, Y)
        Wout = torch.linalg.solve(self.Q + ridge * self.I, self.C)
        self.model.Wout = Wout.cuda()
        
        self.model.eval()
        y_pred, y_true = [], []
        for _, (_, x, y) in tqdm(enumerate(testloader)):
            x = x.cuda()
            with torch.no_grad():
                outputs = self.model(x)
            predicts = torch.topk(outputs, k=1, dim=1, largest=True, sorted=True)[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(y.cpu().numpy())
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        acc_total, grouped = accuracy(y_pred.T[0], y_true, self._known_classes, self.class_increments)
        print(f"Acc total: {acc_total}, Acc grouped: {grouped}")
    
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
        print(f"Optimal ridge: {ridge}")
        return ridge
    
    def finetune(self, train_loader, test_loader, task, epochs=20):     
        task_model = Model()
        task_model.neck.load_state_dict(self.model.neck.state_dict())
        if task != 0:
            task_model.backbone.load_state_dict(self.model.backbone.state_dict())
            task_model.backbone.requires_grad_(False)
            
        task_model.configure_finetune(20)
        task_model = task_model.cuda()
        
        base_lr = 1e-2
        weight_decay = 5e-4
        min_lr = 0.0
        
        # adapter_scale_lr = 1e-2
        # head_scale_lr = 1
        # params = [
        #     {"params": task_model.adapter.parameters(), "lr": adapter_scale_lr * base_lr},
        #     {"params": task_model.head.heads[-1][-1].parameters(), "lr": head_scale_lr * base_lr}
        # ]
        optimizer = optim.SGD(task_model.parameters(), lr=base_lr, momentum=0.9, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
        
        pbar = tqdm(range(epochs))
        for _, epoch in enumerate(pbar):
            task_model.train()
            losses, train_acc = 0.0, 0
            for i, (_, x, y) in enumerate(train_loader):
                x, y = x.cuda(), y.cuda()
                y = torch.where(y - self._known_classes >= 0, y - self._known_classes, -100)
                logits = task_model.tune_forward(x)
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
                    y = torch.where(y - self._known_classes >= 0, y - self._known_classes, -100)
                    logits = task_model.tune_forward(x)
                    correct = (logits.argmax(dim=1) == y).sum().item()
                    test_acc += correct
            test_acc = test_acc * 100 / len(test_loader.dataset)
            
            info = "Loss {:.3f}, Train_acc {:.2f}, Test_acc {:.2f}".format(
                losses / len(train_loader.dataset), train_acc, test_acc)
            pbar.set_description(info)
        
        self.model.neck.load_state_dict(task_model.neck.state_dict())
        if task == 0:
            self.model.backbone.load_state_dict(task_model.backbone.state_dict())
            self.model.backbone.requires_grad_(False)
        
        # for name, param in task_model.backbone.named_parameters():
        #     if "ssf_scale_" in name:
        #         tv = operator.attrgetter(name)(task_model.backbone).detach().clone()
        #         if name not in self.backbone_params:
        #             self.backbone_params[name] = (1.0, [])
        #         self.backbone_params[name][1].append(tv)
        #     elif "ssf_shift_" in name:
        #         tv = operator.attrgetter(name)(task_model.backbone).detach().clone()
        #         if name not in self.backbone_params:
        #             self.backbone_params[name] = (0.0, [])
        #         self.backbone_params[name][1].append(tv)
        # ties_merge(self.model.backbone, self.backbone_params, 1.0, 20)
        
        del task_model
        torch.cuda.empty_cache()
        gc.collect()

learner = Learner()
learner.train(data_manager)