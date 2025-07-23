import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm
from petl import vision_transformer_ssf # register vision_transformer_ssf
import numpy as np
from tqdm import tqdm
from utils.data_manager import DataManager
from utils.toolkit import accuracy


seed = 42
data_manager = DataManager('imagenetr', True, seed, 0, 20, False)  

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.base = timm.create_model(
            "vit_base_patch16_224_ssf", pretrained=True, num_classes=0)
        
        for name, param in self.base.named_parameters():
            if "head." not in name and "ssf_scale" not in name and "ssf_shift_" not in name: 
                param.requires_grad = False
        self.tune_fc = nn.Linear(768, 20, bias=False).cuda()
    
    def configure(self):
        for param in self.base.parameters():
            param.requires_grad = False
        
        self.Win = torch.randn(768, 10000).cuda()
        self.Wout = torch.zeros(10000, 200).cuda()
        
        self.norm = nn.BatchNorm1d(10000, affine=True).cuda()
        
        # self.head[1].requires_grad_(True)
        # self.head[1].track_running_stats = False
        # self.head[1].running_mean = None
        # self.head[1].running_var = None
        # self.optimizer = optim.SGD([self.head[1].weight, self.head[1].bias], lr=1e-3)
        
        self.train()
    
    # def forward(self, x):
    #     f = self.base(x)
    #     f = f @ self.Win
    #     # f = self.norm(f)
    #     f = F.relu(f)
    #     y = f @ self.Wout
    #     return y
    
    def forward(self, x):
        f = self.base(x)
        h = F.relu(f @ self.Win)
        y = h @ self.Wout
        return y
    
    def tune_forward(self, x):
        f = self.base(x)
        y = self.tune_fc(f)
        return y
    
    def show_num_params(self,verbose=False):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Toal number of parameters: {total_params:,}")
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total number of trainable parameters: {trainable_params:,}")
    
class Learner:
    def __init__(self):
        self.model = Net()
        self.model = self.model.cuda()
        
        self._known_classes = 0
        self._classes_seen_so_far = 0
        self.class_increments = []
        
        self.C = torch.zeros(10000, 200)
        self.Q = torch.zeros(10000, 10000)
        self.I = torch.eye(10000)
    
    def train(self, datamanager):
        num_task = datamanager.nb_tasks - 1
        
        for task in range(num_task):
            self._classes_seen_so_far = self._known_classes + datamanager.get_task_size(task+1)
            self.class_increments.append((self._known_classes, self._classes_seen_so_far - 1))
            
            print(f"Learn classes: {self._known_classes} - {self._classes_seen_so_far - 1}")
            self.model.show_num_params()
            self.model.configure()
            
            if task == 0:
                trainset = datamanager.get_dataset(
                    np.arange(self._known_classes, self._classes_seen_so_far),
                    source="train", mode="train")
                trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
                testset = datamanager.get_dataset(
                    np.arange(self._known_classes, self._classes_seen_so_far),
                    source="test", mode="test")
                testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)
                
                # self.tune(trainloader, testloader, self._known_classes)
                
            trainset_CPs = datamanager.get_dataset(
                np.arange(self._known_classes, self._classes_seen_so_far),
                source="train", mode="test")
            trainloader_CPs = DataLoader(trainset_CPs, batch_size=128, shuffle=False, num_workers=4)
            testset_CPs = datamanager.get_dataset(
                np.arange(0, self._classes_seen_so_far),
                source="test", mode="test")
            testloader_CPs = DataLoader(testset_CPs, batch_size=128, shuffle=False, num_workers=4)
            
            self.fit(trainloader_CPs, testloader_CPs)
            # self.adapt(testloader_CPs)
            
            # after task
            self._known_classes = self._classes_seen_so_far
    
    def adapt(self, testloader):
        # self.model.train()
        
        # bn_layer = self.model.head[1]
        # print(f"Before adaptation: {bn_layer.running_mean[:10]}")
        # self.model.eval()
        # y_pred, y_true = [], []
        # for _, (_, x, y, _) in tqdm(enumerate(testloader)):
        #     x = x.cuda()
        #     outputs = self.model(x)
            
        #     # loss = -(F.softmax(outputs, dim=1) * F.log_softmax(outputs, dim=1)).sum(dim=1).mean()
        #     # loss.backward()
        #     # self.model.optimizer.zero_grad()
        #     # self.model.optimizer.step()
            
        #     predicts = torch.topk(outputs, k=1, dim=1, largest=True, sorted=True)[1]
        #     y_pred.append(predicts.cpu().numpy())
        #     y_true.append(y.cpu().numpy())
            
        # y_pred = np.concatenate(y_pred)
        # y_true = np.concatenate(y_true)
        # acc_total, grouped = accuracy(y_pred.T[0], y_true, self._known_classes, self.class_increments)
        # print(f"Acc total: {acc_total}, Acc grouped: {grouped}")

        # print(f"After adaptation: {bn_layer.running_mean[:10]}")
        
        self.model.eval()
        y_pred, y_true = [], []
        for _, (_, x, y, _) in tqdm(enumerate(testloader)):
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
    
    def fit(self, trainloader, testloader):
        fs = []
        ys = []
        for i, (_, x, y) in tqdm(enumerate(trainloader)):
            x, y = x.cuda(), y.cuda()
            with torch.no_grad():
                fx = self.model.base(x)
            fs.append(fx.cpu())
            ys.append(y.cpu())
        
        ys = torch.cat(ys, dim=0)
        fs = torch.cat(fs, dim=0)
        
        H = F.relu(fs @ self.model.Win.cpu())
        Y = F.one_hot(ys, num_classes=200).float()
        
        self.Q = self.Q + H.T @ H
        self.C = self.C + H.T @ Y
        ridge = 1e6
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
    
    def tune(self, trainloader, testloader, starting_label):        
        tune_epochs = 1
        body_lr = 1e-2
        weight_decay = 5e-4
        min_lr = 0.0
        
        optimizer = optim.SGD(
            self.model.parameters(), lr=body_lr, 
            momentum=0.9, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tune_epochs, eta_min=min_lr)
        
        pbar = tqdm(range(tune_epochs))
        for _, epoch in enumerate(pbar):
            self.model.train()
            losses = 0.0
            train_acc = 0
            for i, (_, x, y) in enumerate(trainloader):
                x, y, a = x.cuda(), y.cuda(), a.cuda()
                logits = self.model.tune_forward(x)
                logits_a = self.model.tune_forward(a)
                                
                loss = F.cross_entropy(F.softmax(logits, dim=1), y) + F.cross_entropy(F.softmax(logits_a, dim=1), y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses += loss.item()
                correct = (logits.argmax(dim=1) == y).sum().item()
                train_acc += correct
            scheduler.step()
            train_acc = np.around(train_acc * 100 / len(trainloader.dataset), decimals=2)
            
            self.model.eval()
            test_acc = 0
            for i, (_, x, y, _) in enumerate(testloader):
                x, y = x.cuda(), y.cuda()
                
                with torch.no_grad():
                    logits = self.model.tune_forward(x)
                correct = (logits.argmax(dim=1) == y).sum().item()
                test_acc += correct
                
            test_acc = np.around(test_acc * 100 / len(testloader.dataset), decimals=2)
            
            info = "Loss {:.3f}, Train_acc {:.2f}, Test_acc {:.2f}".format(
                losses / len(trainloader.dataset),
                train_acc,
                test_acc,
            )
            pbar.set_description(info)

learner = Learner()
learner.train(data_manager)