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
        
        self.network = timm.create_model(
            "vit_base_patch16_224_ssf", pretrained=True, num_classes=0)
        self.fc = nn.Linear(768, 20, bias=False).cuda()
        
        embed_dim = self.network.embed_dim
        M = 10000
        num_classes = 200
        
        self.Win = torch.normal(0, 1, (embed_dim, M)).cuda()
        self.Wout = torch.zeros(M, num_classes).cuda()
    
    def update_tune_fc(self):
        self.fc1 = nn.Linear(768, 20, bias=False).cuda()
        self.fc2 = nn.Linear(768, 20, bias=False).cuda()
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
    
    def get_features(self, x):
        return self.network(x)

    def forward(self, x):
        f = self.network(x)
        h = F.relu(f @ self.Win)
        y = h @ self.Wout
        return y
    
    def freeze_backbone(self):
        for name, param in self.network.named_parameters():
            if "head." not in name and "ssf_scale" not in name and "ssf_shift_" not in name: 
                param.requires_grad = False
    
    def show_num_params(self,verbose=False):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Toal number of parameters: {total_params:,}")
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total number of trainable parameters: {trainable_params:,}")
    
class Learner:
    def __init__(self):
        self.model = Net()
        self.model = self.model.cuda()
        self.model.freeze_backbone()
        
        self._known_classes = 0
        self._classes_seen_so_far = 0
        self.class_increments = []
        
        self.C = torch.zeros(10000, 200)
        self.Q = torch.zeros(10000, 10000)
        self.I = torch.eye(10000)
    
    def train(self, datamanager):
        num_task = datamanager.nb_tasks - 1
        
        for task in range(num_task):
            # self.model.show_num_params()
            
            self._classes_seen_so_far = self._known_classes + datamanager.get_task_size(task+1)
            self.class_increments.append((self._known_classes, self._classes_seen_so_far - 1))
            
            print(f"Learn classes: {self._known_classes} - {self._classes_seen_so_far - 1}")
            
            # if task == 0:
            # trainset = datamanager.get_dataset(
            #     np.arange(self._known_classes, self._classes_seen_so_far),
            #     source="train", mode="train")
            # trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
            # testset = datamanager.get_dataset(
            #     np.arange(self._known_classes, self._classes_seen_so_far),
            #     source="test", mode="test")
            # testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)
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
            
            # after task
            self._known_classes = self._classes_seen_so_far
    
    def fit(self, trainloader, testloader):
        fs = []
        ys = []
        for i, (_, x, y, a) in tqdm(enumerate(trainloader)):
            x, y, a = x.cuda(), y.cuda(), a.cuda()
            with torch.no_grad():
                fx = self.model.network(x)
                fa = self.model.network(a)
            fs.append(fx.cpu())
            fs.append(fa.cpu())
            ys.append(y.cpu())
            ys.append(y.cpu())
        
        ys = torch.cat(ys, dim=0)
        fs = torch.cat(fs, dim=0)
        
        H = F.relu(fs @ self.model.Win.cpu())
        Y = F.one_hot(ys, num_classes=200).float()
        
        # fs = torch.cat(fs, dim=0)
        # ys = torch.cat(ys, dim=0)
        # H = F.relu(fs @ self.model.Win.cpu())
        # Y = F.one_hot(ys, num_classes=200).float()
        
        self.Q = self.Q + H.T @ H
        self.C = self.C + H.T @ Y
        ridge = 1e6
        Wout = torch.linalg.solve(self.Q + ridge * self.I, self.C)
        self.model.Wout = Wout.cuda()

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
    
    def tune(self, trainloader, testloader, starting_label):
        self.model.update_tune_fc()
        self.model.show_num_params()
        self.model.train()
        
        tune_epochs = 20
        
        body_lr = 0.01
        head_lr = 0.01
        weight_decay = 5e-4
        min_lr = 0.0
        
        optimizer = optim.SGD(
            self.model.parameters(), lr=body_lr, 
            momentum=0.9, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tune_epochs, eta_min=min_lr)
        
        bn = nn.BatchNorm1d(20, affine=False).cuda()
        
        pbar = tqdm(range(tune_epochs))
        for _, epoch in enumerate(pbar):
            self.model.train()
            losses = 0.0
            train_acc = 0
            for i, (_, x, y, a) in enumerate(trainloader):
                x, y, a = x.cuda(), y.cuda(), a.cuda()
                y -= starting_label
                
                z1 = self.model.fc1(self.model.network(x))
                z2 = self.model.fc2(self.model.network(a))
                
                # if starting_label == 0:
                #     loss = F.cross_entropy(F.softmax(z1, dim=1), y)
                # else:
                #     z2 = self.model.fc2(self.model.network(a))
                #     c = bn(z1).T @ bn(z2)
                #     on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                #     off_diag = off_diagonal(c).pow_(2).sum()
                #     loss = on_diag + 0.1 * off_diag
                
                # z2 = self.model.fc2(self.model.network(a))
                # c = bn(z1).T @ bn(z2)
                # on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                # off_diag = off_diagonal(c).pow_(2).sum()
                # loss = on_diag + 0.1 * off_diag
                
                # print(f"Off-diagonal: {off_diag}, On-diagonal: {on_diag}")
                # print(f"Loss: {loss}, Supervised loss: {supervised_loss}, Self-supervised loss: {self_supervised_loss}")
                
                # loss = F.cross_entropy(F.softmax(z1, dim=1), y) + 0.1 * F.cross_entropy(F.softmax(z2, dim=1), y)
                loss = F.cross_entropy(F.softmax(z1, dim=1), y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses += loss.item()
                correct = (z1.argmax(dim=1) == y).sum().item()
                train_acc += correct
            scheduler.step()
            train_acc = np.around(train_acc * 100 / len(trainloader.dataset), decimals=2)
            
            self.model.eval()
            test_acc = 0
            for i, (_, x, y, _) in enumerate(testloader):
                x, y = x.cuda(), y.cuda()
                y -= starting_label
                
                with torch.no_grad():
                    logits = self.model.fc1(self.model.network(x))
                correct = (logits.argmax(dim=1) == y).sum().item()
                test_acc += correct
                
            test_acc = np.around(test_acc * 100 / len(testloader.dataset), decimals=2)
            
            info = "Loss {:.3f}, Train_acc {:.2f}, Test_acc {:.2f}".format(
                losses / len(trainloader.dataset),
                train_acc,
                test_acc,
            )
            pbar.set_description(info)


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

learner = Learner()
learner.train(data_manager)