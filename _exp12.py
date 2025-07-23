import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import random
import timm
from tqdm import tqdm
import gc
import operator
import os
import logging
import gpytorch

from utils.data_manager import DataManager
from utils.toolkit import count_parameters, accuracy
from petl import vision_transformer_ssf
from experiments.merge_utils import ties_merge
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
    "fine_tune_train_batch_size": 128,
    "fine_tune_train_epochs": 50,
    "merge_trim_top_k": 20,
    "merge_lambda": 1.0
}

# with open("result.txt", "a") as f:
#     f.write(f"{' | '.join('%s: %s' % item for item in config.items())}\n")

class VariationalAdapter(nn.Module):
    def __init__(self, input_dim=768, latent_dim=200, encoder_hidden_dims=[256]):
        super().__init__()
        encoder_layers = []
        current_dim = input_dim
        for h_dim in encoder_hidden_dims:
            encoder_layers.append(nn.Linear(current_dim, h_dim))
            encoder_layers.append(nn.ReLU(inplace=True))
            encoder_layers.append(nn.LayerNorm(h_dim))
            current_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        self.fc_mu = nn.Linear(encoder_hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(encoder_hidden_dims[-1], latent_dim)
        
        decoder_layers = []
        decoder_hidden_dims = encoder_hidden_dims[::-1]
        current_dim = latent_dim
        for h_dim in decoder_hidden_dims:
            decoder_layers.append(nn.Linear(current_dim, h_dim))
            decoder_layers.append(nn.ReLU(inplace=True))
            decoder_layers.append(nn.LayerNorm(h_dim))
            current_dim = h_dim
        decoder_layers.append(nn.Linear(current_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)        
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon
    

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.backbone = timm.create_model(
            "vit_base_patch16_224", pretrained=True, num_classes=0
        )
        self.backbone.requires_grad_(False)
        # self.adapter = nn.Sequential(
        #     nn.LayerNorm(self.backbone.num_features),
        #     nn.Linear(self.backbone.num_features, self.backbone.num_features, bias=False),
        # )
        # trunc_normal_(self.adapter[-1].weight, std=0.02)
        self.adapter = VariationalAdapter(input_dim=self.backbone.num_features, latent_dim=128, encoder_hidden_dims=[256])
        self.head = ContinualLearnerHead(self.backbone.num_features, num_init_cls, with_norm=False)
    
    def forward(self, x):
        f = self.backbone(x)
        f = self.adapter(f)
        y = self.head(f)['logits']
        return y
    
    def __repr__(self):
        trainable_params = count_parameters(self.backbone, trainable=True) + count_parameters(self.head.heads, trainable=True)
        total_params = count_parameters(self.backbone) + count_parameters(self.head.heads)
        return f"Model(trainable_params={trainable_params}, total_params={total_params}, percentage={trainable_params * 100 / total_params:.2f})"
    
class Learner:
    def __init__(self):
        self._known_classes = 0
        self._classes_seen_so_far = 0
        self._class_increments = []
        self.model = Model().to(device)
        self.merged_params = {}
        
    def learn(self, data_manager):
        num_tasks = data_manager.nb_tasks - 1
        self.num_tasks = num_tasks

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
            
            # ties_merge(self.model.adapter, self.merged_params, lamb=config["merge_lambda"], trim_top_k=config["merge_trim_top_k"])
            
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
        print(f"Learn classes: {self._known_classes} - {self._classes_seen_so_far - 1}")
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
        task_model.adapter.load_state_dict(self.model.adapter.state_dict())
        
        base_lr = 1e-2
        weight_decay = 5e-4
        min_lr = 0.0
        
        adapter_scale_lr = 1e-2
        head_scale_lr = 1
        params = [
            {"params": task_model.adapter.parameters(), "lr": adapter_scale_lr * base_lr},
            {"params": task_model.head.heads[-1][-1].parameters(), "lr": head_scale_lr * base_lr}
        ]
        optimizer = optim.SGD(params, lr=base_lr, momentum=0.9, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
        
        pbar = tqdm(range(epochs))
        for _, epoch in enumerate(pbar):
            task_model.train()
            losses, train_acc = 0.0, 0
            for i, (_, x, y) in enumerate(train_loader):
                x, y = x.cuda(), y.cuda()
                y = torch.where(y - self._known_classes >= 0, y - self._known_classes, -100)
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
                    y = torch.where(y - self._known_classes >= 0, y - self._known_classes, -100)
                    logits = task_model(x)
                    correct = (logits.argmax(dim=1) == y).sum().item()
                    test_acc += correct
            test_acc = test_acc * 100 / len(test_loader.dataset)
            
            info = "Loss {:.3f}, Train_acc {:.2f}, Test_acc {:.2f}".format(
                losses / len(train_loader.dataset), train_acc, test_acc)
            pbar.set_description(info)
            
        # with open("result.txt", "a") as f:
        #     f.write(info + "\n")
        
        # mask_scale = torch.abs(task_model.adapter.scale) > torch.abs(self.model.adapter.scale)
        # self.model.adapter.scale.data.copy_(torch.where(mask_scale, task_model.adapter.scale, self.model.adapter.scale))
        # self.model.adapter.shift.data.copy_((self.model.adapter.shift * task + task_model.adapter.shift) / (task + 1))
        
        # if "scale" not in self.merged_params:
        #     self.merged_params["scale"] = (1.0, [])
        # self.merged_params["scale"][1].append(task_model.adapter.scale.detach().clone())
        # if "shift" not in self.merged_params:
        #     self.merged_params["shift"] = (0.0, [])
        # self.merged_params["shift"][1].append(task_model.adapter.shift.detach().clone())
        
        self.model.adapter.load_state_dict(task_model.adapter.state_dict())
        self.model.head.heads[-1][-1].weight.data.copy_(task_model.head.heads[-1][-1].weight) # assign head for model
        
        del task_model
        torch.cuda.empty_cache()
        gc.collect()

learner = Learner()
learner.learn(data_manager)