# sequential peft + adv program
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

def setup_logger(log_file=f'logs/{timestamp}_exp23.log'):
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

fine_tune_epochs = 0
config = {
    "pretrained_backbone": "vit_base_patch16_224",
    "dataset_name": "cub",
    "num_init_class": 20,
    
    "peft": "ssf",
    
    "r": 16,
    "lora_alpha": 16,
    
    "proj_dim": 10000,
    
    "init_matrix": "normal",
    "alpha": 1,
    
    "fine_tune_train_batch_size": 16,
    "fine_tune_train_epochs": [20],
    "finetune": "fs"
}
logger.info(f"{' | '.join('%s: %s' % item for item in config.items())}")


class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, nb_proxy=1, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features * nb_proxy
        self.nb_proxy = nb_proxy
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()
        self.W_rand = None
        self.use_RP=False
        self.W2_rand = None

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input):
        if self.W_rand is not None:
            inn = torch.nn.functional.relu(input @ self.W_rand)
            if self.W2_rand is not None:
                inn = inn @ self.W2_rand
        else:
            inn=input
        out = F.linear(inn,self.weight)
            
        if self.sigma is not None:
            out = self.sigma * out

        return {'logits': out}
    

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        if config["peft"] == "lora":
            self.backbone = timm.create_model(config["pretrained_backbone"], pretrained=True, num_classes=0)
            self.freeze_backbone()
            backbone = timm.create_model(config["pretrained_backbone"], pretrained=True, num_classes=0)
            backbone.requires_grad_(False)
            lora_config = LoraConfig(
                r=config["r"],
                lora_alpha=config["lora_alpha"],
                lora_dropout=0.1,
                bias='none',
                target_modules=["qkv"],
            )
            self.backbone = get_peft_model(backbone, lora_config)
        elif config["peft"] == "ssf":
            self.backbone = timm.create_model(f'{config["pretrained_backbone"]}_ssf', pretrained=True, num_classes=0)
            self.freeze_backbone()
        
        self.fc = None
    
    def setup_RP(self):
        pass
        # self.head = RandomProjectionHead(self.backbone.num_features, config["proj_dim"], 20, init_matrix=config["init_matrix"], alpha=config["alpha"]).to(device)
        # self.head.W_rand = self.head.W_rand.to(device)
        # if self.head.W2_rand:
        #     self.head.W2_rand = self.head.W2_rand.to(device)
    
    def freeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            if ("ssf_scale_" not in name) and ("ssf_shift_" not in name):
                param.requires_grad = False
    
    def update_fc(self, nb_classes):
        fc = CosineLinear(self.backbone.num_features, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.backbone.num_features).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc
    
    def forward(self, x):
        f = self.backbone(x)
        y = self.fc(f)['logits']
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
        self.accuracy_matrix = []        

    def setup_RP(self):
        M, N = 10000, 768
        self.C = torch.zeros(N, 200)
        self.G = torch.zeros(N, N)
        self.I = torch.eye(N)
        
        W_rand = torch.rand(self.model.backbone.num_features, M).cuda()
        self.model.fc.weight = nn.Parameter(
            torch.Tensor(self.model.backbone.num_features, M).to(device="cuda")
        )
        self.model.fc.W_rand = W_rand
        
        probs = torch.rand(M, N).cuda()
        W2_rand = torch.zeros(M, N).cuda()
        sqrt_3 = torch.sqrt(torch.tensor(3.0))
        W2_rand[probs < 1/6] = sqrt_3
        W2_rand[(probs >= 1/6) & (probs < 1/3)] = -sqrt_3
        self.model.fc.W2_rand = W2_rand
        
        self.W_rand = copy.deepcopy(W_rand)
        self.W2_rand = copy.deepcopy(W2_rand)
    
    def learn(self, data_manager):
        self.data_manager = data_manager
        num_tasks = data_manager.nb_tasks - 1

        for task in range(num_tasks):
            self.before_task(task, data_manager)
            
            self.model.fc = None
            self.model.update_fc(self._classes_seen_so_far)
            if task > 0:
                self.model.fc.W_rand = self.W_rand
                self.model.fc.W2_rand = self.W2_rand
            
            if config["finetune"] == "fs":
                if task == 0:
                    trainset = data_manager.get_dataset(
                        np.arange(self._known_classes, self._classes_seen_so_far),
                        source="train", mode="train")
                    train_loader = DataLoader(trainset, batch_size=config["fine_tune_train_batch_size"], shuffle=True, num_workers=4)
                    self.finetune(train_loader)
                    self.setup_RP()
            elif config["finetune"] == "seq":
                trainset = data_manager.get_dataset(
                    np.arange(self._known_classes, self._classes_seen_so_far),
                    source="train", mode="train")
                train_loader = DataLoader(trainset, batch_size=config["fine_tune_train_batch_size"], shuffle=True, num_workers=4)
                self.finetune(train_loader)
            
            trainset_CPs = data_manager.get_dataset(
                np.arange(self._known_classes, self._classes_seen_so_far),
                source="train", mode="test")
            train_loader_CPs = DataLoader(trainset_CPs, batch_size=128, shuffle=False, num_workers=4)
            self.fit(train_loader_CPs)
            
            test_set = self.data_manager.get_dataset(
                np.arange(0, self._classes_seen_so_far), 
                source="test", mode="test")
            test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4)
            self.eval(test_loader)
            self.after_task()
    
    def before_task(self, task, data_manager):
        print(self.model)
        self._classes_seen_so_far = (self._known_classes + data_manager.get_task_size(task))
        self._class_increments.append((self._known_classes, self._classes_seen_so_far - 1))
                
    def after_task(self):
        self._known_classes = self._classes_seen_so_far
    
    def fit(self, train_loader):
        Features_f = []
        label_list = []
        with torch.no_grad():
            for i, (_, x, y) in tqdm(enumerate(train_loader)):
                x, y = x.cuda(), y.cuda()
                embedding = self.model.backbone(x)
                Features_f.append(embedding.cpu())
                label_list.append(y.cpu())
        
        Features_f = torch.cat(Features_f, dim=0)
        label_list = torch.cat(label_list, dim=0)
        
        Features_h = F.relu(Features_f @ self.model.fc.W_rand.cpu()) @ self.model.fc.W2_rand.cpu()
        Y = F.one_hot(label_list, num_classes=200).float()
        
        self.G = self.G + Features_h.T @ Features_h
        self.C = self.C + Features_h.T @ Y
        ridge = self.optimise_ridge_parameter(Features_h, Y)
        Wo = torch.linalg.solve(self.G + ridge * self.I, self.C).T
        self.model.fc.weight.data = Wo[0 : self.model.fc.weight.shape[0], :].to(device="cuda")
    
    def eval(self, test_loader):
        y_pred, y_true = [], []
        with torch.no_grad():
            for _, (_, x, y) in tqdm(enumerate(test_loader)):
                x, y = x.cuda(), y.cuda()
                logits = self.model(x)
                predicts = torch.topk(logits, k=1, dim=1, largest=True, sorted=True)[1]
                y_pred.append(predicts.cpu().numpy())
                y_true.append(y.cpu().numpy())
            y_pred = np.concatenate(y_pred)
            y_true = np.concatenate(y_true)
            acc_total, grouped = accuracy(y_pred.T[0], y_true, self._class_increments)
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
    
    def finetune(self, train_loader):
        base_lr = 1e-2
        weight_decay = 5e-4
        min_lr = 0.0
        optimizer = optim.SGD(self.model.parameters(), lr=base_lr, momentum=0.9, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=fine_tune_epochs, eta_min=min_lr)
        
        pbar = tqdm(range(20))
        for _, epoch in enumerate(pbar):
            self.model.train()
            total_loss, total, correct = 0, 0, 0
            
            for i, (_, x, y) in enumerate(train_loader):
                x, y = x.cuda(), y.cuda()
                y = torch.where(y - self._known_classes >= 0, y - self._known_classes, -100)
                logits = self.model(x)
                
                loss = F.cross_entropy(logits, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                correct += (logits.argmax(dim=1) == y).sum().item()
                total += len(y)
                
                info = f"Epoch {epoch}, Samples: {total * 100 / len(train_loader.dataset):.2f}, Loss: {total_loss / total:.4f}, Acc: {correct * 100 / total:.2f}"
                pbar.set_description(info)
                
            scheduler.step()
        logger.info(f"Train accuracy: {correct * 100 / total:.2f}")

for seed in [1993]:
    logger.info(f"Seed: {seed}")
    set_random(1)

    data_manager = DataManager(config["dataset_name"], True, seed, 20, 20, False)
    
    for epoch in config["fine_tune_train_epochs"]:
        fine_tune_epochs = epoch
        learner = Learner()
        learner.learn(data_manager)    