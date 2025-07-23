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
from inc_net import CosineLinear
from util import compute_metrics, accuracy, set_random
import gc
import time
import math
from merging.task_vectors import TaskVector, merge_max_abs
from merging.ties import merge_methods, state_dict_to_vector, vector_to_state_dict

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

def get_backbone(config):
    if "lora" in config["model_backbone"]:
        if config["model_backbone"] == "vit_base_patch16_224_lora":
            model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        elif config["model_backbone"] == "vit_base_patch16_224_in21k_lora":
            model = timm.create_model("vit_base_patch16_224_in21k", pretrained=True, num_classes=0)
        model.requires_grad_(False)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["qkv"],
            lora_dropout=0.0,
            bias="none"
        )
        model = get_peft_model(model, lora_config)
        return model
    elif "adapter" in config["model_backbone"]:
        tuning_config = EasyDict(
            # AdaptFormer
            ffn_adapt=True,
            ffn_option="parallel",
            ffn_adapter_layernorm_option="none",
            ffn_adapter_init_option="lora",
            ffn_adapter_scalar="0.1",
            ffn_num=64,
            d_model=768,
            # VPT related
            vpt_on=False,
            vpt_num=0,
        )
        if config["model_backbone"] == "vit_base_patch16_224_adapter":
            model = vision_transformer_adapter.vit_base_patch16_224_adapter(num_classes=0,
                global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
        elif config["model_backbone"] == "vit_base_patch16_224_in21k_adapter":
            model = vision_transformer_adapter.vit_base_patch16_224_in21k_adapter(num_classes=0,
                global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
        return model
    elif "ssf" in config["model_backbone"]:
        if config["model_backbone"] == "vit_base_patch16_224_ssf":
            model = timm.create_model("vit_base_patch16_224_ssf", pretrained=True, num_classes=0)
        elif config["model_backbone"] == "vit_base_patch16_224_in21k_ssf":
            model = timm.create_model("vit_base_patch16_224_in21k_ssf", pretrained=True, num_classes=0)
        
        for name, param in model.named_parameters():
            if "ssf_scale_" in name:
                nn.init.ones_(param)
            elif "ssf_shift_" in name:
                nn.init.zeros_(param)
            else:
                param.requires_grad_(False)
        return model

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
    
    def __repr__(self, x):
        return f"DyT(a={self.a}, g={self.g.mean()}, b={self.b.mean()})"

class RunningStatNorm(nn.Module):
    def __init__(self, num_features, momentum=0.01, eps=1e-5):
        super(RunningStatNorm, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        nn.init.normal_(self.gamma, mean=1, std=0.02)
        nn.init.normal_(self.beta, std=0.02)

    def forward(self, x):
        if self.training:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)

            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

            mean = batch_mean
            var = batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        return x_normalized * self.gamma + self.beta
    
    def __repr__(self):
        return f"RunningStatNorm(running_mean={self.running_mean.mean()}, running_var={self.running_var.mean()}, gamma={self.gamma.mean()}, beta={self.beta.mean()})"


class WelfordNormBatch(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(WelfordNormBatch, self).__init__()
        self.num_features = num_features
        self.eps = eps

        self.register_buffer('count', torch.tensor(0., dtype=torch.float))
        self.register_buffer('mean', torch.zeros(num_features))
        self.register_buffer('m2', torch.zeros(num_features))

        self.alpha = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
        nn.init.normal_(self.gamma, mean=1, std=0.02)
        nn.init.normal_(self.beta, std=0.02)

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                batch_mean = x.mean(dim=0)
                batch_var = x.var(dim=0, unbiased=False)
                batch_count = x.size(0)

                self._update_stats(batch_mean, batch_var, batch_count)

        # Compute variance and normalize
        var = self.m2 / self.count.clamp(min=1.0)
        std = torch.sqrt(var + self.eps)
        x_norm = (x - self.mean) / std
        # return self.gamma * torch.tanh(x_norm * self.alpha) + self.beta
        return x_norm

    def _update_stats(self, batch_mean, batch_var, batch_count):
        # Based on Welford's batch update formula
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * (batch_count / total_count)
        m2_delta = self.m2 + batch_var * batch_count + (delta**2) * (self.count * batch_count / total_count)

        self.mean.copy_(new_mean)
        self.m2.copy_(m2_delta)
        self.count += batch_count
    
    def __repr__(self):
        return f"WelfordNorm(count={self.count}, mean={self.mean.mean()}, m2={self.m2.mean()}, gamma={self.gamma.mean()}, beta={self.beta.mean()})"

@torch.no_grad()
def merge_welford_norm(norm1, norm2):
    merged = WelfordNormBatch(norm1.num_features)
    
    n1 = norm1.count
    n2 = norm2.count
    n = n1 + n2

    if n == 0:
        return merged  # return empty stats

    mu1 = norm1.mean
    mu2 = norm2.mean
    m2_1 = norm1.m2
    m2_2 = norm2.m2

    delta = mu2 - mu1
    merged.mean = (n1 * mu1 + n2 * mu2) / n
    merged.m2 = m2_1 + m2_2 + (delta ** 2) * (n1 * n2 / n)
    merged.count = n

    merged.alpha.data = (norm1.alpha.data + norm2.alpha.data) / 2
    merged.gamma.data = (norm1.gamma.data + norm2.gamma.data) / 2
    merged.beta.data = (norm1.beta.data + norm2.beta.data) / 2

    return merged

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self.backbone = get_backbone(config)
        # self.norm = DyT(self.backbone.num_features, 1.0) if config["model_norm"] else None
        # self.norm = RunningStatNorm(self.backbone.num_features, 0.05) if config["model_norm"] else None
        self.norm = WelfordNormBatch(self.backbone.num_features) if config["model_norm"] else None
        self.head = None
        self.fc = None
        
        if self._config["model_concat"]:
            self.base_backbone = get_backbone(config)
            self.base_backbone.requires_grad_(False)
    
    @property
    def feature_dim(self):
        return self.backbone.num_features
    
    def update_fc(self, num_total_classes):
        feature_dim = self.feature_dim if not self._config["model_concat"] else self.feature_dim * 2
        
        fc = CosineLinear(feature_dim, num_total_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            weight = torch.cat([weight, torch.zeros(num_total_classes - nb_output, self.fc.weight.shape[1]).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc
    
    def update_head(self, num_classes):
        if self.head == None:
            self.head = ContinualLearnerHead(self.backbone.num_features, num_classes)
        else:
            self.head.update(num_classes)
        self.head.cuda()
    
    def get_backbone_trainable_params(self):
        params = {}
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                params[name] = param
        return params
    
    def get_features(self, x):
        f = self.backbone(x)
        if self.norm != None:
            f = self.norm(f)
        if not self.training and self._config["model_concat"]:
            base_f = self.base_backbone(x)
            f = torch.cat((f, base_f), dim=1)
        return f
    
    def forward(self, x):
        # if self.fc == None:
        #     f = self.get_features(x)
        #     y = self.head(f)
        #     return y
        
        # f = self.backbone(x)
        # if self.norm != None:
        #     f = self.norm(f)
        # y = self.head(f)
        # y_fc = self.fc(self.get_features(x))
        # cs_logits = y['logits']
        # cp_logits = y_fc['logits']
        
        # norm_cs_logits = F.normalize(cs_logits, p=2, dim=1)
        # norm_cp_logits = F.normalize(cp_logits, p=2, dim=1)
        # max_cs_logits, cs_indices = torch.max(norm_cs_logits, dim=1)
        # max_cp_logits, cp_indices = torch.max(norm_cp_logits, dim=1)
        # mask = max_cs_logits > max_cp_logits
        # preds = torch.where(mask, cs_indices, cp_indices)
        
        f = self.get_features(x)
        y = self.head(f) if self.fc == None else self.fc(f)
        
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

def trim(tensor, topk=100):
    flattened = tensor.view(-1)
    magnitudes = torch.abs(flattened)
    num_keep = max(1, int(len(flattened) * topk / 100))
    threshold = torch.topk(magnitudes, num_keep, largest=True, sorted=True).values[-1]
    mask = magnitudes >= threshold
    trimmed = torch.where(mask, flattened, torch.tensor(0.0, dtype=tensor.dtype))
    
    gamma = torch.sign(trimmed)
    mu = torch.abs(trimmed)
    
    return (trimmed.view_as(tensor), gamma.view_as(tensor), mu.view_as(tensor))

def merge_task_vectors(trimmed_task_vectors):
    gamma_tvs = torch.stack([tv[1] for tv in trimmed_task_vectors], dim=0)
    gamma = torch.sign(gamma_tvs.sum(dim=0))
    mask = (gamma_tvs == gamma)
    tau_tvs = torch.stack([tv[0] for tv in trimmed_task_vectors], dim=0)
    mean_tvs = torch.where(mask, tau_tvs, torch.tensor(0.0, dtype=tau_tvs.dtype)).sum(dim=0) / mask.sum(dim=0).clamp(min=1)

    return mean_tvs

def merge(base_params, tasks_params, method="ties", lamb=1.0, topk=100):
    params = {}
    for name in base_params:
        base_tv = copy.deepcopy(base_params[name])
        task_vectors = [copy.deepcopy(task_params[name]) for task_params in tasks_params]
        tvs = [tv - base_tv for tv in task_vectors]
        if method == "ties":
            tvs = [trim(tv, topk) for tv in tvs]
            merged_tv = merge_task_vectors(tvs)
        elif method == "avg":
            merged_tv = torch.mean(torch.stack(tvs, dim=0), dim=0)
        elif method == "max":
            merged_tv = torch.max(torch.stack(tvs, dim=0), dim=0)[0]
        params[name] = base_tv + lamb * merged_tv
        
    return params

class Learner:
    def __init__(self, config):
        self._config = config
        self._known_classes = 0
        self._total_classes = 0
        self._class_increments = []
        self._cur_task = -1
        self.accuracy_matrix = []
        
        self.model = Model(config)
        torch.save(self.model.backbone.state_dict(), backbone_base())
        self.model.cuda()
        self.model.eval()
        
        self._faa, self._ffm = 0, 0
    
    def learn(self, data_manager):
        self.data_manager = data_manager
        num_tasks = data_manager.nb_tasks
        self._total_classnum = data_manager.get_total_classnum()

        for task in range(num_tasks):
            self.before_task(task, data_manager)
            self.train()
            # self.fit()
            self.merge()
            self.eval()
            self.after_task()
        
    def before_task(self, task, data_manager):
        self._total_classes = self._known_classes + data_manager.get_task_size(task)
        self._class_increments.append((self._known_classes, self._total_classes - 1))
        self._cur_task = task

    def after_task(self):
        self._known_classes = self._total_classes
    
    def eval(self):
        test_set = self.data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=4)
        
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

    def cosine_loss(self, task_model):
        params1 = self.model.get_backbone_trainable_params()
        params2 = task_model.get_backbone_trainable_params()

        cosine_losses = []

        # Only compare matching parameter names
        for name in params1.keys():
            if name in params2:
                p1 = params1[name]
                p2 = params2[name]

                if p1.requires_grad and p2.requires_grad:
                    p1_flat = p1.view(-1)
                    p2_flat = p2.view(-1)

                    # Avoid division by zero in degenerate cases
                    if p1_flat.norm() > 0 and p2_flat.norm() > 0:
                        p1_norm = F.normalize(p1_flat, dim=0)
                        p2_norm = F.normalize(p2_flat, dim=0)
                        cos_sim = torch.dot(p1_norm, p2_norm)
                        cosine_losses.append(1.0 - cos_sim)

        if cosine_losses:
            return torch.stack(cosine_losses).mean()
        else:
            return torch.tensor(0.0, requires_grad=True).cuda()
    
    def entropy_loss(self, logits, temperature=1.0):
        probs = F.softmax(logits / temperature, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        return entropy.mean()
    
    def train(self):
        trainset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="train")
        train_loader = DataLoader(trainset, batch_size=self._config["train_batch_size"], shuffle=True, num_workers=4)
        
        task_model = Model(self._config).cuda()
        # if self._cur_task > 0:
        #     task_model.backbone.load_state_dict(self.model.backbone.state_dict(), strict=True)
            
            # if task_model.norm != None:
            #     # task_model.norm.g.data.copy_(self.model.norm.g.data)
            #     # task_model.norm.b.data.copy_(self.model.norm.b.data)
            #     task_model.norm.load_state_dict(self.model.norm.state_dict(), strict=True)
            
        task_model.update_head(self._total_classes - self._known_classes)
        print(task_model)
        
        epochs = self._config["train_epochs"]
        min_lr = self._config["train_min_lr"]
        base_lr = self._config["train_lr"]
        
        optimizer = optim.SGD(task_model.parameters(), lr=base_lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
        
        pbar = tqdm(range(epochs))
        for _, epoch in enumerate(pbar):
            task_model.train()
            total_loss, total_acc, total = 0, 0, 0

            for _, (_, _, x, y) in enumerate(train_loader):
                x, y = x.cuda(), y.cuda()
                y = torch.where(y - self._known_classes >= 0, y - self._known_classes, -100)
                
                logits = task_model(x)['logits']
                loss = F.cross_entropy(logits, y)
                                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_acc += (logits.argmax(dim=1) == y).sum().item()
                total += len(y)
                
                info = f"Task {self._cur_task + 1}, Epoch {epoch + 1}, Loss: {total_loss / total:.4f}, Acc: {total_acc / total:.4f}"
                pbar.set_description(info)  
                
            scheduler.step()
            
        if self.model.norm != None:
            # if self._config["model_norm_coeff"] == "n":
            #     self.model.norm.a.data.copy_((self.model.norm.a.data * self._cur_task + task_model.norm.a.data) / (self._cur_task + 1)) # good
            # else:
            #     self.model.norm.a.data.copy_((self.model.norm.a.data * self._config["model_norm_coeff"] + task_model.norm.a.data * (1 - self._config["model_norm_coeff"])))
            #     self.model.norm.g.data.copy_((self.model.norm.g.data * self._config["model_norm_coeff"] + task_model.norm.g.data * (1 - self._config["model_norm_coeff"])))
            #     self.model.norm.b.data.copy_((self.model.norm.b.data * self._config["model_norm_coeff"] + task_model.norm.b.data * (1 - self._config["model_norm_coeff"])))
                
            # self.model.norm.g.data.copy_(task_model.norm.g.data)
            # self.model.norm.b.data.copy_(task_model.norm.b.data)
            # self.model.norm.load_state_dict(task_model.norm.state_dict(), strict=True)
            
            norm = merge_welford_norm(self.model.norm, task_model.norm) if self._cur_task > 0 else task_model.norm
            self.model.norm.load_state_dict(norm.state_dict(), strict=True)
            print(f"Norm: {self.model.norm}")
        
        if self._config["model_ncm"]:
            self.model.update_fc(self._total_classes)
            
            trainset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="test")
            train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
            task_model.eval()
            embedding_list = []
            label_list = []
            with torch.no_grad():
                for i, batch in tqdm(enumerate(train_loader)):
                    (_, _, data,label)=batch
                    data=data.cuda()
                    label=label.cuda()
                    embedding = task_model.get_features(data)
                    embedding_list.append(embedding.cpu())
                    label_list.append(label.cpu())
            embedding_list = torch.cat(embedding_list, dim=0)
            label_list = torch.cat(label_list, dim=0)

            class_list=np.unique(trainset.labels)
            for class_index in class_list:
                data_index=(label_list==class_index).nonzero().squeeze(-1)
                embedding=embedding_list[data_index]
                proto=embedding.mean(0)
                self.model.fc.weight.data[class_index]=proto
        else:
            self.model.update_head(self._total_classes - self._known_classes)
            self.model.head.heads[-1].load_state_dict(task_model.head.heads[0].state_dict(), strict=True)
        
        task_model.cpu()
        torch.save(task_model.backbone.state_dict(), backbone_checkpoint(self._cur_task))
        
        del task_model
        gc.collect()
        torch.cuda.empty_cache()
    
    def merge(self):
        base_params = torch.load(backbone_base())
        tasks_params = [torch.load(backbone_checkpoint(task)) for task in range(self._cur_task + 1)]
        backbone_params = merge(base_params, tasks_params, lamb=self._config["model_merge_coef"], method=self._config["model_merge"], topk=self._config["model_merge_topk"])
        self.model.backbone.load_state_dict(backbone_params, strict=False)
        self.model.cuda()
    
    def fit(self):
        self.model.update_fc(self._total_classes)
        
        if self._cur_task == 0:
            M = self._config["model_M"]
            self.model.fc.weight.data = torch.zeros(self._total_classes, M).cuda()
            self.model.fc.W_rand = torch.randn(self.model.feature_dim, M).cuda()
            self.W_rand = copy.deepcopy(self.model.fc.W_rand)
            self.Q = torch.zeros(M, self._total_classnum)
            self.G = torch.zeros(M, M)
            self.H = torch.zeros(0, M)
            self.Y = torch.zeros(0, self._total_classnum)
        else:
            self.model.fc.W_rand = self.W_rand
        
        self.model.backbone.load_state_dict(torch.load(backbone_checkpoint(self._cur_task)), strict=False)
        self.model.fc.use_RP = True
        
        trainset_CPs = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="test")
        train_loader_CPs = DataLoader(trainset_CPs, batch_size=512, shuffle=False, num_workers=4)
        
        fs = []
        ys = []
        with torch.no_grad():
            for _, (_, _, x, y) in tqdm(enumerate(train_loader_CPs)):
                x, y = x.cuda(), y.cuda()
                f = self.model.get_features(x)
                fs.append(f.cpu())
                ys.append(y.cpu())
                
        fs = torch.cat(fs, dim=0)
        ys = torch.cat(ys, dim=0)
        
        Y = F.one_hot(ys, self._total_classnum).float()
        H = F.relu(fs @ self.model.fc.W_rand.cpu())
        
        dG = H.T @ H
        diag_mask = torch.eye(dG.size(0), dtype=torch.bool, device=dG.device)
        dG = dG * (~diag_mask * 0.9 + diag_mask.float())
        
        self.Q += H.T @ Y
        self.G += dG
        
        self.H = torch.cat([self.H, H], dim=0)
        self.Y = torch.cat([self.Y, Y], dim=0)
        
        ridge = self.optimise_ridge_parameter(self.H, self.Y)
        Wo = torch.linalg.solve(self.G + ridge * torch.eye(self.G.size(dim=0)), self.Q).T
        self.model.fc.weight.data = Wo[0 : self._total_classes, :].cuda()
    
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
    "cub": (10, 10),
    "imagenetr": (20, 20),
    "imageneta": (20, 20)
}

for dataset_name in ["cub"]:
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
            "model_backbone": "vit_base_patch16_224_in21k_lora",
            "model_M": 10000,
            "model_merge": "max",
            "model_merge_coef": 1.0,
            "model_merge_topk": 100,
            "model_norm": False,
            "model_norm_coeff": 0.9,
            "model_ncm": True,
            "model_concat": False
        }
        config.update(model_config)
        
        train_config = {
            "train_epochs": 5,
            "train_batch_size": 48,
            "train_min_lr": 0.0,
            "train_lr": 1e-2,
            "train_seq": True
        }
        
        config.update(train_config)
        
        for item in config.items():
            logger.info(f"{item[0]}: {item[1]}")
        
        learner = Learner(config)
        learner.learn(data_manager)
            
        faa.append(learner._faa)
        ffm.append(learner._ffm)
    
    logger.info(f"FAA: {faa}, FFM: {ffm}")    
    logger.info(f"End experiment in {time.time() - start_time}\n")
    
# 2025-03-20 01:40:30,601 - seed: 1993
# 2025-03-20 01:40:30,601 - dataset_name: imagenetr
# 2025-03-20 01:40:30,601 - dataset_init_cls: 20
# 2025-03-20 01:40:30,601 - dataset_increment: 20
# 2025-03-20 01:40:30,601 - model_backbone: supervised_imagenet_1k
# 2025-03-20 01:40:30,601 - model_finetune: lora
# 2025-03-20 01:40:30,601 - model_lora_rank: 16
# 2025-03-20 01:40:30,601 - model_lora_alpha: 16
# 2025-03-20 01:40:30,602 - model_with_bias: True
# 2025-03-20 01:40:30,602 - model_affine: dyt
# 2025-03-20 01:40:30,602 - train_epochs: 20
# 2025-03-20 01:40:30,602 - train_batch_size: 48
# 2025-03-20 01:40:30,602 - train_min_lr: 0.0
# 2025-03-20 01:40:30,602 - train_lr: 0.01
# 2025-03-20 01:40:30,602 - train_entropy_weight: 0.0
# 2025-03-20 01:40:30,602 - train_finetune: last
# 2025-03-20 01:40:30,602 - train_merge: ties
# 2025-03-20 01:40:30,602 - train_topk: 20
# 2025-03-20 01:46:01,468 - Total Acc: 93.18, Grouped Acc: [93.18]
# 2025-03-20 01:46:01,468 - Final Average Accuracy (FAA): 93.18
# 2025-03-20 01:46:01,468 - Final Forgetting Measure (FFM): 0.00
# 2025-03-20 01:51:07,016 - Total Acc: 89.70, Grouped Acc: [91.15, 88.13]
# 2025-03-20 01:51:07,017 - Final Average Accuracy (FAA): 89.64
# 2025-03-20 01:51:07,017 - Final Forgetting Measure (FFM): 2.03
# 2025-03-20 01:55:50,283 - Total Acc: 87.60, Grouped Acc: [91.15, 86.71, 84.49]
# 2025-03-20 01:55:50,283 - Final Average Accuracy (FAA): 87.45
# 2025-03-20 01:55:50,283 - Final Forgetting Measure (FFM): 1.73
# 2025-03-20 02:00:30,191 - Total Acc: 85.55, Grouped Acc: [90.86, 85.28, 84.16, 80.91]
# 2025-03-20 02:00:30,191 - Final Average Accuracy (FAA): 85.30
# 2025-03-20 02:00:30,191 - Final Forgetting Measure (FFM): 1.83
# 2025-03-20 02:04:58,273 - Total Acc: 83.58, Grouped Acc: [89.84, 84.97, 83.66, 80.91, 75.98]
# 2025-03-20 02:04:58,273 - Final Average Accuracy (FAA): 83.07
# 2025-03-20 02:04:58,273 - Final Forgetting Measure (FFM): 1.83
# 2025-03-20 02:10:23,316 - Total Acc: 82.72, Grouped Acc: [86.94, 84.18, 82.34, 79.68, 75.56, 85.15]
# 2025-03-20 02:10:23,316 - Final Average Accuracy (FAA): 82.31
# 2025-03-20 02:10:23,316 - Final Forgetting Measure (FFM): 2.80
# 2025-03-20 02:14:48,980 - Total Acc: 82.13, Grouped Acc: [86.36, 82.75, 82.01, 80.04, 76.39, 84.7, 80.49]
# 2025-03-20 02:14:48,981 - Final Average Accuracy (FAA): 81.82
# 2025-03-20 02:14:48,981 - Final Forgetting Measure (FFM): 2.67
# 2025-03-20 02:20:34,528 - Total Acc: 81.89, Grouped Acc: [85.49, 81.17, 81.19, 80.56, 75.56, 84.09, 81.2, 83.54]
# 2025-03-20 02:20:34,528 - Final Average Accuracy (FAA): 81.60
# 2025-03-20 02:20:34,528 - Final Forgetting Measure (FFM): 2.88
# 2025-03-20 02:24:44,503 - Total Acc: 81.09, Grouped Acc: [85.63, 81.96, 79.87, 80.04, 73.92, 83.79, 81.72, 84.09, 74.48]
# 2025-03-20 02:24:44,504 - Final Average Accuracy (FAA): 80.61
# 2025-03-20 02:24:44,504 - Final Forgetting Measure (FFM): 2.88
# 2025-03-20 02:29:32,232 - Total Acc: 80.10, Grouped Acc: [84.03, 79.43, 76.9, 80.21, 73.92, 82.88, 82.07, 83.68, 75.52, 78.76]
# 2025-03-20 02:29:32,232 - Final Average Accuracy (FAA): 79.74
# 2025-03-20 02:29:32,232 - Final Forgetting Measure (FFM): 3.48
# 2025-03-20 02:29:32,237 - End experiment in 2941.636727809906