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
from peft import LoraConfig, get_peft_model
import copy
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
from petl import vision_transformer_ssf
from petl import vision_transformer_adapter
from easydict import EasyDict
from _exp import ContinualLearnerHead
from util import compute_metrics, set_random, accuracy
# from utils.toolkit import accuracy
from merging.task_vectors import TaskVector, merge_max_abs, merge_avg
from merging.ties import merge_methods, state_dict_to_vector, vector_to_state_dict
from inc_net import BaseNet, CosineLinear
import gc
import time


os.makedirs("logs/checkpoints", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

def setup_logger(log_file=f'logs/{timestamp}_exp45.log'):
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

def get_convnet(config):
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

class SimpleVitNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self.convnet = get_convnet(config)        
        self.norm = DyT(768, 1.0) if config["model_norm"] else None
        self.fc = None

    @property
    def feature_dim(self):
        return 768
    
    def update_fc(self, nb_classes):
        fc = CosineLinear(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.fc.weight.shape[1]).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def get_features(self, x):
        f = self.convnet(x)
        if self.norm != None:
            f = self.norm(f)
        return f
    
    def forward(self, x):
        f = self.get_features(x)
        out = self.fc(f)
        return out
    
    def show_num_params(self):
        trainable_params = count_parameters(self, trainable=True)
        total_params = count_parameters(self)
        return f"Model(trainable_params={trainable_params}, total_params={total_params}, percentage={trainable_params * 100 / total_params:.2f})"

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self.backbone = get_convnet(config)        
        self.norm = DyT(768, 1.0) if config["model_norm"] else None
        self.head = None
    
    @property
    def feature_dim(self):
        return 768
    
    def update_head(self, num_classes):
        if self.head == None:
            self.head = ContinualLearnerHead(self.feature_dim, num_classes)
        else:
            self.head.update(num_classes)
        self.head.cuda()
    
    def apply_lora(self):
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["qkv"],
            lora_dropout=0.0,
            bias="none"
        )
        self.backbone = get_peft_model(self.backbone, lora_config)
    
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
        return f
    
    def forward(self, x):
        f = self.get_features(x)
        y = self.head(f)
        return y
    
    def __repr__(self):
        trainable_params = count_parameters(self, trainable=True)
        total_params = count_parameters(self)
        return f"Model(trainable_params={trainable_params}, total_params={total_params}, percentage={trainable_params * 100 / total_params:.2f})"


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

def merge(base_params, tasks_params, method, lamb=1.0, topk=100):
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
            stacked = torch.stack(tvs, dim=0)
            abs_stacked = stacked.abs()
            indices = torch.argmax(abs_stacked, dim=0)
            merged_tv = torch.gather(stacked, 0, indices.unsqueeze(0)).squeeze(0)

        params[name] = base_tv + lamb * merged_tv
        
    return params

class Learner:
    def __init__(self, config):
        self._config = config
        self._known_classes = 0
        self._total_classes = 0
        self._class_increments = []
        self._cur_task = -1
        
        self._cs_accuracy_matrix = []
        self._cp_accuracy_matrix = []
        self._accuracy_matrix = []
        
        self.cs_model = Model(config)
        torch.save(self.cs_model.get_backbone_trainable_params(), self.backbone_base())
        self.cs_model.cuda()
        
        self.cp_model = SimpleVitNet(config).cuda()
        
        self._faa, self._ffm = 0, 0

    def learn(self, data_manager):
        self.data_manager = data_manager
        num_tasks = data_manager.nb_tasks
        self._total_classnum = data_manager.get_total_classnum()

        for task in range(num_tasks):
            logger.info(f"Start task {task + 1}")
            self.before_task(task)
            self.train()
            self.fit()
            self.merge()
            self.eval()
            self.after_task()
    
    def test_fit(self):
        M = self._config["model_M"]
        self.cp_model.fc.W_rand = torch.randn(self.cp_model.feature_dim, M).cuda()
        self.Q = torch.zeros(M, self._total_classnum)
        self.G = torch.zeros(M, M)
        
        Hs, Ys = [], []
        for task in range(10):
            self.cp_model.convnet.load_state_dict(torch.load(self.backbone_checkpoint(task)), strict=False)
            self.cp_model.fc.use_RP = True
            
            trainset_CPs = data_manager.get_dataset(np.arange(task*10, (task+1)*10), source="train", mode="test")
            train_loader_CPs = DataLoader(trainset_CPs, batch_size=512, shuffle=False, num_workers=4)
            
            fs = []
            ys = []
            with torch.no_grad():
                for _, (_, _, x, y) in tqdm(enumerate(train_loader_CPs)):
                    x, y = x.cuda(), y.cuda()
                    f = self.cp_model.get_features(x)
                    fs.append(f.cpu())
                    ys.append(y.cpu())
                    
            fs = torch.cat(fs, dim=0)
            ys = torch.cat(ys, dim=0)
            
            Y = F.one_hot(ys, self._total_classnum).float()
            H = F.relu(fs @ self.cp_model.fc.W_rand.cpu())
            
            dG = H.T @ H
            diag_mask = torch.eye(dG.size(0), dtype=torch.bool, device=dG.device)
            dG = dG * (~diag_mask * 0.1 + diag_mask.float())
            
            self.Q += H.T @ Y
            self.G += dG
            
            Hs.append(H)
            Ys.append(Y)
        
        Hs = torch.cat(Hs, dim=0)
        Ys = torch.cat(Ys, dim=0)
        
        ridge = self.optimise_ridge_parameter(Hs, Ys)
        
        Wo = torch.linalg.solve(self.G + ridge * torch.eye(self.G.size(dim=0)), self.Q).T
        self.cp_model.fc.weight.data = Wo[0 : self._total_classes, :].cuda()
        
        # # disjoint assignment
        # M = self._config["model_M"]
        # self.cp_model.fc.W_rand = torch.randn(self.cp_model.feature_dim, M).cuda()
        
        # for task in range(10):
        #     self.cp_model.convnet.load_state_dict(torch.load(self.backbone_checkpoint(0)), strict=False)
        #     self.cp_model.fc.use_RP = True
            
        #     trainset_CPs = data_manager.get_dataset(np.arange(task*10, (task+1)*10), source="train", mode="test")
        #     train_loader_CPs = DataLoader(trainset_CPs, batch_size=256, shuffle=False, num_workers=4)
            
        #     fs = []
        #     ys = []
        #     with torch.no_grad():
        #         for _, (_, _, x, y) in tqdm(enumerate(train_loader_CPs)):
        #             x, y = x.cuda(), y.cuda()
        #             f = self.cp_model.convnet(x)
        #             fs.append(f.cpu())
        #             ys.append(y.cpu())
                    
        #     fs = torch.cat(fs, dim=0)
        #     ys = torch.cat(ys, dim=0)
            
        #     Y = F.one_hot(ys, self._total_classnum).float()
        #     H = F.relu(fs @ self.cp_model.fc.W_rand.cpu())
            
        #     G = H.T @ H
        #     diag_mask = torch.eye(G.size(0), dtype=torch.bool, device=G.device)
        #     G = G * (~diag_mask * 0.9 + diag_mask.float())
        #     Q = H.T @ Y
            
        #     ridge = self.optimise_ridge_parameter(H, Y)
        #     Wo = torch.linalg.solve(G + ridge * torch.eye(G.size(dim=0)), Q).T
        #     self.cp_model.fc.weight.data = Wo[task*10 : (task+1)*10, :].cuda()
    
    def before_task(self, task):
        self._total_classes = self._known_classes + self.data_manager.get_task_size(task)
        self._class_increments.append((self._known_classes, self._total_classes - 1))
        self._cur_task = task
        
        self.cs_model.update_head(self._total_classes - self._known_classes)
        
        del self.cp_model.fc
        self.cp_model.fc = None
        self.cp_model.update_fc(self._total_classes)

    def after_task(self):
        self._known_classes = self._total_classes
    
    def eval(self):
        test_set = self.data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=4)
        
        self.cs_model.eval()
        self.cp_model.eval()
        
        y_true, y_cs_pred, y_cp_pred = [], [], []
        y_pred = []
        
        with torch.no_grad():
            for _, (_, _, x, y) in tqdm(enumerate(test_loader)):
                x, y = x.cuda(), y.cuda()
                
                cs_logits = self.cs_model(x)['logits']
                cp_logits = self.cp_model(x)['logits']
                
                cs_preds = cs_logits.argmax(dim=1)
                cp_preds = cp_logits.argmax(dim=1)
                
                y_cs_pred.append(cs_preds.cpu().numpy())
                y_cp_pred.append(cp_preds.cpu().numpy())
                
                norm_cs_logits = F.normalize(cs_logits, p=2, dim=1)
                norm_cp_logits = F.normalize(cp_logits, p=2, dim=1)
                max_cs_logits, cs_indices = torch.max(norm_cs_logits, dim=1)
                max_cp_logits, cp_indices = torch.max(norm_cp_logits, dim=1)
                mask = max_cs_logits > max_cp_logits
                preds = torch.where(mask, cs_indices, cp_indices)
                y_pred.append(preds.cpu().numpy())
                
                y_true.append(y.cpu().numpy())
        
        y_pred = np.concatenate(y_pred)
        y_cs_pred = np.concatenate(y_cs_pred)
        y_cp_pred = np.concatenate(y_cp_pred)
        y_true = np.concatenate(y_true)
        
        acc_cs_total, cs_grouped = accuracy(y_cs_pred.T, y_true, self._class_increments)
        logger.info(f"Total CS Acc: {acc_cs_total:.2f}, Grouped Acc: {cs_grouped}")
        acc_cp_total, cp_grouped = accuracy(y_cp_pred.T, y_true, self._class_increments)
        logger.info(f"Total CP Acc: {acc_cp_total:.2f}, Grouped Acc: {cp_grouped}")
        
        acc_total, grouped = accuracy(y_pred.T, y_true, self._class_increments)
        logger.info(f"Total Acc: {acc_total:.2f}, Grouped Acc: {grouped}")

        self._cs_accuracy_matrix.append(cs_grouped)
        self._cp_accuracy_matrix.append(cp_grouped)
        
        self._accuracy_matrix.append(grouped)

        num_tasks = len(self._cs_accuracy_matrix)
        cs_accuracy_matrix = np.zeros((num_tasks, num_tasks))
        cp_accuracy_matrix = np.zeros((num_tasks, num_tasks))
        accuracy_matrix = np.zeros((num_tasks, num_tasks))
        for i in range(num_tasks):
            for j in range(i + 1):
                cs_accuracy_matrix[i, j] = self._cs_accuracy_matrix[i][j]
                cp_accuracy_matrix[i, j] = self._cp_accuracy_matrix[i][j]
                accuracy_matrix[i, j] = self._accuracy_matrix[i][j]
                

        cs_faa, cs_ffm = compute_metrics(cs_accuracy_matrix)
        logger.info(f"Final CS Average Accuracy (FAA): {cs_faa:.2f}, Final CS Forgetting Measure (FFM): {cs_ffm:.2f}")
        cp_faa, cp_ffm = compute_metrics(cp_accuracy_matrix)
        logger.info(f"Final CP Average Accuracy (FAA): {cp_faa:.2f}, Final CP Forgetting Measure (FFM): {cp_ffm:.2f}")
        
        faa, ffm = compute_metrics(accuracy_matrix)
        logger.info(f"Final Average Accuracy (FAA): {faa:.2f}, Final Forgetting Measure (FFM): {ffm:.2f}")
    
    def train(self):
        if not os.path.exists(self.backbone_checkpoint(self._cur_task)) or not os.path.exists(self.head_checkpoint(self._cur_task)) or self._config["reset"]:
            trainset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="train")
            train_loader = DataLoader(trainset, batch_size=self._config["train_batch_size"], shuffle=True, num_workers=4)
            
            task_model = Model(self._config).cuda()
            if self._config["train_seq"]:
                if self._cur_task > 0:
                    logger.info(f"Sequential training from task {self._cur_task}")
                    
                    backbone_params = torch.load(self.backbone_checkpoint(self._cur_task - 1))
                    task_model.backbone.load_state_dict(backbone_params, strict=False)
                    # task_model.backbone.load_state_dict(self.cs_model.backbone.state_dict(), strict=True)
                    
                    if task_model.norm != None:
                        norm_params = torch.load(self.norm_checkpoint(self._cur_task - 1))
                        task_model.norm.g.data.copy_(norm_params['g'])
                        task_model.norm.b.data.copy_(norm_params['b'])
                        # task_model.norm.load_state_dict(self.cs_model.norm.state_dict())
                        
            task_model.update_head(self._total_classes - self._known_classes)
            
            epochs = self._config["train_epochs"]
            optimizer = optim.SGD(task_model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
            
            pbar = tqdm(range(epochs))
            for _, epoch in enumerate(pbar):
                task_model.train()
                total_loss, total, total_acc = 0, 0, 0

                for _, (_, _, x, y) in enumerate(train_loader):
                    x, y = x.cuda(), y.cuda()
                    y = torch.where(y >= self._known_classes, y - self._known_classes, -100)
                    
                    logits = task_model(x)['logits']
                    loss = F.cross_entropy(logits, y)
                                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    total += len(y)
                    total_acc += (logits.argmax(dim=1) == y).sum().item()
                    
                    info = f"Task {self._cur_task + 1}, Epoch {epoch + 1}, Loss: {total_loss / total:.4f}, Acc: {total_acc * 100 / total:.2f}"
                    pbar.set_description(info)
                scheduler.step()
                
                # if epoch == 5 or epoch == 10:
                #     task_model.cpu()
                #     torch.save(task_model.get_backbone_trainable_params(), self.backbone_checkpoint(self._cur_task, epoch))
                #     torch.save(task_model.head.heads[-1].state_dict(), self.head_checkpoint(self._cur_task, epoch))
                #     task_model.cuda()
            
            task_model.cpu()
            torch.save(task_model.get_backbone_trainable_params(), self.backbone_checkpoint(self._cur_task))
            if self._config["model_norm"]:
                torch.save(task_model.norm.state_dict(), self.norm_checkpoint(self._cur_task))
            torch.save(task_model.head.heads[-1].state_dict(), self.head_checkpoint(self._cur_task))
            
            del task_model
            gc.collect()
            torch.cuda.empty_cache()

    def fit(self):
        if self._cur_task == 0:
            M = self._config["model_M"]
            self.cp_model.fc.weight.data = torch.zeros(self._total_classes, M).cuda()
            self.cp_model.fc.W_rand = torch.randn(self.cp_model.feature_dim, M).cuda()
            self.W_rand = copy.deepcopy(self.cp_model.fc.W_rand)
            self.Q = torch.zeros(M, self._total_classnum)
            self.G = torch.zeros(M, M)
            self.H = torch.zeros(0, M)
            self.Y = torch.zeros(0, self._total_classnum)
        else:
            self.cp_model.fc.W_rand = self.W_rand
        
        # self.cp_model.convnet.load_state_dict(torch.load(self.backbone_checkpoint(self._cur_task)), strict=False)
        self.cp_model.convnet.load_state_dict(torch.load(self.backbone_checkpoint(0)), strict=False)
        self.cp_model.fc.use_RP = True
        
        trainset_CPs = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="test")
        train_loader_CPs = DataLoader(trainset_CPs, batch_size=512, shuffle=False, num_workers=4)
        
        fs = []
        ys = []
        with torch.no_grad():
            for _, (_, _, x, y) in tqdm(enumerate(train_loader_CPs)):
                x, y = x.cuda(), y.cuda()
                f = self.cp_model.get_features(x)
                fs.append(f.cpu())
                ys.append(y.cpu())
                
        fs = torch.cat(fs, dim=0)
        ys = torch.cat(ys, dim=0)
        
        Y = F.one_hot(ys, self._total_classnum).float()
        H = F.relu(fs @ self.cp_model.fc.W_rand.cpu())
        
        dG = H.T @ H
        diag_mask = torch.eye(dG.size(0), dtype=torch.bool, device=dG.device)
        dG = dG * (~diag_mask * 0.9 + diag_mask.float())
        
        self.Q += H.T @ Y
        self.G += dG
        
        self.H = torch.cat([self.H, H], dim=0)
        self.Y = torch.cat([self.Y, Y], dim=0)
        
        ridge = self.optimise_ridge_parameter(self.H, self.Y)
        # ridge = self.optimise_ridge_parameter(H, Y)
        Wo = torch.linalg.solve(self.G + ridge * torch.eye(self.G.size(dim=0)), self.Q).T
        self.cp_model.fc.weight.data = Wo[0 : self._total_classes, :].cuda()
    
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
    
    def merge(self):
        # task_vectors = [
        #     TaskVector(self.backbone_base(), self.backbone_checkpoint(task)) for task in range(self._cur_task + 1)]
        
        # # task_vectors = []
        # # task_vectors.append(TaskVector(self.backbone_base(), self.backbone_checkpoint(self._cur_task)))
        
        # # params = {}
        # # if self._cur_task > 0:
        # #     for key in torch.load(self.backbone_base()).keys():
        # #         params[key] = self.cp_model.convnet.state_dict()[key].cpu()
        # #     task_vectors.append(TaskVector(None, None, params))
        
        # # sim_matrix = torch.zeros((len(task_vectors), len(task_vectors)))
        # # for i in range(len(task_vectors)):
        # #     for j in range(i + 1):
        # #         tv_i = torch.cat([params.view(-1) for params in task_vectors[i].vector.values()])
        # #         tv_j = torch.cat([params.view(-1) for params in task_vectors[j].vector.values()])
        # #         sim_matrix[i, j] = sim_matrix[j, i] = torch.cosine_similarity(tv_i, tv_j, dim=0)
        # # print(sim_matrix)
        
        # if self._config["model_merge"] == "max":
        #     merged_tv = merge_max_abs(task_vectors)
        # elif self._config["model_merge"] == "ties":
        #     reset_type = 'topk'
        #     reset_thresh = self._config["model_merge_topk"]
        #     resolve = self._config["model_merge_resolve"]
        #     merge = 'dis-mean'
        #     tv_flat_checks = torch.vstack([state_dict_to_vector(tv.vector) for tv in task_vectors])
            
        #     merged_flat_tv = merge_methods(
        #         reset_type,
        #         tv_flat_checks,
        #         reset_thresh=reset_thresh,
        #         resolve_method=resolve,
        #         merge_func=merge,
        #     )
        #     merged_tv = vector_to_state_dict(
        #         merged_flat_tv, task_vectors[0].vector, remove_keys=[]
        #     )
        #     merged_tv = TaskVector(vector=merged_tv)
        # elif self._config["model_merge"] == "avg":
        #     merged_tv = merge_avg(task_vectors)
        
        # backbone_params = merged_tv.apply_to(self.backbone_base(), scaling_coef=self._config["model_merge_coef"])
        # self.cs_model.backbone.load_state_dict(backbone_params, strict=False)
        # self.cp_model.convnet.load_state_dict(backbone_params, strict=False)
        
        base_params = torch.load(self.backbone_base())
        tasks_params = [torch.load(self.backbone_checkpoint(task)) for task in range(self._cur_task + 1)]
        backbone_params = merge(base_params, tasks_params, method=self._config["model_merge"], lamb=self._config["model_merge_coef"], topk=self._config["model_merge_topk"])
        self.cs_model.backbone.load_state_dict(backbone_params, strict=False)
        # self.cp_model.convnet.load_state_dict(backbone_params, strict=False)
        
        if self._config["model_norm"]:
            self.cs_model.cpu()
            self.cp_model.cpu()
            norm_params = torch.load(self.norm_checkpoint(self._cur_task))
            self.cs_model.norm.g.data.copy_(norm_params['g'])
            self.cs_model.norm.b.data.copy_(norm_params['b'])
            if self._config["model_norm_coeff"] == "n":
                self.cs_model.norm.a.data.copy_((self.cs_model.norm.a.data * self._cur_task + norm_params['a']) / (self._cur_task + 1))
            else:
                coeff = self._config["model_norm_coeff"]
                self.cs_model.norm.a.data.copy_(self.cs_model.norm.a.data * coeff + norm_params['a'] * (1 - coeff))
            self.cp_model.norm.load_state_dict(self.cs_model.norm.state_dict())
            self.cs_model.cuda()
            self.cp_model.cuda()
        
        self.cs_model.head.heads[-1].load_state_dict(torch.load(self.head_checkpoint(self._cur_task)), strict=True)

    def prepend_checkpoint(self):
        prepend = f'{self._config["dataset_name"]}_{self._config["seed"]}_{self._config["model_backbone"]}_20'
        if self._config["train_seq"]:
            prepend += "_seq"
        return prepend
    
    def backbone_checkpoint(self, task, epoch=None):
        path = f'/media/ellen/HardDisk/cl/logs/checkpoints/{self.prepend_checkpoint()}_backbone_{task}'
        path += f'_{epoch}.pt' if epoch is not None else '.pt'
        return path

    def backbone_base(self):
        return f'/media/ellen/HardDisk/cl/logs/checkpoints/{self.prepend_checkpoint()}_backbone_base.pt'

    def backbone_first_session(self):
        return f'/media/ellen/HardDisk/cl/logs/checkpoints/backbone_first_session.pt'

    def norm_checkpoint(self, task, epoch=None):
        path = f'/media/ellen/HardDisk/cl/logs/checkpoints/{self.prepend_checkpoint()}_norm_{task}'
        path += f'_{epoch}.pt' if epoch is not None else '.pt'
        return path

    def head_checkpoint(self, task, epoch=None):
        path = f'/media/ellen/HardDisk/cl/logs/checkpoints/{self.prepend_checkpoint()}_head_{task}'
        path += f'_{epoch}.pt' if epoch is not None else '.pt'
        return path
    
def set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
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
    "cub": (20, 20),
    "imagenetr": (20, 20),
    "imageneta": (20, 20)
}

os.makedirs("/media/ellen/HardDisk/cl/logs/checkpoints", exist_ok=True) 

for dataset_name in ["cub"]:
    set_random(1)
    
    start_time = time.time()
    faa, ffm = [], []
    for seed in [1993]:
        config = {
            "seed": seed,
            "reset": True
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
        logger.info("Class Order: ["+",".join([str(x) for x in data_manager._class_order])+"]")
        
        model_config = {
            "model_backbone": "vit_base_patch16_224_in21k_adapter",
            "model_M": 10000,
            "model_merge": "max",
            "model_merge_coef": 1.0,
            "model_merge_topk": 100,
            "model_merge_resolve": "none",
            "model_norm": False,
            "model_norm_coeff": 0.9
        }
        config.update(model_config)
        
        train_config = {
            "train_epochs": 20,
            "train_batch_size": 48,
            "train_fit_epochs": 20,
            "train_fit_lr": 1e-2,
            "train_fit_min_lr": 0.0,
            "train_fit_weight_decay": 5e-4,
            "train_seq": True
        }
        
        config.update(train_config)
        
        for item in config.items():
            logger.info(f"{item[0]}: {item[1]}")
        
        learner = Learner(config)
        learner.learn(data_manager)
    
    logger.info(f"End experiment in {time.time() - start_time}\n")