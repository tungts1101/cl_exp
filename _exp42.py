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
from peft import LoraConfig, get_peft_model, TaskType
import copy
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
from petl import vision_transformer_ssf
from petl import vision_transformer_dyt
from _exp import ContinualLearnerHead
from util import compute_metrics, accuracy, set_random
from merging.task_vectors import TaskVector, merge_max_abs
from merging.ties import merge_methods, state_dict_to_vector, vector_to_state_dict
import gc
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

def setup_logger(log_file=f'logs/{timestamp}_exp37.log'):
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

class DySSF(nn.Module):
    def __init__(self, C, alpha):
        super().__init__()
        self.a = nn.Parameter(torch.ones(1) * alpha)
        self.g = nn.Parameter(torch.ones(C))
        self.b = nn.Parameter(torch.zeros(C))
        
        nn.init.normal_(self.g, mean=1, std=0.02)
        nn.init.normal_(self.b, std=0.02)
    
    def forward(self, x):
        x = torch.relu(self.a * x)
        return self.g * x + self.b

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self.affine = None
        
        if config["finetune"] == "ssf":
            self.backbone = timm.create_model("vit_base_patch16_224_ssf", pretrained=True, num_classes=0)
            self.freeze_backbone()
            self.init_weights()
        elif config["finetune"] == "dyt":
            self.backbone = timm.create_model("vit_base_patch16_224_dyt", pretrained=True, num_classes=0)
            self.freeze_backbone()
        elif config["finetune"] == "lora":
            self.backbone = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
            self.freeze_backbone()
            self.apply_lora()
            if self._config["affine"] == "dyt":
                self.affine = DySSF(768, 1.0)
        elif config["finetune"] == "full":
            self.backbone = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        
        self.head = ContinualLearnerHead(768, 20, with_norm=False)
    
    def apply_lora(self):
        lora_config = LoraConfig(
            r=self._config["lora_rank"],
            lora_alpha=self._config["lora_alpha"],
            target_modules=["qkv"],
            lora_dropout=0.1,
            bias="none"
        )
        self.backbone = get_peft_model(self.backbone, lora_config)
    
    def init_weights(self):
        for name, param in self.backbone.named_parameters():
            if "ssf_scale_" in name:
                nn.init.ones_(param)
            elif "ssf_shift_" in name:
                nn.init.zeros_(param)
    
    def freeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            if "ssf_" not in name and "head" not in name and "dyt" not in name:
                param.requires_grad_(False)
    
    def get_backbone_trainable_params(self):
        params = {}
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                params[name] = param
        return params
    
    def get_affine_params(self):
        return {"a": self.affine.a.cpu().item(), "g": self.affine.g.mean().cpu().item(), "b": self.affine.b.mean().cpu().item()}
    
    def forward(self, x):
        f = self.backbone(x)
        if self.affine != None:
            f = self.affine(f)
        y = self.head(f)['logits']
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

def trim(tensor, top_k=50):
    flattened = tensor.view(-1)
    magnitudes = torch.abs(flattened)
    num_keep = max(1, int(len(flattened) * top_k / 100))
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

def merge(base_params, tasks_params, method="ties", lamb=1.0, trim_top_k=100):
    params = {}
    for name in base_params:
        base_tv = copy.deepcopy(base_params[name])
        task_vectors = [copy.deepcopy(task_params[name]) for task_params in tasks_params]
        tvs = [tv - base_tv for tv in task_vectors]
        if method == "ties":
            tvs = [trim(tv, trim_top_k) for tv in tvs]
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
        self._classes_seen_so_far = 0
        self._class_increments = []
        self._cur_task = -1
        self.accuracy_matrix = []
        
        self.model = Model(config)
        torch.save(self.model.get_backbone_trainable_params(), backbone_base())
        self.model.cuda()
        
        print(self.model)
    
    def learn(self, data_manager):
        self.data_manager = data_manager
        num_tasks = data_manager.nb_tasks

        for task in range(num_tasks):
            self.before_task(task, data_manager)
            
            self.train()
            if self._config["backbone_merge"] != "none":
                self.merge()
            self.eval()
            self.after_task()

    def merge(self):
        base_params = torch.load(backbone_base())
        tasks_params = [torch.load(backbone_checkpoint(task)) for task in range(self._cur_task + 1)]
        backbone_params = merge(base_params, tasks_params, lamb=1.0, method=self._config["backbone_merge"])
        self.model.backbone.load_state_dict(backbone_params, strict=False)
    
    def before_task(self, task, data_manager):
        self._classes_seen_so_far = self._known_classes + data_manager.get_task_size(task)
        self._class_increments.append((self._known_classes, self._classes_seen_so_far - 1))
        self._cur_task = task

    def after_task(self):
        self.model.head.update(self._classes_seen_so_far - self._known_classes)
        self.model.head.cuda()
        self._known_classes = self._classes_seen_so_far
    
    def eval(self):
        test_set = self.data_manager.get_dataset(
            np.arange(0, self._classes_seen_so_far), source="test", mode="test")
        test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4)
        
        self.model.eval()
        
        y_true, y_pred = [], []
        with torch.no_grad():
            for _, (_, _, x, y) in tqdm(enumerate(test_loader)):
                x, y = x.cuda(), y.cuda()
                
                logits = self.model(x)
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

    def entropy_loss(self, logits, threshold=-1):
        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=1)
        if threshold > 0:
            max_probs, _ = torch.max(probs, dim=1)
            mask = (max_probs > threshold).float()
            return (entropy * mask).mean()
        return entropy.mean()
    
    def energy_score(self, logits, temperature=1.0):
        return -temperature * torch.logsumexp(logits / temperature, dim=1)

    def energy_loss(self, logits_id, logits_ood):
        m_id = -10
        m_ood = -1
        temperature = 1.0
        
        e_id = self.energy_score(logits_id, temperature)
        e_ood = self.energy_score(logits_ood, temperature)
        loss_id = torch.pow(F.relu(e_id - m_id), 2).mean()
        loss_ood = torch.pow(F.relu(m_ood - e_ood), 2).mean()
        # loss_id = F.relu(e_id - m_id).mean()
        # loss_ood = F.relu(m_ood - e_ood).mean()
        # return loss_id + loss_ood
        
        return loss_ood
    
    def train(self):
        trainset = data_manager.get_dataset(
            np.arange(self._known_classes, self._classes_seen_so_far), source="train", mode="train")
        train_loader = DataLoader(trainset, batch_size=self._config["batch_size"], shuffle=True, num_workers=4)
        
        ood_trainset = self._ood_data_manager.get_dataset(np.arange(0, 365), source="train", mode="train")
        ood_trainloader = DataLoader(ood_trainset, batch_size=self._config["batch_size"], shuffle=True, num_workers=4)
        
        task_model = Model(self._config).cuda()
        if self._cur_task > 0:
            if self._config["seq_finetune"] == "last":
                backbone_params = torch.load(backbone_checkpoint(self._cur_task - 1))
            elif self._config["seg_finetune"] == "first":
                backbone_params = torch.load(backbone_checkpoint(0))
            else:
                backbone_params = torch.load(backbone_base())
            task_model.backbone.load_state_dict(backbone_params, strict=False)
        
        # if task_model.affine != None:
        #     task_model.affine.load_state_dict(self.model.affine.state_dict(), strict=True)
        #     # task_model.affine.g.data.copy_(self.model.affine.g.data)
        #     # task_model.affine.b.data.copy_(self.model.affine.b.data)
        
        epochs = self._config["epochs"]
        min_lr = self._config["min_lr"]
        base_lr = self._config["lr"]
        backbone_lr = self._config["backbone_lr"]
        head_lr = self._config["head_lr"]
        
        params = [
            {"params": task_model.get_backbone_trainable_params().values(), "lr": base_lr * backbone_lr, "momentum": 0.9, "weight_decay": 5e-4},
            {"params": task_model.head.heads[0].parameters(), "lr": base_lr * head_lr, "momentum": 0.9, "weight_decay": 5e-4}
        ]
        optimizer = optim.SGD(params, lr=base_lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
        
        pbar = tqdm(range(epochs))
        for _, epoch in enumerate(pbar):
            self.model.train()
            total_loss, total, correct = 0, 0, 0
            total_reg_loss = 0

            # for i, (in_set, out_set) in enumerate(zip(train_loader, ood_trainloader)):
            #     x = torch.cat([in_set[2], out_set[2]], dim=0).cuda()
            #     y = in_set[3].cuda()
                
            #     x, y = x.cuda(), y.cuda()
            #     y = torch.where(y - self._known_classes >= 0, y - self._known_classes, -100)
                
            #     logits = task_model(x)
            #     ce_loss = F.cross_entropy(logits[:len(in_set[2])], y)
                
            #     if self._config["reg_method"] == "energy":
            #         reg_loss = self.energy_loss(logits[:len(in_set[2])], logits[len(in_set[2]):])
            #         loss = ce_loss + self._config["reg_energy_lambda"] * reg_loss
            #     elif self._config["reg_method"] == "entropy":
            #         # reg_loss = self.entropy(logits[:len(in_set[2])]).mean() - self.entropy(logits[len(in_set[2]):]).mean()
            #         # reg_loss = self.entropy(logits[:len(in_set[2])]).mean() # 73.97
            #         reg_loss = -self.entropy(logits[len(in_set[2]):]).mean()
            #         loss = ce_loss + self._config["reg_entropy_lambda"] * reg_loss
            
            for i, (_, _, x, y) in enumerate(train_loader):
                x, y = x.cuda(), y.cuda()
                y = torch.where(y - self._known_classes >= 0, y - self._known_classes, -100)
                
                logits = task_model(x)
                scaled_logits = logits / self._config["logits_temperature"]
                
                ce_loss = F.cross_entropy(scaled_logits, y)
                reg_loss = self.entropy_loss(scaled_logits)
                loss = ce_loss - self._config["reg_entropy_lambda"] * reg_loss
                                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_reg_loss += reg_loss.item()
                # correct += (logits[:len(in_set[2])].argmax(dim=1) == y).sum().item()
                correct += (logits.argmax(dim=1) == y).sum().item()
                total += len(y)

                info = f"Task {self._cur_task + 1}, Epoch {epoch + 1}, Loss: {total_loss / total:.4f}, Reg loss: {total_reg_loss / total:.4f}, Acc: {correct * 100 / total:.2f}"
                pbar.set_description(info)
            scheduler.step()
        
        # if self._config["seq_finetune"] == "last":
        #     self.model.backbone.load_state_dict(task_model.get_backbone_trainable_params(), strict=False)
        #     if self.model.affine != None:
        #         # self.model.affine.load_state_dict(task_model.affine.state_dict(), strict=True)
                
        #         self.model.affine.a.data = (0.5 * self.model.affine.a.data + 0.5 * task_model.affine.a.data)
        #         self.model.affine.g.data.copy_(task_model.affine.g.data)
        #         self.model.affine.b.data.copy_(task_model.affine.b.data)
                
        #         # self.model.affine.g.data = (0.5 * self.model.affine.g.data + 0.5 * task_model.affine.g.data)
        #         # self.model.affine.b.data = (0.5 * self.model.affine.b.data + 0.5 * task_model.affine.b.data)
        #         print(self.model.get_affine_params())
        
        task_model.cpu()
        if self._config["backbone_merge"] == "none":
            self.model.backbone.load_state_dict(task_model.get_backbone_trainable_params(), strict=False)
        else:
            torch.save(task_model.get_backbone_trainable_params(), backbone_checkpoint(self._cur_task))
        
        self.model.head.heads[-1].load_state_dict(task_model.head.heads[0].state_dict(), strict=True)
        
        del task_model
        gc.collect()
        torch.cuda.empty_cache()

def set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

for seed in [1993]:
    start_time = time.time()
    set_random(1)
    
    model_config = {
        "finetune": "ssf",
        "lora_rank": 16,
        "lora_alpha": 16,
        "affine": "none",
    }
    
    train_config = {
        "epochs": 5,
        "lr": 1e-2,
        "backbone_lr": 1,
        "head_lr": 1,
        "min_lr": 1e-5,
        "batch_size": 48,
        "logits_temperature": 2.0,
        "reg_method": "entropy",
        "reg_entropy_lambda": 0.1,
        "reg_energy_lambda": 0.1,
        "seq_finetune": "last",
    }
    
    config = {
        "seed": seed,
        "dataset": "cub",
        "backbone_merge": "ties"
    }
    
    config.update(model_config)
    config.update(train_config)
    
    for item in config.items():
        logger.info(f"{item[0]}: {item[1]}")

    data_manager = DataManager(config["dataset"], True, config["seed"], 20, 20, False)
    ood_data_manager = DataManager("places365", True, seed, 20, 20, True)
    learner = Learner(config)
    learner._ood_data_manager = ood_data_manager
    learner.learn(data_manager)
    logger.info(f"End experiment in {time.time() - start_time}\n")