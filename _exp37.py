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
from util import compute_metrics, accuracy, set_random
import gc
import time

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
        x = torch.tanh(self.a * x)
        return self.g * x + self.b
    
class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self.affine = None
        self.head = None
        
        if "ssf" in config["model_backbone"]:
            if config["model_backbone"] == "supervised_imagenet_1k_ssf":
                self.backbone = timm.create_model("vit_base_patch16_224_ssf", pretrained=True, num_classes=0)
            elif config["model_backbone"] == "supervised_imagenet_21k_ssf":
                self.backbone = timm.create_model("vit_base_patch16_224_in21k_ssf", pretrained=True, num_classes=0)
            self.freeze_backbone()
            self.init_weights()
        elif "vpt" in config["model_backbone"]:
            if config["model_backbone"] == "supervised_imagenet_1k_vpt":
                basicmodelname="vit_base_patch16_224" 
            elif config["model_backbone"] == "supervised_imagenet_21k_vpt":
                basicmodelname="vit_base_patch16_224_in21k"
            
            VPT_type=config["model_prompt_type"]
            Prompt_Token_num=config["model_prompt_token_num"]

            self.backbone = build_promptmodel(modelname=basicmodelname,  Prompt_Token_num=Prompt_Token_num, VPT_type=VPT_type)
            prompt_state_dict = self.backbone.obtain_prompt()
            self.backbone.load_prompt(prompt_state_dict)
            self.backbone.num_features = 768
        elif "lora" in config["model_backbone"]:
            if config["model_backbone"] == "supervised_imagenet_1k_lora":
                self.backbone = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
            elif config["model_backbone"] == "supervised_imagenet_21k_lora":
                self.backbone = timm.create_model("vit_base_patch16_224_in21k", pretrained=True, num_classes=0)
            self.freeze_backbone()
            self.apply_lora()
            
        if self._config["model_affine"]:
            self.affine = DySSF(self.backbone.num_features, 1.0)
    
    def update_head(self, num_classes):
        if self.head == None:
            self.head = ContinualLearnerHead(
                self.backbone.num_features, 
                num_classes, 
                with_norm=False, 
                with_bias=self._config["model_with_bias"])
        else:
            self.head.update(num_classes)
        self.head.cuda()
    
    def apply_lora(self):
        lora_config = LoraConfig(
            r=self._config["model_lora_rank"],
            lora_alpha=self._config["model_lora_alpha"],
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
            if "ssf_" not in name and "head" not in name:
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
        torch.save(self.model.get_backbone_trainable_params(), backbone_base())
        self.model.cuda()
        print(self.model)
        
        self._faa, self._ffm = 0, 0
    
    def learn(self, data_manager):
        self.data_manager = data_manager
        num_tasks = data_manager.nb_tasks

        for task in range(num_tasks):
            self.before_task(task, data_manager)
            
            self.train()
            if self._config["train_merge"] != "none":
                self.merge()
            self.eval()
            self.after_task()

    def merge(self):
        base_params = torch.load(backbone_base())
        tasks_params = [torch.load(backbone_checkpoint(task)) for task in range(self._cur_task + 1)]
        backbone_params = merge(base_params, tasks_params, lamb=1.0, method=self._config["train_merge"], topk=self._config["train_topk"])
        self.model.backbone.load_state_dict(backbone_params, strict=False)
    
    def before_task(self, task, data_manager):
        self._total_classes = self._known_classes + data_manager.get_task_size(task)
        self._class_increments.append((self._known_classes, self._total_classes - 1))
        self._cur_task = task
        self.model.update_head(self._total_classes - self._known_classes)

    def after_task(self):
        self._known_classes = self._total_classes
    
    def eval(self):
        test_set = self.data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
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
        
        self._faa, self._ffm = faa, ffm

    def entropy_loss(self, logits, threshold=-1):
        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=1)
        if threshold > 0:
            max_probs, _ = torch.max(probs, dim=1)
            mask = (max_probs > threshold).float()
            return (entropy * mask).mean()
        return entropy.mean()
    
    def train(self):
        trainset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="train")
        train_loader = DataLoader(trainset, batch_size=self._config["train_batch_size"], shuffle=True, num_workers=4)
        
        task_model = Model(self._config).cuda()
        task_model.update_head(self._total_classes - self._known_classes)
        
        if self._cur_task > 0:
            if self._config["train_finetune"] == "last":
                backbone_params = torch.load(backbone_checkpoint(self._cur_task - 1))
            elif self._config["train_finetune"] == "first":
                backbone_params = torch.load(backbone_checkpoint(0))
            elif self._config["train_finetune"] == "all":
                backbone_params = self.model.get_backbone_trainable_params()
            else:
                backbone_params = torch.load(backbone_base())
            task_model.backbone.load_state_dict(backbone_params, strict=False)
        
        if task_model.affine != None:
            # task_model.affine.load_state_dict(self.model.affine.state_dict(), strict=True)
            task_model.affine.g.data.copy_(self.model.affine.g.data)
            task_model.affine.b.data.copy_(self.model.affine.b.data)
        
        epochs = self._config["train_epochs"]
        min_lr = self._config["train_min_lr"]
        base_lr = self._config["train_lr"]
        
        optimizer = optim.SGD(task_model.parameters(), lr=base_lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
        
        pbar = tqdm(range(epochs))
        for _, epoch in enumerate(pbar):
            self.model.train()
            total_loss, total, correct = 0, 0, 0

            for _, (_, _, x, y) in enumerate(train_loader):
                x, y = x.cuda(), y.cuda()
                y = torch.where(y - self._known_classes >= 0, y - self._known_classes, -100)
                
                logits = task_model(x)
                scaled_logits = logits / self._config["train_logits_temperature"]
                loss = F.cross_entropy(scaled_logits, y)
                loss -= self._config["train_entropy_weight"] * self.entropy_loss(scaled_logits)
                                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                correct += (logits.argmax(dim=1) == y).sum().item()
                total += len(y)

                info = f"Task {self._cur_task + 1}, Epoch {epoch + 1}, Loss: {total_loss / total:.4f}, Acc: {correct * 100 / total:.2f}"
                pbar.set_description(info)
            scheduler.step()
        
        if self.model.affine != None:
            # self.model.affine.load_state_dict(task_model.affine.state_dict(), strict=True)
            self.model.affine.a.data = (self._cur_task * self.model.affine.a.data + task_model.affine.a.data) / (self._cur_task + 1)
            self.model.affine.g.data.copy_(task_model.affine.g.data)
            self.model.affine.b.data.copy_(task_model.affine.b.data)
            
            # self.model.affine.g.data = (0.5 * self.model.affine.g.data + 0.5 * task_model.affine.g.data)
            # self.model.affine.b.data = (0.5 * self.model.affine.b.data + 0.5 * task_model.affine.b.data)
            print(self.model.get_affine_params())
        
        task_model.cpu()
        torch.save(task_model.get_backbone_trainable_params(), backbone_checkpoint(self._cur_task))
        self.model.head.heads[-1].load_state_dict(task_model.head.heads[0].state_dict(), strict=True)
        
        if self._config["train_merge"] == "none":
            self.model.backbone.load_state_dict(task_model.get_backbone_trainable_params(), strict=False)
        
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

data_table = {
    "cifar224": (10, 10),
    "cars": (16, 20),
    "vtab": (10, 10),
    "omnibenchmark": (30, 30),
    "cub": (20, 20),
    "imagenetr": (20, 20),
    "imageneta": (20, 20)
}

for dataset_name in ["cifar224"]:
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
            "model_backbone": "supervised_imagenet_1k_lora",
            "model_prompt_type": "Deep",
            "model_prompt_token_num": 20,
            "model_lora_rank": 16,
            "model_lora_alpha": 16,
            "model_with_bias": True,
            "model_affine": True,
        }
        config.update(model_config)
        
        train_config = {
            "train_epochs": 5,
            "train_batch_size": 48,
            "train_min_lr": 0.0,
            "train_lr": 1e-2,
            "train_logits_temperature": 1.,
            "train_entropy_weight": 0.1,
            "train_finetune": "last",
            "train_merge": "max",
            "train_topk": 20
        }
        
        config.update(train_config)
        
        for item in config.items():
            logger.info(f"{item[0]}: {item[1]}")
        
        learner = Learner(config)
        learner.learn(data_manager)
            
            # faa.append(learner._faa)
            # ffm.append(learner._ffm)
    
    # logger.info(f"FAA: {faa}, FFM: {ffm}")    
    logger.info(f"End experiment in {time.time() - start_time}\n")