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
from _exp import ContinualLearnerHead, RandomProjectionHead, plot_heatmap, Buffer
from util import compute_metrics, accuracy, set_random
import math
from merging.task_vectors import TaskVector, merge_max_abs
from merging.ties import merge_methods, state_dict_to_vector, vector_to_state_dict
import gc
import operator


os.makedirs("logs/checkpoints", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

def setup_logger(log_file=f'logs/{timestamp}_exp35.log'):
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

config = {
}
logger.info(f"{' | '.join('%s: %s' % item for item in config.items())}")

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        # self.freeze_backbone()
        self.head = ContinualLearnerHead(768, 20, with_norm=False)
    
        self.init_weights()
    
    def init_weights(self):
        for name, param in self.backbone.named_parameters():
            if "ssf_scale_" in name:
                nn.init.ones_(param)
            elif "ssf_shift_" in name:
                nn.init.zeros_(param)
    
    def freeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            if "ssf_" not in name:
                param.requires_grad_(False)
    
    def get_backbone_trainable_params(self):
        params = {}
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                params[name] = param
        return params
    
    def forward(self, x):
        f = self.backbone(x)
        y = self.head(f)['logits']
        return y
    
    def __repr__(self):
        trainable_params = count_parameters(self, trainable=True)
        total_params = count_parameters(self)
        return f"Model(trainable_params={trainable_params}, total_params={total_params}, percentage={trainable_params * 100 / total_params:.2f})"
    
os.makedirs("/media/ellen/HardDisk/cl/logs/checkpoints", exist_ok=True) 
   
def backbone_checkpoint(task):
    return f"/media/ellen/HardDisk/cl/logs/checkpoints/seq_ft_imagenetr_backbone_{task}.pt"

def head_checkpoint(task):
    return f"/media/ellen/HardDisk/cl/logs/checkpoints/seq_ft_imagenetr_head_{task}.pt"

def backbone_base():
    return "/media/ellen/HardDisk/cl/logs/checkpoints/seq_ft_imagenetr_backbone_base.pt"

def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr

def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)
    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)
    return _lr_adjuster

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

def merge(trimmed_task_vectors):
    gamma_tvs = torch.stack([tv[1] for tv in trimmed_task_vectors], dim=0)
    gamma = torch.sign(gamma_tvs.sum(dim=0))
    mask = (gamma_tvs == gamma)
    tau_tvs = torch.stack([tv[0] for tv in trimmed_task_vectors], dim=0)
    mean_tvs = torch.where(mask, tau_tvs, torch.tensor(0.0, dtype=tau_tvs.dtype)).sum(dim=0) / mask.sum(dim=0).clamp(min=1)

    return mean_tvs

def ties_merge(base_params, tasks_params, lamb=1.0, trim_top_k=100):
    params = {}
    for name in base_params:
        base_tv = copy.deepcopy(base_params[name])
        task_vectors = [copy.deepcopy(task_params[name]) for task_params in tasks_params]
        tvs = [tv - base_tv for tv in task_vectors]
        tvs = [trim(tv, trim_top_k) for tv in tvs]
        merged_tv = merge(tvs)
        params[name] = base_tv + lamb * merged_tv
        
    return params

class Learner:
    def __init__(self):
        self._known_classes = 0
        self._classes_seen_so_far = 0
        self._class_increments = []
        self.model = Model()
        
        torch.save(self.model.get_backbone_trainable_params(), backbone_base())
        
        self.model.cuda()
        
        self.accuracy_matrix = []
        self.cur_task = -1
    
    def learn(self, data_manager):
        self.data_manager = data_manager
        num_tasks = data_manager.nb_tasks

        for task in range(num_tasks):
            print(f"Starting task {task}")
            
            self.before_task(task, data_manager)
            
            trainset = data_manager.get_dataset(
                np.arange(self._known_classes, self._classes_seen_so_far), source="train", mode="train")
            train_loader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
            # self.seq_finetune(train_loader)
            self.finetune(train_loader)
            
            # self.merge()

            test_set = self.data_manager.get_dataset(
                np.arange(0, self._classes_seen_so_far), source="test", mode="test")
            test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4)
            self.eval(test_loader)
            
            self.after_task()

    def are_models_close(self, params1, params2, atol=1e-8, rtol=1e-5):
        if len(params1) != len(params2):
            return False
        return all(torch.allclose(p1, p2, atol=atol, rtol=rtol) for p1, p2 in zip(params1, params2))
    
    def merge(self):
        self.merged = Model()
        
        task_vectors = [
            TaskVector(backbone_base(), backbone_checkpoint(task)) for task in range(self.cur_task + 1)]
        
        reset_type = 'topk'
        reset_thresh = 70
        resolve = 'mass'
        merge = 'dis-mean'
        tv_flat_checks = torch.vstack([state_dict_to_vector(tv.vector) for tv in task_vectors])
        
        print(f"\nMerging with TIES merging: pruning {reset_type}-{reset_thresh}, resolve sign by {resolve}, merge by {merge}")
        merged_flat_tv = merge_methods(
            reset_type,
            tv_flat_checks,
            reset_thresh=reset_thresh,
            resolve_method=resolve,
            merge_func=merge)
        merged_tv = vector_to_state_dict(merged_flat_tv, task_vectors[0].vector, remove_keys=[])
        merged_tv = TaskVector(vector=merged_tv)
        
        # merged_tv = merge_max_abs(task_vectors)
        
        merged_params = merged_tv.apply_to(backbone_base(), scaling_coef=1.0)
        self.merged.backbone.load_state_dict(merged_params, strict=True)
        
        # base_params = torch.load(backbone_base())
        # tasks_params = [torch.load(backbone_checkpoint(task)) for task in range(self.cur_task + 1)]
        # merged_params = ties_merge(base_params, tasks_params, lamb=1.0)
        
        # self.merged.backbone.load_state_dict(merged_params, strict=False)
        
        for task in range(self.cur_task + 1):
            self.merged.head.heads[-1].load_state_dict(torch.load(head_checkpoint(task)), strict=True)
            if task < self.cur_task:
                self.merged.head.update(20)
                
        self.merged.requires_grad_(False)
        self.merged.cuda()
        self.merged.eval()
        
        # print(f"Value task: {task_vectors[0].vector['blocks.11.mlp.fc2.weight'][0, :5]}")
        # print(f"Value base: {torch.load(backbone_base())['blocks.11.mlp.fc2.weight'][0, :5]}")
        # print(f"Value: {self.merged.backbone.state_dict()['blocks.11.mlp.fc2.weight'][0, :5]}")
        # print(f"Head: {self.merged.head.heads[-1].state_dict()['0.weight'][0, :5]}")
        
        # if self.cur_task == 0:
        #     for params1, params2 in zip(self.model.parameters(), self.merged.parameters()):
        #         assert torch.allclose(params1, params2)
            

    def before_task(self, task, data_manager):
        self._classes_seen_so_far = self._known_classes + data_manager.get_task_size(task)
        self._class_increments.append((self._known_classes, self._classes_seen_so_far - 1))
        self.cur_task = task

    def after_task(self):
        self._known_classes = self._classes_seen_so_far
    
    def eval(self, test_loader):
        y_true, y_pred = [], []
        with torch.no_grad():
            for _, (_, a, x, y) in tqdm(enumerate(test_loader)):
                x, y = x.cuda(), y.cuda()
                # logits = self.merged(x)
                
                logits = []
                for i in range(self.cur_task + 1):
                    self.model.backbone.load_state_dict(torch.load(backbone_checkpoint(i)), strict=True)
                    self.model.head.heads[-1].load_state_dict(torch.load(head_checkpoint(i)), strict=True)
                        
                    logits.append(self.model(x))
                        
                logits = torch.stack(logits, dim=1)
                max_logit_values, max_logit_indices = torch.max(logits, dim=-1)
                
                temperature = 1
                energy = -torch.logsumexp(logits/ temperature, dim=-1)
                
                task_indices_offset = torch.arange(self.cur_task+1, dtype=torch.long, device=device) * 20
                max_logit_indices += task_indices_offset
                
                top_values, top_indices = torch.topk(energy, 1, dim=-1, largest=False)
                predicts = max_logit_indices[torch.arange(max_logit_indices.shape[0]), top_indices[:, 0]]
                
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

    def seq_finetune2(self, train_loader):
        task_model = Model()
        # if self.cur_task > 0:
        #     task_model.backbone.load_state_dict(torch.load(backbone_checkpoint(self.cur_task - 1)), strict=False)
        task_model.cuda()
        
        epochs = config["fine_tune_train_epochs"]
        weight_decay = 5e-4
        min_lr = 0.0
        optimizer = optim.SGD(task_model.parameters(), lr=1e-2, momentum=0.9, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)

        pbar = tqdm(range(epochs))
        for _, epoch in enumerate(pbar):
            task_model.train()
            total_loss, total, correct = 0, 0, 0

            for i, (_, a, x, y) in enumerate(train_loader):
                a, x, y = a.cuda(), x.cuda(), y.cuda()
                y = torch.where(y - self._known_classes >= 0, y - self._known_classes, -100)
                
                logits = task_model(x)
                loss = F.cross_entropy(logits, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                correct += (logits.argmax(dim=1) == y).sum().item()
                total += len(y)

                info = f"Task {self.cur_task}, Epoch {epoch}, %: {total * 100 / len(train_loader.dataset):.2f}, Loss: {total_loss / total:.4f}, Acc: {correct * 100 / total:.2f}"
                pbar.set_description(info)

            scheduler.step()
        
        task_model.cpu()
        torch.save(task_model.get_backbone_trainable_params(), backbone_checkpoint(self.cur_task))
        torch.save(task_model.head.heads[-1].state_dict(), head_checkpoint(self.cur_task))

        del task_model
        gc.collect()
        torch.cuda.empty_cache()
    
    def finetune(self, train_loader):
        task_model = Model()
        task_model.cuda()
        
        epochs = config["fine_tune_train_epochs"]
        weight_decay = 5e-4
        min_lr = 0.0
        params = [
            {"params": task_model.get_backbone_trainable_params().values(), "lr": 1e-5, weight_decay: weight_decay},
            {"params": task_model.head.heads[-1].parameters(), "lr": 1e-2, weight_decay: weight_decay},
        ]
        optimizer = optim.SGD(params, momentum=0.9)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)

        pbar = tqdm(range(epochs))
        for _, epoch in enumerate(pbar):
            task_model.train()
            total_loss, total, correct = 0, 0, 0

            for i, (_, a, x, y) in enumerate(train_loader):
                a, x, y = a.cuda(), x.cuda(), y.cuda()
                y = torch.where(y - self._known_classes >= 0, y - self._known_classes, -100)
                
                logits = task_model(x)
                loss = F.cross_entropy(logits, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                correct += (logits.argmax(dim=1) == y).sum().item()
                total += len(y)

                info = f"Task {self.cur_task}, Epoch {epoch}, %: {total * 100 / len(train_loader.dataset):.2f}, Loss: {total_loss / total:.4f}, Acc: {correct * 100 / total:.2f}"
                pbar.set_description(info)

            scheduler.step()
        
        task_model.cpu()
        torch.save(task_model.get_backbone_trainable_params(), backbone_checkpoint(self.cur_task))
        torch.save(task_model.head.heads[-1].state_dict(), head_checkpoint(self.cur_task))

        del task_model
        gc.collect()
        torch.cuda.empty_cache()
    
    def seq_finetune(self, train_loader):
        task_model = Model()
        # if self.cur_task > 0:
        #     task_model.backbone.load_state_dict(torch.load(backbone_checkpoint(self.cur_task - 1)), strict=True)
        task_model.cuda()
        print(task_model)
        
        epochs = config["fine_tune_train_epochs"]
        lr = 1e-5
        weight_decay = 0.1
        warmup_length = 500
        
        params = [p for p in task_model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        num_batches = len(train_loader)
        scheduler = cosine_lr(optimizer, lr, warmup_length, epochs * num_batches)

        pbar = tqdm(range(epochs))
        for _, epoch in enumerate(pbar):
            task_model.train()
            total_loss, total, correct = 0, 0, 0

            for i, (_, a, x, y) in enumerate(train_loader):
                a, x, y = a.cuda(), x.cuda(), y.cuda()
                y = torch.where(y - self._known_classes >= 0, y - self._known_classes, -100)
                
                logits = task_model(x)
                
                loss = F.cross_entropy(logits, y)
                
                step = i + epoch * num_batches
                scheduler(step)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()

                total_loss += loss.item()
                correct += (logits.argmax(dim=1) == y).sum().item()
                total += len(y)

                info = f"Task {self.cur_task}, Epoch {epoch}, %: {total * 100 / len(train_loader.dataset):.2f}, Loss: {total_loss / total:.4f}, Acc: {correct * 100 / total:.2f}"
                pbar.set_description(info)
        
        task_model.cpu()
        torch.save(task_model.get_backbone_trainable_params(), backbone_checkpoint(self.cur_task))
        torch.save(task_model.head.heads[-1].state_dict(), head_checkpoint(self.cur_task))

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
    logger.info(f"Seed: {seed}")
    set_random(1)

    for dataset in ["cub", "imagenetr"]:
        data_manager = DataManager(dataset, True, seed, 20, 20, False)

        for epoch in [20]:
            config.update(
                {
                    "fine_tune_train_batch_size": 64,
                    "fine_tune_train_epochs": epoch,
                }
            )
            logger.info(f"{' | '.join('%s: %s' % item for item in config.items())}")

            learner = Learner()
            learner.learn(data_manager)