import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
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
from torch.optim.swa_utils import AveragedModel, SWALR


os.makedirs("logs/checkpoints", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

def setup_logger(log_file=f'logs/{timestamp}_exp38.log'):
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

        self.backbone = timm.create_model("vit_base_patch16_224_ssf", pretrained=True, num_classes=0)
        self.freeze_backbone()
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

def backbone_base():
    return "/media/ellen/HardDisk/cl/logs/checkpoints/2_exp_imagenetr_backbone_base.pt"

def head_base():
    return "/media/ellen/HardDisk/cl/logs/checkpoints/2_exp_imagenetr_head_base.pt"
   
def backbone_checkpoint(task):
    return f"/media/ellen/HardDisk/cl/logs/checkpoints/2_exp_imagenetr_backbone_{task}.pt"

def head_checkpoint(task):
    return f"/media/ellen/HardDisk/cl/logs/checkpoints/2_exp_imagenetr_head_{task}.pt"

def merged_head_checkpoint():
    return f"/media/ellen/HardDisk/cl/logs/checkpoints/2_exp_imagenetr_merged_head.pt"

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
        self.cur_task = -1
        self.accuracy_matrix = []
        
        self.model = Model()
        torch.save(self.model.get_backbone_trainable_params(), backbone_base())
        self.model.cuda()
    
    def learn(self, data_manager):
        self.data_manager = data_manager
        num_tasks = data_manager.nb_tasks

        for task in range(num_tasks):
            self.before_task(task, data_manager)
            
            trainset = data_manager.get_dataset(
                np.arange(self._known_classes, self._classes_seen_so_far), source="train", mode="train")
            train_loader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
            self.finetune(train_loader)

            test_set = self.data_manager.get_dataset(
                np.arange(0, self._classes_seen_so_far), source="test", mode="test")
            test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4)
            self.eval(test_loader)
            
            self.after_task()

    def before_task(self, task, data_manager):
        self._classes_seen_so_far = self._known_classes + data_manager.get_task_size(task)
        self._class_increments.append((self._known_classes, self._classes_seen_so_far - 1))
        self.cur_task = task

    def after_task(self):
        self.model.head.update(self._classes_seen_so_far - self._known_classes)
        self.model.head.cuda()
        self._known_classes = self._classes_seen_so_far
    
    def eval(self, test_loader):
        self.model.eval()
        
        y_true, y_pred = [], []
        with torch.no_grad():
            for _, (_, _, x, y) in tqdm(enumerate(test_loader)):
                x, y = x.cuda(), y.cuda()
                
                # if self.cur_task > 0:
                #     logits = []
                #     for i in range(self.cur_task + 1):
                #         self.model.backbone.load_state_dict(torch.load(backbone_checkpoint(i)), strict=False)
                #         self.model.head.heads[0].load_state_dict(torch.load(head_checkpoint(i)), strict=True)
                #         logits.append(self.model(x))
                        
                #     logits = torch.stack(logits, dim=1)
                #     _, logit_indices = torch.max(logits, dim=-1)
                    
                #     temperature = 1
                #     energy = -torch.logsumexp(logits/ temperature, dim=-1)
                    
                #     task_indices_offset = torch.arange(self.cur_task+1, dtype=torch.long, device=device) * 20
                #     logit_indices += task_indices_offset
                    
                #     _, top_indices = torch.topk(energy, 1, dim=-1, largest=False)
                    
                #     predicts = logit_indices[torch.arange(logit_indices.shape[0]), top_indices]
                #     y_pred.append(predicts.cpu().numpy())
                # else:
                #     self.model.backbone.load_state_dict(torch.load(backbone_checkpoint(0)), strict=False)
                #     self.model.head.heads[0].load_state_dict(torch.load(head_checkpoint(0)), strict=True)
            
                #     logits = self.model(x)
                #     y_pred.append(logits.argmax(dim=1).cpu().numpy())
                
                logits = []
                for i in range(self.cur_task + 1):
                    self.model.backbone.load_state_dict(torch.load(backbone_checkpoint(i)), strict=False)
                    self.model.head.heads[0].load_state_dict(torch.load(head_checkpoint(i)), strict=True)
                    logits.append(self.model(x))
                    
                logits = torch.stack(logits, dim=1)
                _, logit_indices = torch.max(logits, dim=-1)
                
                temperature = 1
                energy = -torch.logsumexp(logits/ temperature, dim=-1)
                
                task_indices_offset = torch.arange(self.cur_task+1, dtype=torch.long, device=device) * 20
                logit_indices += task_indices_offset
                
                _, top_indices = torch.topk(energy, 1, dim=-1, largest=False)
                
                predicts = logit_indices[torch.arange(logit_indices.shape[0]), top_indices[:, 0]]
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

    def finetune(self, train_loader, val_ratio=0.2, max_epochs=20, target_acc=95.0):
        def mixup_data(x, a, alpha=0.4):
            """Apply MixUp augmentation"""
            lambda_val = torch.distributions.Beta(alpha, alpha).sample().cuda()
            x_mix = lambda_val * x + (1 - lambda_val) * a
            return x_mix
        
        task_model = Model().cuda()
        
        # Optimizer & Learning Rate Scheduler
        task_optimizer = optim.AdamW(task_model.parameters(), lr=3e-4, weight_decay=1e-4)
        scaler = torch.amp.GradScaler(device)
        swa_model = AveragedModel(task_model)
        swa_scheduler = SWALR(task_optimizer, anneal_strategy="cos", swa_lr=5e-3)

        # Split dataset into train and validation
        train_size = int((1 - val_ratio) * len(train_loader.dataset))
        val_size = len(train_loader.dataset) - train_size
        train_dataset, val_dataset = random_split(train_loader.dataset, [train_size, val_size])

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_loader.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=train_loader.batch_size, shuffle=False)

        # LR scheduler that reduces LR on validation loss plateau
        task_scheduler = optim.lr_scheduler.ReduceLROnPlateau(task_optimizer, mode='min', factor=0.5, patience=3, verbose=True)

        best_acc = 0.0
        for epoch in range(max_epochs):
            task_model.train()
            total_loss, total, correct = 0, 0, 0

            for step, (_, a, x, y) in enumerate(train_loader):
                x, a, y = x.cuda(), a.cuda(), y.cuda()
                y = torch.where(y - self._known_classes >= 0, y - self._known_classes, -100)

                # Apply MixUp
                x_mix = mixup_data(x, a, alpha=0.4)

                with torch.amp.autocast('cuda'):
                    logits = task_model(x_mix)
                    loss = F.cross_entropy(logits, y, label_smoothing=0.1)

                scaler.scale(loss).backward()

                if (step + 1) % 4 == 0:  # Gradient Accumulation
                    scaler.step(task_optimizer)
                    scaler.update()
                    task_optimizer.zero_grad()

                correct += (logits.argmax(dim=1) == y).sum().item()
                total += len(y)
                total_loss += loss.item()

            train_acc = correct * 100 / total
            train_loss = total_loss / total

            # Validation step
            task_model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for _, (_, _, x, y) in enumerate(val_loader):
                    x, y = x.cuda(), y.cuda()
                    y = torch.where(y - self._known_classes >= 0, y - self._known_classes, -100)

                    logits = task_model(x)
                    val_correct += (logits.argmax(dim=1) == y).sum().item()
                    val_total += len(y)

            val_acc = val_correct * 100 / val_total

            info = f"Task {self.cur_task}, Epoch {epoch+1}/{max_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}, Val Acc: {val_acc:.2f}"
            print(info)

            # Reduce LR if validation loss plateaus
            task_scheduler.step(train_loss)

            # Update SWA model
            swa_model.update_parameters(task_model)
            swa_scheduler.step()

            # Early stopping condition
            if val_acc >= target_acc:
                print(f"Validation accuracy {val_acc:.2f}% reached. Stopping early.")
                break
        
        task_model.cpu()
        torch.save(task_model.get_backbone_trainable_params(), backbone_checkpoint(self.cur_task))
        torch.save(task_model.head.heads[0].state_dict(), head_checkpoint(self.cur_task))
        
        # base_params = torch.load(backbone_base())
        # task_params = [torch.load(backbone_checkpoint(i)) for i in range(self.cur_task + 1)]
        # merged_params = ties_merge(base_params, task_params, lamb=1.0)
        # self.model.backbone.load_state_dict(merged_params, strict=False)
        
        # test_set = self.data_manager.get_dataset(np.arange(self._known_classes, self._classes_seen_so_far), source="test", mode="test")
        # test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4)
        # total = len(test_loader.dataset)
        
        # test_acc = 0
        # with torch.no_grad():
        #     for _, (_, _, x, y) in enumerate(test_loader):
        #         x, y = x.cuda(), y.cuda()
        #         logits = self.model(x)
        #         test_acc += (logits.argmax(dim=1) == y).sum().item()
        # print(f"Before test acc: {test_acc * 100 / total:.2f}")
        
        # temperature = 1.0
        # model_optimizer = optim.SGD(self.model.head.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)
        # self.model.train()
        # for _ in range(2):
        #     for _, (_, _, x, y) in enumerate(train_loader):
        #         x, y = x.cuda(), y.cuda()
                
        #         logits = self.model(x)
        #         probs = F.softmax(logits / temperature, dim=1)
        #         neg_entropy = (probs * torch.log(probs)).sum(dim=1).mean()
                
        #         model_optimizer.zero_grad()
        #         neg_entropy.backward()
        #         model_optimizer.step()
            
        # test_acc = 0
        # with torch.no_grad():
        #     for _, (_, _, x, y) in enumerate(test_loader):
        #         x, y = x.cuda(), y.cuda()
        #         logits = self.model(x)
        #         test_acc += (logits.argmax(dim=1) == y).sum().item()
        #         total += len(y)
        # print(f"After test acc: {test_acc * 100 / total:.2f}")
        
        # self.model.head.heads[-1].load_state_dict(task_model.head.heads[0].state_dict(), strict=True)
        
        # if self.cur_task == 0:
        #     self.model.backbone.load_state_dict(task_model.get_backbone_trainable_params(), strict=False)
        # else:
        #     # backbone_params = {}
        #     # model_backbone = self.model.get_backbone_trainable_params()
        #     # task_backbone = task_model.get_backbone_trainable_params()
        #     # for name in model_backbone:
        #     #     backbone_params[name] = torch.max(model_backbone[name], task_backbone[name])
        #     # self.model.backbone.load_state_dict(backbone_params, strict=False)
        #     # task_model.cpu()
        #     # self.model.cpu()
            
        #     base_params = torch.load(backbone_base())
        #     task_params = [torch.load(backbone_checkpoint(i)) for i in range(self.cur_task + 1)]
        #     merged_params = ties_merge(base_params, [task_params], lamb=1.0)
        #     self.model.backbone.load_state_dict(merged_params, strict=False)
            
        #     # tasks_params = [self.model.get_backbone_trainable_params(), task_model.get_backbone_trainable_params()]
        #     # merged_params = ties_merge(base_params, tasks_params, lamb=1.0)
        #     # self.model.backbone.load_state_dict(merged_params, strict=False)
        #     # task_model.cuda()
        #     # self.model.cuda()

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

    for dataset in ["imagenetr"]:
        data_manager = DataManager(dataset, True, seed, 20, 20, False)

        config.update(
            {
                "fine_tune_train_batch_size": 64,
            }
        )
        logger.info(f"{' | '.join('%s: %s' % item for item in config.items())}")

        learner = Learner()
        learner.learn(data_manager)