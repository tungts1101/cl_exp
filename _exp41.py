import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ChainDataset
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
from utils.toolkit import count_parameters, target2onehot, tensor2numpy
from petl import vision_transformer_ssf
from _exp import ContinualLearnerHead, RandomProjectionHead, plot_heatmap, Buffer
from util import compute_metrics, accuracy, set_random
import math
from merging.task_vectors import TaskVector, merge_max_abs
from merging.ties import merge_methods, state_dict_to_vector, vector_to_state_dict
from test_adaptation import configure_model, check_model, collect_params, Tent
import gc
import time
from collections import defaultdict


os.makedirs("logs/checkpoints", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

def setup_logger(log_file=f'logs/{timestamp}_exp34.log'):
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
        self.init_weights()
        
        # self.backbone = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        # self.freeze_backbone()
        # self.apply_lora()
        
        self.head = ContinualLearnerHead(768, 200, with_norm=False)
    
    def apply_lora(self):
        """Wraps backbone with LoRA adapters."""
        lora_config = LoraConfig(
            r=16,  # LoRA rank
            lora_alpha=32,
            target_modules=["qkv", "fc2"],
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
            if "ssf_" not in name:
                param.requires_grad_(False)
    
    def get_backbone_trainable_params(self):
        return {name: param for name, param in self.backbone.named_parameters() if param.requires_grad}
    
    def forward(self, x):
        f = self.backbone(x)
        y = self.head(f)['logits']
        return y
    
    def __repr__(self):
        trainable_params = count_parameters(self, trainable=True)
        total_params = count_parameters(self)
        return f"Model(trainable_params={trainable_params}, total_params={total_params}, percentage={trainable_params * 100 / total_params:.2f})"


os.makedirs("/media/ellen/HardDisk/cl/logs/checkpoints", exist_ok=True) 
   
def ranpac_checkpoint():
    return "/media/ellen/HardDisk/cl/logs/checkpoints/imagenetr_ranpac.pt"

def backbone_checkpoint(task):
    return f"/media/ellen/HardDisk/cl/logs/checkpoints/2_rb_imagenetr_backbone_{task}.pt"

def head_checkpoint(task):
    return f"/media/ellen/HardDisk/cl/logs/checkpoints/2_rb_imagenetr_head_{task}.pt"

def backbone_ft_checkpoint(task):
    return f"/media/ellen/HardDisk/cl/logs/checkpoints/2_rb_imagenetr_backbone_{task}_ft.pt"

def head_ft_checkpoint(task):
    return f"/media/ellen/HardDisk/cl/logs/checkpoints/2_rb_imagenetr_head_{task}_ft.pt"

def backbone_base():
    return "/media/ellen/HardDisk/cl/logs/checkpoints/2_rb_imagenetr_backbone_base.pt"

def head_base():
    return "/media/ellen/HardDisk/cl/logs/checkpoints/2_rb_imagenetr_head_base.pt"

def head_merge(task):
    return f"/media/ellen/HardDisk/cl/logs/checkpoints/2_rb_imagenetr_head_merge_{task}.pt"

class ReplayBuffer(Dataset):
    def __init__(self, buffer_size=100):
        self._buffer_size = buffer_size
        self._tasks = {}
        self._buffer = []
    
    def update(self):
        self._buffer = []
        for task in self._tasks:
            if len(self._tasks[task]) > self._buffer_size:
                # random.shuffle(self._tasks[task])
                classes = defaultdict(list)
                for x, y in self._tasks[task]:
                    classes[y].append(x)
                num_samples_per_class = self._buffer_size // len(classes)
                
                task_buffer = []
                remaining = []
                for clz in classes:
                    random.shuffle(classes[clz])
                    task_buffer.extend([(x, clz) for x in classes[clz][:num_samples_per_class]])
                    remaining.extend([(x, clz) for x in classes[clz][num_samples_per_class:]])
                
                task_buffer.extend(remaining[:self._buffer_size - len(task_buffer)])
                self._tasks[task] = task_buffer
            self._buffer.extend(self._tasks[task])
    
    def add(self, task, x, y):
        if task not in self._tasks:
            self._tasks[task] = []
        self._tasks[task].append((x, y))
    
    def __len__(self):
        return len(self._buffer)
    
    def __getitem__(self, idx):
        return idx, -1, self._buffer[idx][0], self._buffer[idx][1]
                

class Learner:
    def __init__(self):
        self._known_classes = 0
        self._classes_seen_so_far = 0
        self._class_increments = []
        
        self.model = Model().cuda()
        
        self.accuracy_matrix = []
        self.cur_task = -1
        self._buffer = ReplayBuffer(512)
        
    def learn(self, data_manager):
        self.data_manager = data_manager
        num_tasks = data_manager.nb_tasks

        for task in range(num_tasks):
            start_time = time.time()
            self.before_task(task, data_manager)
            
            trainset = data_manager.get_dataset(
                np.arange(self._known_classes, self._classes_seen_so_far), source="train", mode="train")
            train_loader = DataLoader(trainset, batch_size=48, shuffle=True, num_workers=4)
            self.finetune(train_loader)

            test_set = self.data_manager.get_dataset(
                np.arange(0, self._classes_seen_so_far), source="test", mode="test")
            test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4)
            self.eval(test_loader)
            
            self.after_task()
            print(f"Finish task {task}, time: {time.time() - start_time:.2f}")
            
    def before_task(self, task, data_manager):
        print(self.model)
        self._classes_seen_so_far = self._known_classes + data_manager.get_task_size(task)
        self._class_increments.append((self._known_classes, self._classes_seen_so_far - 1))
        self.cur_task = task

    def after_task(self):
        self._known_classes = self._classes_seen_so_far
    
    def eval(self, test_loader):
        self.model.eval()
        
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for _, (_, _, x, y) in tqdm(enumerate(test_loader)):
                x, y = x.cuda(), y.cuda()
                
                logits = []
                for i in range(self.cur_task + 1):
                    self.model.backbone.load_state_dict(torch.load(backbone_checkpoint(i)), strict=False)
                    self.model.head.heads[0].load_state_dict(torch.load(head_checkpoint(i)), strict=True)
                    logits.append(self.model(x))
                    
                logits = torch.stack(logits, dim=1)
                max_logit_values, max_logit_indices = torch.max(logits, dim=-1)
                
                temperature = 1
                energy = -torch.logsumexp(logits/ temperature, dim=-1)
                
                # task_indices_offset = torch.arange(self.cur_task+1, dtype=torch.long, device=device) * 20
                # max_logit_indices += task_indices_offset

                _, top_indices = torch.topk(energy, 1, dim=-1, largest=False)
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
    
    def finetune(self, train_loader):
        print(f"Task {self.cur_task} | {len(train_loader.dataset)}")
        
        # task_model = Model().cuda()
        # epochs = 3
        # min_lr = 0.0
        # optimizer = optim.SGD(task_model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-2)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
        
        # pbar = tqdm(range(epochs))
        # for _, epoch in enumerate(pbar):
        #     task_model.train()
        #     total_loss, total, correct = 0, 0, 0

        #     for i, (_, _, x, y) in enumerate(train_loader):
        #         x, y = x.cuda(), y.cuda()
                
        #         logits = task_model(x)
        #         loss = F.cross_entropy(logits, y)
                                
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()

        #         total_loss += loss.item()
        #         correct += (logits.argmax(dim=1) == y).sum().item()
        #         total += len(y)

        #         info = f"Epoch {epoch+1}, Loss: {total_loss / total:.4f}, Acc: {correct * 100 / total:.2f}"
        #         pbar.set_description(info)
        #     scheduler.step()
            
        # task_model.cpu()
        # torch.save(task_model.get_backbone_trainable_params(), backbone_checkpoint(self.cur_task))
        # torch.save(task_model.head.heads[0].state_dict(), head_checkpoint(self.cur_task))
        
        # del task_model
        # gc.collect()
        # torch.cuda.empty_cache()
        
        for (_, _, x, y) in train_loader:
            for i in range(len(x)):
                self._buffer.add(self.cur_task, x[i], y[i])
        self._buffer.update()
        # for task in self._buffer._tasks.keys():
        #     print(f"Task {task}: {len(self._buffer._tasks[task])}")
        
        # dataset = ChainDataset([self._buffer, train_loader.dataset])
        
        loader = DataLoader(self._buffer, batch_size=16, shuffle=True, num_workers=4)
        
        for task in range(self.cur_task + 1):
            task_model = Model().cuda()
            # if task < self.cur_task:
            task_model.backbone.load_state_dict(torch.load(backbone_checkpoint(task)), strict=False)
            # task_model.backbone.requires_grad_(False)
            task_model.head.heads[0].load_state_dict(torch.load(head_checkpoint(task)), strict=True)
        
            epochs = 3 if task < self.cur_task else 5
            # epochs = 5
            min_lr = 0.0
            optimizer = optim.AdamW(task_model.parameters(), lr=3e-4, weight_decay=1e-2)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)

            m_in = -25
            m_out = -7
            T = 1.0
            
            pbar = tqdm(range(epochs))
            for _, epoch in enumerate(pbar):
                task_model.train()
                total_loss, total, correct = 0, 0, 0

                for i, (_, _, x, y) in enumerate(loader):
                    x, y = x.cuda(), y.cuda()
                    mask = (y >= task * 20) & (y < (task + 1) * 20)
                    # y = torch.where(y - self._known_classes >= 0, y - self._known_classes, -100)
                    
                    logits = task_model(x)
                    ce_loss = F.cross_entropy(logits, y)
                    # ce_loss = F.cross_entropy(logits[mask], y[mask]) if torch.any(mask) else torch.tensor(0.0).cuda()
                    
                    # E_in = -T * torch.logsumexp(logits[mask] / T, dim=1) if torch.any(mask) else torch.tensor(0.0).cuda()
                    # E_out = -T * torch.logsumexp(logits[~mask] / T, dim=1) if torch.any(~mask) else torch.tensor(0.0).cuda()
                    # # en_loss = torch.pow(F.relu(E_in - m_in), 2).mean() + torch.pow(F.relu(m_out - E_out), 2).mean()
                    # en_loss = torch.pow(F.relu(m_out - E_out), 2).mean()
                    
                    E_in = -T * torch.logsumexp(logits[mask] / T, dim=1) if torch.any(mask) else torch.tensor(0.0).cuda()
                    E_out = -T * torch.logsumexp(logits[~mask] / T, dim=1) if torch.any(~mask) else torch.tensor(0.0).cuda()

                    en_loss = torch.pow(F.relu(E_in - m_in), 2).mean() + torch.pow(F.relu(m_out - E_out), 2).mean()
                    # en_loss = torch.pow(F.relu(m_out - E_out), 2).mean()
                    loss = ce_loss + 0.1 * en_loss
                                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    correct += (logits.argmax(dim=1) == y).sum().item()
                    total += len(y)

                    info = f"Task {task+1}, Epoch {epoch+1}, %: {total / len(loader.dataset):.2f}, Loss: {total_loss / total:.4f}, Acc: {correct * 100 / total:.2f}"
                    pbar.set_description(info)
                
                for i, (_, _, x, y) in enumerate(train_loader):
                    x, y = x.cuda(), y.cuda()
                    mask = (y >= task * 20) & (y < (task + 1) * 20)
                    # y = torch.where(y - self._known_classes >= 0, y - self._known_classes, -100)
                    
                    logits = task_model(x)
                    ce_loss = F.cross_entropy(logits, y)
                    # ce_loss = F.cross_entropy(logits[mask], y[mask]) if torch.any(mask) else torch.tensor(0.0).cuda()
                    
                    # E_in = -T * torch.logsumexp(logits[mask] / T, dim=1) if torch.any(mask) else torch.tensor(0.0).cuda()
                    # E_out = -T * torch.logsumexp(logits[~mask] / T, dim=1) if torch.any(~mask) else torch.tensor(0.0).cuda()
                    # # en_loss = torch.pow(F.relu(E_in - m_in), 2).mean() + torch.pow(F.relu(m_out - E_out), 2).mean()
                    # en_loss = torch.pow(F.relu(m_out - E_out), 2).mean()
                    
                    E_in = -T * torch.logsumexp(logits[mask] / T, dim=1) if torch.any(mask) else torch.tensor(0.0).cuda()
                    E_out = -T * torch.logsumexp(logits[~mask] / T, dim=1) if torch.any(~mask) else torch.tensor(0.0).cuda()

                    en_loss = torch.pow(F.relu(E_in - m_in), 2).mean() + torch.pow(F.relu(m_out - E_out), 2).mean()
                    # en_loss = torch.pow(F.relu(m_out - E_out), 2).mean()
                    loss = ce_loss + 0.1 * en_loss
                                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    correct += (logits.argmax(dim=1) == y).sum().item()
                    total += len(y)

                    info = f"Task {task+1}, Epoch {epoch+1}, %: {total / len(loader.dataset):.2f}, Loss: {total_loss / total:.4f}, Acc: {correct * 100 / total:.2f}"
                    pbar.set_description(info)

                scheduler.step()
            
            task_model.cpu()
            torch.save(task_model.get_backbone_trainable_params(), backbone_checkpoint(task))
            torch.save(task_model.head.heads[0].state_dict(), head_checkpoint(task))
            
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

    data_manager = DataManager("imagenetr", True, seed, 20, 20, False)
    
    # for batch_size, temperature in [(8, 1.0), (8, 10.0), (16, 1.0), (16, 10.0), (32, 1.0), (32, 10.0), (64, 1.0), (64, 10.0)]:
    for batch_size, temperature in [(32, 10.0)]:
        print("\n\n")
        config.update(
            {
                "finetune_bs": batch_size,
                "temperature": temperature
            }
        )
        logger.info(f"{' | '.join('%s: %s' % item for item in config.items())}")

        learner = Learner()
        learner.learn(data_manager)