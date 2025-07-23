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


def setup_logger(log_file=f'logs/{timestamp}_exp29.log'):
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
        self.head = ContinualLearnerHead(768, 10)
    
    def reset(self):
        for name, param in self.backbone.named_parameters():
            if "ssf_shift" in name:
                nn.init.zeros_(param)
            elif "ssf_scale" in name:
                nn.init.ones_(param)
        self.head.reset([0])
    
    def freeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            if "head" not in name and "ssf_" not in name:
                param.requires_grad_(False)
                
    def forward(self, x):
        f = self.backbone(x)
        y = self.head(f)['logits']
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
        self.cur_task = -1
        
        self.backbones = []
        self.heads = []

    def learn(self, data_manager):
        self.data_manager = data_manager
        num_tasks = data_manager.nb_tasks - 1

        trainset = data_manager.get_dataset(
            np.arange(0, 200), source="train", mode="train")
        train_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
        
        test_set = self.data_manager.get_dataset(
            np.arange(0, 200), source="test", mode="test")
        test_loader = DataLoader(test_set, batch_size=128, shuffle=True, num_workers=4)
        
        self.finetune(train_loader, test_loader)
        self.after_task()

    def before_task(self, task, data_manager):
        print(self.model)
        self._classes_seen_so_far = self._known_classes + data_manager.get_task_size(task)
        self._class_increments.append((self._known_classes, self._classes_seen_so_far - 1))
        self.cur_task = task

    def after_task(self):
        self._known_classes = self._classes_seen_so_far
        
    def finetune(self, train_loader, test_loader):
        epochs = config["fine_tune_train_epochs"]
        weight_decay = 5e-4
        min_lr = 0.0
        optimizer = optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.9, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)

        pbar = tqdm(range(epochs))
        for _, epoch in enumerate(pbar):
            self.model.train()
            total_loss, total, correct = 0, 0, 0

            for i, (_, a, x, y) in enumerate(train_loader):
                x, y = x.cuda(), y.cuda()
                y = y // 20
                logits = self.model(x)

                loss = F.cross_entropy(logits, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                correct += (logits.argmax(dim=1) == y).sum().item()
                total += len(y)

                info = f"Epoch {epoch}, Samples: {total * 100 / len(train_loader.dataset):.2f}, Loss: {total_loss / total:.4f}, Train Acc: {correct * 100 / total:.2f}"
                pbar.set_description(info)

            scheduler.step()
            
            self.model.eval()
            test_total, test_correct = 0, 0
            
            with torch.no_grad():
                for _, (_, a, x, y) in tqdm(enumerate(test_loader)):
                    a, x, y = a.cuda(), x.cuda(), y.cuda()
                    y = y // 20
                    logits = self.model(x)
                    
                    test_correct += (logits.argmax(dim=1) == y).sum().item()
                    test_total += len(y)
            
            print(f"Test Acc: {test_correct * 100 / test_total:.2f}")
        
        # params = {}
        # for name, param in self.model.backbone.named_parameters():
        #     if "ssf_" in name:
        #         params[name] = copy.deepcopy(param)
        # self.backbones.append(params)
        # self.heads.append(copy.deepcopy(self.model.head.state_dict()))


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

    num_incr_cls = 20
    num_init_cls = 20
    data_manager = DataManager("imagenetr", True, seed, num_incr_cls, num_init_cls, False)

    for epoch in [10]:
        config.update(
            {
                "fine_tune_train_batch_size": 64,
                "fine_tune_train_epochs": epoch,
            }
        )
        logger.info(f"{' | '.join('%s: %s' % item for item in config.items())}")

        learner = Learner()
        learner.learn(data_manager)