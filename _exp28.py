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



def setup_logger(log_file=f'logs/{timestamp}_exp27.log'):
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

class Program(nn.Module):
    def __init__(self, img_shape=(3, 224, 224), mask_size=50):
        super().__init__()
        if isinstance(mask_size, int):
            mask_size = (mask_size, mask_size)
        
        self.mask_size = mask_size
        self.img_shape = img_shape

        self.W = nn.Parameter(torch.randn(1, *img_shape), requires_grad=True)
        self.M = nn.Parameter(torch.ones(1, *img_shape), requires_grad=False)
        self.M[:, :, 
               (img_shape[1] - mask_size[0])//2:(img_shape[1] + mask_size[0])//2, 
               (img_shape[2] - mask_size[1])//2:(img_shape[2] + mask_size[1])//2] = 0
    
    def reset(self):
        self.W.data = torch.randn(1, *self.img_shape).to(self.W.device)
    
    def forward(self, x):
        B, C, H, W = x.size()
        if C != self.img_shape[0]:
            x = x.repeat(1, self.img_shape[0] // C, 1, 1)
        x_scaled = F.interpolate(x, size=self.mask_size, mode='bicubic', align_corners=False)
        background = torch.zeros(B, *self.img_shape).to(x.device)
        background[:, :, 
                   (self.img_shape[1] - self.mask_size[0])//2:(self.img_shape[1] + self.mask_size[0])//2,
                   (self.img_shape[2] - self.mask_size[1])//2:(self.img_shape[2] + self.mask_size[1])//2] = x_scaled

        x_adv = background + torch.tanh(self.W * self.M)
        return x_adv

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.program = Program(mask_size=180)
        self.backbone = timm.create_model("vit_base_patch16_224_ssf", pretrained=True, num_classes=0)
        self.freeze_backbone()
        self.head = ContinualLearnerHead(768, 20)
        
        self.task_classification = timm.create_model("vit_base_patch16_224", pretrained=True)
    
    def reset(self):
        self.program.reset()
        
        for name, param in self.backbone.named_parameters():
            if "ssf_shift" in name:
                nn.init.zeros_(param)
            elif "ssf_scale" in name:
                nn.init.ones_(param)
    
    def freeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            if "head" not in name and "ssf_" not in name:
                param.requires_grad_(False)
    
    def normalize(self, x_adv):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x_adv.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x_adv.device)
        x_adv = (x_adv - mean) / std
        return x_adv
    
    def label_mapping(self, y_adv):
        # B, C = y_adv.size()
        # y_adv = y_adv.view(B, 10, C // 10)
        # y_adv = y_adv.sum(dim=2)
        y_adv = y_adv[:, :40]
        return y_adv
    
    def task_forward(self, x):
        y = self.task_classification(x)
        y = self.label_mapping(y)
        return y
    
    def forward(self, x):
        x_adv = self.program(x)
        x_adv = self.normalize(x_adv)
        y_adv = self.backbone(x_adv)
        y_adv = self.head(y_adv)['logits']
        # y_adv = self.label_mapping(y_adv)
        return y_adv
    
    def __repr__(self):
        trainable_params = count_parameters(self, trainable=True)
        total_params = count_parameters(self)
        return f"Model(trainable_params={trainable_params}, total_params={total_params}, percentage={trainable_params * 100 / total_params:.2f})"


def entropy(logits):
    return -F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1)

def energy(logits):
    return -torch.logsumexp(logits, dim=1)




class Learner:
    def __init__(self):
        self._known_classes = 0
        self._classes_seen_so_far = 0
        self._class_increments = []
        self.model = Model().to(device)
        self.accuracy_matrix = []
        self.cur_task = -1
        
        self.programs = []
        self.backbones = []

    def learn(self, data_manager):
        self.data_manager = data_manager
        num_tasks = data_manager.nb_tasks - 1

        for task in range(num_tasks):
            self.before_task(task, data_manager)

            # if task == 0:
            trainset = data_manager.get_dataset(
                np.arange(self._known_classes, self._classes_seen_so_far),
                source="train",
                mode="train_adv",
            )
            train_loader = DataLoader(
                trainset,
                batch_size=64,
                shuffle=True,
                num_workers=4,
            )
            self.finetune(train_loader)

            test_set = self.data_manager.get_dataset(
                np.arange(0, self._classes_seen_so_far), source="test", mode="test"
            )
            test_loader = DataLoader(
                test_set, batch_size=8, shuffle=True, num_workers=4
            )
            self.eval(test_loader)
            self.after_task()
            
            if task == 3:
                break

    def before_task(self, task, data_manager):
        print(self.model)
        self._classes_seen_so_far = self._known_classes + data_manager.get_task_size(
            task
        )
        self._class_increments.append(
            (self._known_classes, self._classes_seen_so_far - 1)
        )
        self.cur_task = task

    def after_task(self):
        self.model.reset()
        self.model.head.update(
            self._classes_seen_so_far - self._known_classes, freeze_old=True
        )
        self.model.head.to(device)
        self._known_classes = self._classes_seen_so_far

    def eval(self, test_loader):
        y_pred, y_true = [], []

        with torch.no_grad():
            for _, (_, x, y) in tqdm(enumerate(test_loader)):
                x, y = x.cuda(), y.cuda()
                
                # task_logits = self.model.task_forward(x)
                # task_predicts = torch.topk(task_logits, k=1, dim=1, largest=True, sorted=True)[1]
                # print(task_predicts)
                
                for _, (ix, iy) in enumerate(zip(x, y)):
                    print(f"True label: {iy}")
                    sum_entropies = []
                    min_entropies = []
                    
                    total_sum_entropies = []
                    total_min_entropies = []
                    
                    for i, program in enumerate(self.programs):
                        self.model.program.load_state_dict(program)
                        self.model.backbone.load_state_dict(self.backbones[i], strict=False)
                        
                        
                        logits = self.model(ix.unsqueeze(0))
                        entropy = -F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1)
                        
                        sum_entropies.append(entropy.squeeze(0)[i*20:(i+1)*20].sum().item())
                        min_value, min_index = torch.min(entropy.squeeze(0)[i*20:(i+1)*20], dim=0)
                        min_entropies.append((min_value.item(), min_index.item()))
                        
                        total_sum_entropies.append(entropy.squeeze(0).sum().item())
                        total_min_value, total_min_index = torch.min(entropy.squeeze(0), dim=0)
                        total_min_entropies.append((total_min_value.item(), total_min_index.item()))
                        
                    print(f"Sum entropies: {sum_entropies}, Min entropies: {min_entropies}")
                    print(f"Total Sum entropies: {total_sum_entropies}, Total Min entropies: {total_min_entropies}")
            
                break                       
                        
                
                # logits = self.model(x)
                # predicts = torch.topk(logits, k=1, dim=1, largest=True, sorted=True)[1]

                # y_pred.append(predicts.cpu().numpy())
                # y_true.append(y.cpu().numpy())
            
        # y_pred = np.concatenate(y_pred)
        # y_true = np.concatenate(y_true)
        # acc_total, grouped = accuracy(y_pred.T[0], y_true, self._class_increments)
        # logger.info(f"Total Acc: {acc_total:.2f}, Grouped Acc: {grouped}")

        # self.accuracy_matrix.append(grouped)

        # num_tasks = len(self.accuracy_matrix)
        # accuracy_matrix = np.zeros((num_tasks, num_tasks))
        # for i in range(num_tasks):
        #     for j in range(i + 1):
        #         accuracy_matrix[i, j] = self.accuracy_matrix[i][j]

        # faa, ffm = compute_metrics(accuracy_matrix)
        # logger.info(f"Final Average Accuracy (FAA): {faa:.2f}")
        # logger.info(f"Final Forgetting Measure (FFM): {ffm:.2f}")

    def finetune(self, train_loader):
        epochs = config["fine_tune_train_epochs"]
        
        base_lr = 1e-2
        backbone_lr = 1
        head_lr = 1
        weight_decay = 5e-4
        min_lr = 0.0
        # params = [
        #     {"params": self.model.backbone.parameters(), "lr": base_lr * backbone_lr},
        #     {"params": self.model.head.parameters(), "lr": base_lr * head_lr},
        # ]
        optimizer = optim.SGD(
            self.model.parameters(), lr=1e-2, momentum=0.9, weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=min_lr
        )
        
        task_cls_params = [
            {"params": self.model.task_classification.parameters(), "lr": 1e-4},
        ]
        task_cls_optimizer = optim.AdamW(task_cls_params, weight_decay=5e-4)
        task_cls_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)

        pbar = tqdm(range(epochs))
        for _, epoch in enumerate(pbar):
            self.model.train()
            total_loss, total, correct = 0, 0, 0
            total_cls_loss = 0
            total_task_cls_loss = 0

            for i, (_, x, y) in enumerate(train_loader):
                x, y = x.cuda(), y.cuda()
                y_task = y // 10
                y = torch.where(
                    y - self._known_classes >= 0, y - self._known_classes, -100
                )
                logits = self.model(x)[:, -20:]

                cls_loss = F.cross_entropy(logits, y)
                loss = cls_loss
                
                task_cls_loss = F.cross_entropy(self.model.task_forward(x), y_task)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                task_cls_optimizer.zero_grad()
                task_cls_loss.backward()
                task_cls_optimizer.step()

                total_loss += loss.item()
                correct += (logits.argmax(dim=1) == y).sum().item()
                total += len(y)
                total_cls_loss += cls_loss.item()
                total_task_cls_loss += task_cls_loss.item()

                info = f"Epoch {epoch}, Samples: {total * 100 / len(train_loader.dataset):.2f}, Loss: {total_loss / total:.4f}, Cls loss: {total_cls_loss / total:.4f}, Task cls loss: {total_task_cls_loss / total:.2f}, Acc: {correct * 100 / total:.2f}"
                pbar.set_description(info)

            scheduler.step()
            task_cls_scheduler.step()
            
        logger.info(f"Train accuracy: {correct * 100 / total:.2f}")
        
        self.programs.append(copy.deepcopy(self.model.program.state_dict()))
        params = {}
        for name, param in self.model.backbone.named_parameters():
            if "ssf_" in name:
                params[name] = param.detach().clone()
        self.backbones.append(params)


dataset_name = "cub"

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

    num_classes = 200
    num_init_cls = 20
    data_manager = DataManager(dataset_name, True, seed, 20, 20, False)

    for epoch in [5]:
        config.update(
            {
                "fine_tune_train_batch_size": 64,
                "fine_tune_train_epochs": epoch,
            }
        )
        logger.info(f"{' | '.join('%s: %s' % item for item in config.items())}")

        learner = Learner()
        learner.learn(data_manager)