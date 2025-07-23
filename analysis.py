import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import random
import timm
from tqdm import tqdm
import os
from datetime import datetime
from peft import LoraConfig, get_peft_model
import copy
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
from petl import vision_transformer_ssf
from petl import vision_transformer_adapter
from petl.vpt import build_promptmodel
from easydict import EasyDict
from inc_net import CosineLinear
from util import accuracy, set_random
import time
from deps import StreamingLDA
import logging


timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

def setup_logger(log_file=f'logs/{timestamp}_analysis.log'):
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
            bias="none",
            init_lora_weights="gaussian"
        )
        model = get_peft_model(model, lora_config)
        return model

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self.backbone = get_backbone(config)
        self.fc = None
    
    @property
    def feature_dim(self):
        return self.backbone.num_features
    
    def update_fc(self, num_total_classes):
        feature_dim = self.feature_dim
        
        fc = CosineLinear(feature_dim, num_total_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            weight = torch.cat([weight, torch.zeros(num_total_classes - nb_output, self.fc.weight.shape[1]).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc
    
    def get_backbone_trainable_params(self):
        params = {}
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                params[name] = param
        return params
    
    def get_features(self, x):
        f = self.backbone(x)
        return f
    
    def forward(self, x):
        f = self.get_features(x)
        y = self.fc(f)
        
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

def merge(base_params, tasks_params, method="ties", lamb=1.0, topk=100):
    params = {}
    for name in base_params:
        base_tv = base_params[name].clone()
        task_vectors = [task_params[name] for task_params in tasks_params]
        
        tvs = [task_vectors[i] - base_tv for i in range(len(task_vectors))]
        
        if method == "ties":
            tvs = [trim(tv, topk) for tv in tvs]
            merged_tv = merge_task_vectors(tvs)
        elif method == "max":
            merged_tv = torch.max(torch.stack(tvs, dim=0), dim=0)[0]
        elif method == "min":
            merged_tv = torch.min(torch.stack(tvs, dim=0), dim=0)[0]
        elif method == "max_abs":
            stacked = torch.stack(tvs, dim=0)
            abs_stacked = torch.abs(stacked)
            max_idx = torch.argmax(abs_stacked, dim=0)
            merged_tv = torch.gather(stacked, 0, max_idx.unsqueeze(0)).squeeze(0)
            
        params[name] = base_tv + lamb * merged_tv
        
    return params

def compute_metrics(accuracy_matrix):
    faa = np.mean(accuracy_matrix[-1])
    if accuracy_matrix.shape[0] == 1:
        return faa, 0.0, 0.0
    final_acc_per_task = accuracy_matrix[-1]
    max_acc_per_task = np.max(accuracy_matrix, axis=0)
    ffm = np.mean(max_acc_per_task[:-1] - final_acc_per_task[:-1])
    ffd = np.max(max_acc_per_task[:-1] - final_acc_per_task[:-1]) - np.min(max_acc_per_task[:-1] - final_acc_per_task[:-1])
    
    return faa, ffm, ffd

class Learner:
    def __init__(self, config):
        self._config = config
        self._known_classes = 0
        self._total_classes = 0
        self._class_increments = []
        self._cur_session = -1
        self.accuracy_matrix = []
        
        self.model = Model(config)
        torch.save(self.model.get_backbone_trainable_params(), self.backbone_base())
        self.model.cuda()
        self.model.eval()
    
    def learn(self, data_manager):
        self.data_manager = data_manager
        num_tasks = data_manager.nb_tasks
        self._total_classnum = data_manager.get_total_classnum()

        for task in range(num_tasks):
            self.before_task(task, data_manager)
            # self.train()
            # self.eval()
            self.after_task()
        
        self.plot()
        
        # results = []
        # for until_task in range(num_tasks):
        #     print(f"Training until task {until_task}")
        #     for task in range(num_tasks):
        #         self.before_task(task, data_manager)
        #         self.sub_train(until_task)
        #         faa, ffm, ffd = self.eval()
        #         self.after_task()
            
        #     results.append((round(faa, 2), round(ffm, 2), round(ffd, 2)))
            
        #     self._known_classes = 0
        #     self._total_classes = 0
        #     self._class_increments = []
        #     self._cur_session = -1
        #     self.accuracy_matrix = []
        #     self.model.fc = None
            
        # print(results)
        # logger.info(f"Results: {results}")
        
    def before_task(self, task, data_manager):
        self._total_classes = self._known_classes + data_manager.get_task_size(task)
        self._class_increments.append((self._known_classes, self._total_classes - 1))
        self._cur_session = task

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
        print(f"Accuracy: {acc_total:.2f}, Grouped: {grouped}")

        self.accuracy_matrix.append(grouped)

        num_tasks = len(self.accuracy_matrix)
        accuracy_matrix = np.zeros((num_tasks, num_tasks))
        for i in range(num_tasks):
            for j in range(i + 1):
                accuracy_matrix[i, j] = self.accuracy_matrix[i][j]

        faa, ffm, ffd = compute_metrics(accuracy_matrix)
        print(f"FAA: {faa:.2f}, FFM: {ffm:.2f}, FFD: {ffd:.2f}")
        return faa, ffm, ffd
    
    def merge(self):
        if self._config["model_merge_base"] == "init":
            base_params = torch.load(self.backbone_base())
        elif self._config["model_merge_base"] == "first":
            base_params = torch.load(self.backbone_checkpoint(0))
            
        session_params = [torch.load(self.backbone_checkpoint(session)) for session in range(self._cur_session + 1)]
        backbone_params = merge(base_params, session_params, method=self._config["model_merge"], lamb=self._config["model_merge_coef"], topk=self._config["model_merge_topk"])
        self.model.backbone.load_state_dict(backbone_params, strict=False)
    
    def train(self):
        if self._config["model_merge"] == "none":
            self.model.backbone.load_state_dict(torch.load(self.backbone_checkpoint(self._cur_session)), strict=False)
        else:
            self.merge()
        
        self.model.update_fc(self._total_classes)
        
        trainset = self.data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="test")
        train_loader = DataLoader(trainset, batch_size=128, shuffle=False, num_workers=4)
        
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in tqdm(enumerate(train_loader)):
                (_, _, data,label)=batch
                data=data.cuda()
                label=label.cuda()
                embedding = self.model.get_features(data)
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
    
    def sub_train(self, task):
        if self._config["model_merge"] == "none":
            self.model.backbone.load_state_dict(torch.load(self.backbone_checkpoint(self._cur_session)), strict=False)
        else:
            if self._cur_session <= task:
                print(f"Performing merge for the current session {self._cur_session}")
                self.merge()
        
        self.model.update_fc(self._total_classes)
        
        trainset = self.data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="test")
        train_loader = DataLoader(trainset, batch_size=128, shuffle=False, num_workers=4)
        
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in tqdm(enumerate(train_loader)):
                (_, _, data,label)=batch
                data=data.cuda()
                label=label.cuda()
                embedding = self.model.get_features(data)
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
    
    def prefix(self):
        return f"{self._config['seed']}_{self._config['dataset_name']}_{self._config['model_backbone']}_{self._config['train_method']}_{self._config['model_merge']}_{self._config['model_merge_base']}"
    
    def backbone_base(self):
        return f"/media/ellen/HardDisk/cl/logs/checkpoints/{self.prefix()}_backbone_base.pt"
    
    def backbone_checkpoint(self, task):
        return f"/media/ellen/HardDisk/cl/logs/checkpoints/{self.prefix()}_backbone_{task}.pt"
    
    def plot(self):
        num_tasks = data_manager.nb_tasks
        base_params = torch.load(self.backbone_base())
        base_params = torch.cat([base_params[name_param].view(-1) for name_param in base_params])
        
        tasks_params = [torch.load(self.backbone_checkpoint(session)) for session in range(num_tasks)]
        tasks_params = [torch.cat([tasks_params[i][name_param].view(-1) for name_param in tasks_params[i]]) for i in range(num_tasks)]
        tasks_params = [tasks_params[i] - base_params for i in range(num_tasks)]
        
        tasks_matrix = torch.stack(tasks_params, dim=0)  # shape: (N, D)
        norm_tasks_matrix = F.normalize(tasks_matrix, p=2, dim=1).cpu().detach().numpy()

        cosine_sim_matrix = norm_tasks_matrix @ norm_tasks_matrix.T  # (N, D) x (D, N) -> (N, N)
        print(f"Cosine similarity matrix:\n{cosine_sim_matrix}")
        
        import seaborn as sns
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 5))  # Slightly smaller figure to fit side-by-side in LaTeX

        sns.heatmap(
            cosine_sim_matrix,
            annot=True,
            fmt=".2f",  # Optional: format numbers to 2 decimal places
            annot_kws={"size": 10},  # Increase number size in cells
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            square=True,
            cbar_kws={
                "shrink": 0.6,      # Decrease colorbar height
                "aspect": 10,       # Make colorbar narrower
                "pad": 0.02         # Reduce spacing between heatmap and colorbar
            }
        )

        # plt.title("Task Vector Similarity (Separate Training)", fontsize=14)
        plt.xlabel("Task", fontsize=12)
        plt.ylabel("Task", fontsize=12)

        tick_positions = np.arange(cosine_sim_matrix.shape[0])
        tick_labels = [str(i + 1) for i in tick_positions]

        plt.xticks(ticks=tick_positions, labels=tick_labels, fontsize=10)
        plt.yticks(ticks=tick_positions, labels=tick_labels, fontsize=10)

        plt.tight_layout(pad=1.5)
        plt.savefig("research_gram_matrix_sep.pdf", dpi=800)
        plt.close()
        
        # if self._config["model_merge"] == "ties":
        #     tasks_merged = torch.sum(tasks_matrix, dim=0)
        # elif self._config["model_merge"] == "max":
        #     tasks_merged = torch.max(tasks_matrix, dim=0)[0]
        
        # tasks_sign = torch.sign(tasks_merged)
        # mask = tasks_sign != torch.sign(tasks_matrix)
        # mask = torch.sum(mask, dim=1)
        # print(f"Mask: {mask}")
        
        # final_params = torch.cat([final_params[name_param].view(-1) for name_param in final_params])
        # final_params = final_params - base_params

        # # Compute total number of elements in final_params
        # total_params = final_params.numel()

        # similarity_counts = []

        # for task_param in tasks_params:
        #     count = (task_param.sign() == final_params.sign()).sum().item()
        #     similarity_counts.append(int(count))

        # print(f"Total trainable parameters in final model: {total_params}")
        # print("Number of same-sign elements compared to final_params for each task:")
        # for i, count in enumerate(similarity_counts):
        #     print(f"Task {i + 1}: {count}")

        # similarity_ratios = [round(s / total_params, 2) for s in similarity_counts]
        # print(similarity_ratios)
    
def set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

data_table = {
    "cifar224": (10, 10),
    "vtab": (10, 10),
    "omnibenchmark": (30, 30),
    "cub": (20, 20),
    "imagenetr": (20, 20),
    "imageneta": (20, 20)
}

for model_backbone in ["vit_base_patch16_224_lora"]:
    for dataset_name in ["cifar224"]:
        start_time = time.time()
        faa, ffm = [], []
        for seed in [1994]:
            set_random(1)
            
            config = {
                "seed": seed,
                "reset": False
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
            
            for model_merge in (["ties"]): # "max", "max_abs", "ties", "none"
                model_config = {
                    "model_backbone": model_backbone,
                    "model_merge": model_merge,
                    "model_merge_base": "init",
                    "model_merge_coef": 1.0,
                    "model_merge_topk": 100,
                }
                config.update(model_config)
                
                if model_merge == "none":
                    train_config = {
                        "train_epochs": 10,
                        "train_batch_size": 48,
                        "train_method": "sep"
                    }
                    
                    config.update(train_config)
                    
                    for item in config.items():
                        print(f"{item[0]}: {item[1]}")
                    
                    learner = Learner(config)
                    learner.learn(data_manager)
                else:
                    for train_method in ["sep"]: # "first", "sep", "last"
                        train_config = {
                            "train_epochs": 10,
                            "train_batch_size": 48,
                            "train_method": train_method
                        }
                        
                        config.update(train_config)
                        
                        for item in config.items():
                            print(f"{item[0]}: {item[1]}")
                        
                        learner = Learner(config)
                        learner.learn(data_manager)