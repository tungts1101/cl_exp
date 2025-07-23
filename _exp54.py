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
from util import accuracy, set_random
import gc
import time
from deps import StreamingLDA


timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

def setup_logger(log_file=f'logs/{timestamp}_exp53.log'):
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)
    
    logger.propagate = False

    # file_handler = logging.FileHandler(log_file)
    # file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # file_formatter = logging.Formatter('%(asctime)s - %(message)s')
    console_formatter = logging.Formatter('%(asctime)s [%(filename)s] => %(message)s')
    
    # file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)
    
    # logger.addHandler(file_handler)
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


class ReplayBuffer:
    def __init__(self, buffer_size: int, total_num_classes: int):
        self._buffer_size = buffer_size
        self._total_num_classes = total_num_classes
        self._buffer_size_per_class = buffer_size // total_num_classes
        self._buffer = {i: [] for i in range(total_num_classes)}

    def add(self, x: torch.Tensor, y: torch.Tensor):
        for i in range(y.size(0)):
            clz = y[i].item()
            self._buffer[clz].append((x[i].cpu(), y[i].cpu()))

    def truncate(self):
        for target, records in self._buffer.items():
            if len(records) > self._buffer_size_per_class:
                self._buffer[target] = random.sample(records, self._buffer_size_per_class)
    
    def __iter__(self, batch_size: int = 32):
        all_records = []
        for target, records in self._buffer.items():
            all_records.extend(records)
        random.shuffle(all_records)

        for i in range(0, len(all_records), batch_size):
            batch = all_records[i:i + batch_size]
            x_batch = torch.stack([x for x, _ in batch])
            y_batch = torch.tensor([y for _, y in batch])
            yield x_batch, y_batch
    
    @property
    def size(self):
        return sum(len(samples) for samples in self._buffer.values())
    

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self.backbone = get_backbone(config)
        self.head = None
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
    
    def update_head(self, num_classes, freeze_old=True):
        if self.head == None:
            self.head = ContinualLearnerHead(self.backbone.num_features, num_classes)
        else:
            self.head.update(num_classes, freeze_old=freeze_old)
        self.head.cuda()
    
    def get_backbone_trainable_params(self):
        # params = {}
        # for name, param in self.backbone.named_parameters():
        #     if param.requires_grad:
        #         params[name] = param
        # return params
        return [p for p in self.backbone.parameters() if p.requires_grad]
    
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
    

os.makedirs("/media/ellen/HardDisk/cl/logs/checkpoints", exist_ok=True) 

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
        # base_tv = copy.deepcopy(base_params[name])
        # task_vectors = [copy.deepcopy(task_params[name]) for task_params in tasks_params]
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


class ModelPopulation:
    def __init__(self, pop_size: int, mutation_method: str, buffer, model):
        self._pop_size = pop_size
        self._buffer = buffer
        self._mutation_method = mutation_method
        self._population = []
        self._model = model
        params = model.get_backbone_trainable_params()
        self._base_params = torch.nn.utils.parameters_to_vector(params)
        
    @property
    def size(self):
        return len(self._population)
    
    def add_individual(self, individual):
        self._population.append(individual)        
    
    def get_base_checkpoint(self):
        return f"/media/ellen/HardDisk/cl/logs/checkpoints/model_base.pt"
    
    def get_model_checkpoint(self, task, index):
        return f"/media/ellen/HardDisk/cl/logs/checkpoints/{task}_model_{index}.pt"
    
    def evaluate(self, individual):
        params = self._model.get_backbone_trainable_params()
        torch.nn.utils.vector_to_parameters(individual, params)
        
        features_list = []
        targets_list = []
        
        self._model.cuda().eval()
        with torch.no_grad():
            for samples, targets in self._buffer:
                samples, targets = samples.cuda(), targets.cuda()
                features = self._model.get_features(samples)
                
                features_list.append(features.cpu())
                targets_list.append(targets.cpu())
        
        features_all = torch.cat(features_list, dim=0)
        targets_all = torch.cat(targets_list, dim=0)
        
        centroids = []
        unique_classes = targets_all.unique()
        for c in unique_classes:
            mask = targets_all == c
            centroid = features_all[mask].mean(dim=0)
            centroids.append(centroid)
        
        centroids = torch.stack(centroids, dim=0)
        dists = torch.cdist(features_all, centroids)
        preds = dists.argmin(dim=1)
        acc = (preds == targets_all).float().mean().item()
        
        return acc
    
    def crossover(self):
        prev_population = list(self._population[:self._pop_size])
        curr_population = list(self._population[self._pop_size:])
        temp_population = []
        
        for i in range(len(curr_population)):
            for j in range(len(prev_population)):
                # print(curr_population[i].shape, curr_population[i].device)
                # print(prev_population[j].shape, prev_population[j].device)
                # print(self._base_params.shape, self._base_params.device)
                
                task_vectors = [curr_population[i] - self._base_params, prev_population[j] - self._base_params]
                merged_vector = torch.max(torch.stack(task_vectors, dim=0), dim=0)[0]
                temp_population.append(self._base_params + merged_vector)
        
        self._population = sorted(
            temp_population, key=lambda x: self.evaluate(x), reverse=True
        )[:self._pop_size]
        if len(self._population) < self._pop_size:
            self._population += curr_population[:self._pop_size - len(self._population)]
    
    def evolve(self):
        print(f"Buffer size: {self._buffer.size}")
        self.crossover()
    
    def get(self):
        return self._population[0]
    
class Learner:
    def __init__(self, config):
        self._config = config
        self._known_classes = 0
        self._total_classes = 0
        self._class_increments = []
        self._cur_task = -1
        self.accuracy_matrix = []
        
        self.model = Model(config)
        
        self._faa, self._ffm = 0, 0
    
    def learn(self, data_manager):
        self.data_manager = data_manager
        
        self.buffer = ReplayBuffer(config["train_buffer_size"], data_manager.get_total_classnum())
        self.pop = ModelPopulation(
            pop_size=self._config["model_pool"],
            mutation_method="none",
            buffer=self.buffer,
            model=self.model
        )
        self.model.cuda().eval()
        
        num_tasks = data_manager.nb_tasks
        self._total_classnum = data_manager.get_total_classnum()

        for task in range(num_tasks):
            self.before_task(task, data_manager)
            self.train()
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
                
                # logits = self._slda_classifier.predict(features)
                
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

        faa, ffm, ffd = compute_metrics(accuracy_matrix)
        logger.info(f"Final Average Accuracy (FAA): {faa:.2f}")
        logger.info(f"Final Forgetting Measure (FFM): {ffm:.2f}")
        logger.info(f"Final Forgetting Discrepancy (FFD): {ffd:.2f}")
    
    def train(self):
        trainset = self.data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="train")
        train_loader = DataLoader(trainset, batch_size=self._config["train_batch_size"], shuffle=True, num_workers=4)
        
        with torch.no_grad():
            for _, (_, _, x, y) in enumerate(train_loader):
                x, y = x.cuda(), y.cuda()
                self.buffer.add(x, y)
        self.buffer.truncate()
        
        task_model = Model(self._config)
        task_model.update_head(self._total_classes - self._known_classes)
        
        if self._cur_task == 0:
            for i in range(self._config["model_pool"]):
                torch.manual_seed(self._config["seed"] + i)
                individual = task_model.get_backbone_trainable_params()
                individual = torch.nn.utils.parameters_to_vector(individual)
                self.pop.add_individual(individual)
            torch.manual_seed(1)
        
        print(f"Population size: {self.pop.size}")
        
        curr_population = []
        for i_individual, prev_individual in enumerate(self.pop._population):
            params = task_model.get_backbone_trainable_params()
            torch.nn.utils.vector_to_parameters(prev_individual, params)
            task_model.head.reset([0])
            task_model.cuda()
            # print(f"{task_model}")
            
            epochs = self._config["train_epochs"]
            optimizer = optim.SGD(task_model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
            
            for epoch in range(epochs):
                task_model.train()
                total_loss, total_acc, total = 0, 0, 0

                for _, (_, _, x, y) in enumerate(train_loader):
                    x, y = x.cuda(), y.cuda()
                    y = torch.where(y - self._known_classes >= 0, y - self._known_classes, -100)
                    
                    features = task_model.get_features(x)
                    logits = task_model.head(features)['logits']
                    loss = F.cross_entropy(logits, y)
                                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    total_acc += (logits.argmax(dim=1) == y).sum().item()
                    total += len(y)
                
                scheduler.step()
                
                if epoch % 2 == 1:
                    info = (
                        f"Task {self._cur_task + 1}, "
                        f"Individual {i_individual + 1}/{self.pop.size}, "
                        f"Epoch {epoch + 1}/{epochs}, "
                        f"Loss: {total_loss / total:.4f}, "
                        f"Acc: {total_acc / total:.4f}"
                    )
                    logger.info(info)
        
            task_model.cpu()
            curr_individual = task_model.get_backbone_trainable_params()
            curr_individual = torch.nn.utils.parameters_to_vector(curr_individual)
            curr_population.append(curr_individual)
        
        for individual in curr_population:
            self.pop.add_individual(individual)
                
        del task_model
        gc.collect()
        torch.cuda.empty_cache()
        
        self.pop.evolve()
        print(f"Population size: {self.pop.size}")
        
        candidate = self.pop.get()
        model_params = self.model.get_backbone_trainable_params()
        torch.nn.utils.vector_to_parameters(candidate, model_params)
        
        self.model.cuda().eval()
        
        self.model.update_fc(self._total_classes)
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
    
def set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# data_table = {
#     "cifar224": [(5, 20, 20), (10, 10, 10), (20, 5, 5)],
#     "imagenetr": [(5, 40, 40), (10, 20, 20), (20, 10, 10)],
#     "imageneta": [(5, 40, 40), (10, 20, 20), (20, 10, 10)],
#     "cub": [(5, 40, 40), (10, 20, 20), (20, 10, 10)],
#     "omnibenchmark": [(5, 60, 60), (10, 30, 30), (20, 15, 15)],
#     "vtab": [(5, 10, 10)]
# }
data_table = {
    "cifar224": [(10, 10, 10)],
}

for model_backbone in ["vit_base_patch16_224_lora"]:
    for dataset_name in ["cifar224"]:
        start_time = time.time()
        faa, ffm = [], []
        for seed in [1993]:
            set_random(1)
            
            config = {
                "seed": seed,
                "reset": False,
            }
            
            for dataset_num_task, dataset_init_cls, dataset_increment in data_table[dataset_name]:
                dataset_config = {
                    "dataset_name": dataset_name,
                    "dataset_num_task": dataset_num_task,
                    "dataset_init_cls": dataset_init_cls,
                    "dataset_increment": dataset_increment
                }
                config.update(dataset_config)
                
                data_manager = DataManager(config["dataset_name"], True, config["seed"], config["dataset_init_cls"], config["dataset_increment"], False)
                
                for model_merge in ["ties"]:
                    model_config = {
                        "model_backbone": model_backbone,
                        "model_merge": model_merge,
                        "model_merge_base": "init",
                        "model_merge_coef": 1.0,
                        "model_merge_topk": 100,
                        "model_pool": 10
                    }
                    config.update(model_config)
                    
                    train_config = {
                        "train_epochs": 10,
                        "train_batch_size": 48,
                        "train_method": "last",
                        "train_buffer_size": 128*100,
                    }
                    
                    config.update(train_config)
                    
                    for item in config.items():
                        logger.info(f"{item[0]}: {item[1]}")
                    
                    learner = Learner(config)
                    learner.learn(data_manager)
                        
        logger.info(f"End experiment in {time.time() - start_time}\n")