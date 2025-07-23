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
from util import compute_metrics, accuracy, set_random
import gc
import time
from deps import StreamingLDA


timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

def setup_logger(log_file=f'logs/{timestamp}_exp50.log'):
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
    
    def update_head(self, num_classes):
        if self.head == None:
            self.head = ContinualLearnerHead(self.backbone.num_features, num_classes)
        else:
            self.head.update(num_classes)
        self.head.cuda()
    
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
        base_tv = copy.deepcopy(base_params[name])
        task_vectors = [copy.deepcopy(task_params[name]) for task_params in tasks_params]
        
        tvs = [task_vectors[i] - base_tv for i in range(len(task_vectors))]
        
        if method == "ties":
            tvs = [trim(tv, topk) for tv in tvs]
            merged_tv = merge_task_vectors(tvs)
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
        self._cur_session = -1
        self.accuracy_matrix = []
        
        self.model = Model(config)
        torch.save(self.model.get_backbone_trainable_params(), self.backbone_base())
        self.model.cuda()
        self.model.eval()
        
        self._faa, self._ffm = 0, 0
    
    def learn(self, data_manager, is_plot=False):
        self.data_manager = data_manager
        num_tasks = data_manager.nb_tasks
        self._total_classnum = data_manager.get_total_classnum()
        self._slda_classifier = StreamingLDA(input_shape=self.model.feature_dim, num_classes=self._total_classnum)

        if not is_plot:
            for task in range(num_tasks):
                self.before_task(task, data_manager)
                self.train()
                self.eval()
                self.after_task()
        else:
            self.plot()
        
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
    
    def merge(self, session_coeff=None):
        base_params = torch.load(self.backbone_base())  
        session_params = [torch.load(self.backbone_checkpoint(session)) for session in range(self._cur_session + 1)]
        backbone_params = merge(base_params, session_params, method=self._config["model_merge"], lamb=self._config["model_merge_coef"], topk=self._config["model_merge_topk"])
        self.model.backbone.load_state_dict(backbone_params, strict=False)
    
    def train(self):
        if not os.path.exists(self.backbone_checkpoint(self._cur_session)) or self._config["reset"]:
            trainset = self.data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="train")
            train_loader = DataLoader(trainset, batch_size=self._config["train_batch_size"], shuffle=True, num_workers=4)
            
            session_model = Model(self._config)
            if self._config["train_method"] == "seq": # sequential training
                session_model.backbone.load_state_dict(self.model.backbone.state_dict(), strict=True)
            elif self._config["train_method"] == "last":
                if self._cur_session > 0:
                    session_model.backbone.load_state_dict(torch.load(self.backbone_checkpoint(self._cur_session - 1)), strict=False)
            elif self._config["train_method"] == "first":
                if self._cur_session > 0:
                    session_model.backbone.load_state_dict(torch.load(self.backbone_checkpoint(0)), strict=False)
            session_model.update_head(self._total_classes - self._known_classes)
            session_model.cuda()
            print(session_model)
            
            epochs = self._config["train_epochs"]
            
            optimizer = optim.SGD(session_model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
            
            for epoch in range(epochs):
                session_model.train()
                total_loss, total_acc, total = 0, 0, 0

                for _, (_, _, x, y) in enumerate(train_loader):
                    x, y = x.cuda(), y.cuda()
                    y = torch.where(y - self._known_classes >= 0, y - self._known_classes, -100)
                    
                    features = session_model.get_features(x)
                    logits = session_model.head(features)['logits']
                    loss = F.cross_entropy(logits, y)
                                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    total_acc += (logits.argmax(dim=1) == y).sum().item()
                    total += len(y)
                
                scheduler.step()
                
                if epoch % 2 == 1:
                    info = f"Task {self._cur_session + 1}, Epoch {epoch + 1}/{epochs}, Loss: {total_loss / total:.4f}, Acc: {total_acc / total:.4f}"
                    logger.info(info)

            session_model.cpu()
            torch.save(session_model.get_backbone_trainable_params(), self.backbone_checkpoint(self._cur_session))
        
            del session_model
            gc.collect()
            torch.cuda.empty_cache()
        
        if self._config["model_merge"] == "none":
            print("Load session model...")
            self.model.backbone.load_state_dict(torch.load(self.backbone_checkpoint(self._cur_session)), strict=False)
        else:
            print("Perform model merging...")
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
        return f"{self._config['seed']}_{self._config['dataset_name']}_{self._config['model_backbone']}_{self._config['train_method']}_{self._config['model_merge']}"
    
    def backbone_base(self):
        return f"/media/ellen/HardDisk/cl/logs/checkpoints/{self.prefix()}_backbone_base.pt"
    
    def backbone_checkpoint(self, task):
        return f"/media/ellen/HardDisk/cl/logs/checkpoints/{self.prefix()}_backbone_{task}.pt"
    
    def plot(self):
        num_tasks = data_manager.nb_tasks
        base_params = torch.load(self.backbone_base())
        tasks_params = [torch.load(self.backbone_checkpoint(session)) for session in range(num_tasks)]
        
        base_params = torch.cat([base_params[name_param].view(-1) for name_param in base_params])
        tasks_params = [torch.cat([tasks_params[i][name_param].view(-1) for name_param in tasks_params[i]]) for i in range(num_tasks)]
        tasks_params = [tasks_params[i] - base_params for i in range(num_tasks)]
        
        tasks_matrix = torch.stack(tasks_params, dim=0)  # shape: (N, D)
        norm_tasks_matrix = F.normalize(tasks_matrix, p=2, dim=1).cpu().detach().numpy()

        cosine_sim_matrix = norm_tasks_matrix @ norm_tasks_matrix.T  # (N, D) x (D, N) -> (N, N)
        print(f"Cosine similarity matrix:\n{cosine_sim_matrix}")
        
        if self._config["model_merge"] == "ties":
            tasks_merged = torch.sum(tasks_matrix, dim=0)
        elif self._config["model_merge"] == "max":
            tasks_merged = torch.max(tasks_matrix, dim=0)[0]
        
        tasks_sign = torch.sign(tasks_merged)
        mask = tasks_sign != torch.sign(tasks_matrix)
        mask = torch.sum(mask, dim=1)
        print(f"Mask: {mask}")
    
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

for model_backbone in ["vit_base_patch16_224_lora"]:
    for dataset_name in ["cifar224", "imagenetr", "imageneta", "cub", "omnibenchmark", "vtab"]:
        start_time = time.time()
        faa, ffm = [], []
        for seed in [1994, 1995, 1996]:
            set_random(1)
            
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
            
            for model_merge in ["max"]:
                model_config = {
                    "model_backbone": model_backbone,
                    "model_merge": model_merge,
                    "model_merge_coef": 1.0,
                    "model_merge_topk": 100,
                }
                config.update(model_config)
                
                if model_merge == "none":
                    for train_method in ["seq"]:
                        train_config = {
                            "train_epochs": 10,
                            "train_batch_size": 48,
                            "train_method": train_method
                        }
                        
                        config.update(train_config)
                        
                        for item in config.items():
                            logger.info(f"{item[0]}: {item[1]}")
                        
                        learner = Learner(config)
                        learner.learn(data_manager)
                else:
                    for train_method in ["last"]:
                        train_config = {
                            "train_epochs": 10,
                            "train_batch_size": 48,
                            "train_method": train_method
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