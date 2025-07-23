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
from easydict import EasyDict
from util import compute_metrics, set_random, accuracy
from inc_net import CosineLinear
import gc
import time


os.makedirs("logs/checkpoints", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

def setup_logger(log_file=f'logs/{timestamp}_exp46.log'):
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
            bias="none"
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


class SimpleVitNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self.backbone = get_backbone(config)
        self.fc = None

    @property
    def feature_dim(self):
        return 768
    
    def update_fc(self, nb_classes):
        fc = CosineLinear(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.fc.weight.shape[1]).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def get_features(self, x):
        f = self.backbone(x)
        return f
    
    def forward(self, x):
        f = self.get_features(x)
        out = self.fc(f)
        return out
    
    def get_backbone_trainable_params(self):
        params = {}
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                params[name] = param
        return params

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

def merge(base_params, tasks_params, method, lamb=1.0, topk=100):
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
            stacked = torch.stack(tvs, dim=0)
            abs_stacked = stacked.abs()
            indices = torch.argmax(abs_stacked, dim=0)
            merged_tv = torch.gather(stacked, 0, indices.unsqueeze(0)).squeeze(0)

        params[name] = base_tv + lamb * merged_tv
        
    return params


class Learner:
    def __init__(self, config):
        self._config = config
        self._known_classes = 0
        self._total_classes = 0
        self._class_increments = []
        self._cur_task = -1
        
        self._accuracy_matrix = []
        
        self.model = SimpleVitNet(config)
        torch.save(self.model.get_backbone_trainable_params(), self.backbone_base())
        self.model.cuda()
        
        self._faa, self._ffm = 0, 0

    def learn(self, data_manager):
        self.data_manager = data_manager
        num_tasks = data_manager.nb_tasks
        self._total_classnum = data_manager.get_total_classnum()

        for task in range(num_tasks):
            self.before_task(task)
            self.train()
            self.fit()
            self.merge()
            self.eval()
            self.after_task()

    def before_task(self, task):
        self._total_classes = self._known_classes + self.data_manager.get_task_size(task)
        self._class_increments.append((self._known_classes, self._total_classes - 1))
        self._cur_task = task
        
        del self.model.fc
        self.model.fc = None
        self.model.update_fc(self._total_classes)

    def after_task(self):
        self._known_classes = self._total_classes
    
    def eval(self):
        test_set = self.data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=4)
        
        self.model.eval()
        
        y_true, y_pred = [], []
        with torch.no_grad():
            for _, (_, _, x, y) in tqdm(enumerate(test_loader)):
                x, y = x.cuda(), y.cuda()
                logits = self.model(x)['logits']
                preds = logits.argmax(dim=1)
                y_pred.append(preds.cpu().numpy())
                y_true.append(y.cpu().numpy())
        
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        
        acc_total, grouped = accuracy(y_pred.T, y_true, self._class_increments)
        logger.info(f"Total Acc: {acc_total:.2f}, Grouped Acc: {grouped}")
        
        self._accuracy_matrix.append(grouped)

        num_tasks = len(self._accuracy_matrix)
        accuracy_matrix = np.zeros((num_tasks, num_tasks))
        for i in range(num_tasks):
            for j in range(i + 1):
                accuracy_matrix[i, j] = self._accuracy_matrix[i][j]
                
        faa, ffm = compute_metrics(accuracy_matrix)
        logger.info(f"Final Average Accuracy (FAA): {faa:.2f}, Final Forgetting Measure (FFM): {ffm:.2f}")
        
        self._faa, self._ffm = faa, ffm
    
    def train(self):
        if not os.path.exists(self.backbone_checkpoint(self._cur_task)) or self._config["reset"]:
            trainset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="train")
            train_loader = DataLoader(trainset, batch_size=self._config["train_batch_size"], shuffle=True, num_workers=4)
            
            task_model = SimpleVitNet(self._config).cuda()
            if self._config["train_seq"]:
                if self._cur_task > 0:
                    # backbone_params = torch.load(self.backbone_checkpoint(self._cur_task - 1))
                    # task_model.backbone.load_state_dict(backbone_params, strict=False)
                    task_model.backbone.load_state_dict(self.model.backbone.state_dict(), strict=True)
                        
            task_model.update_fc(self._total_classes - self._known_classes)
            
            epochs = self._config["train_epochs"]
            optimizer = optim.SGD(task_model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
            
            pbar = tqdm(range(epochs))
            for _, epoch in enumerate(pbar):
                task_model.train()
                total_loss, total, total_acc = 0, 0, 0

                for _, (_, _, x, y) in enumerate(train_loader):
                    x, y = x.cuda(), y.cuda()
                    y = torch.where(y >= self._known_classes, y - self._known_classes, -100)
                    
                    logits = task_model(x)['logits']
                    loss = F.cross_entropy(logits, y)
                                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    total += len(y)
                    total_acc += (logits.argmax(dim=1) == y).sum().item()
                    
                    info = f"Task {self._cur_task + 1}, Epoch {epoch + 1}, Loss: {total_loss / total:.4f}, Acc: {total_acc * 100 / total:.2f}"
                    pbar.set_description(info)
                scheduler.step()
                
            task_model.cpu()
            torch.save(task_model.get_backbone_trainable_params(), self.backbone_checkpoint(self._cur_task))
            
            del task_model
            gc.collect()
            torch.cuda.empty_cache()

    def fit(self):
        if self._cur_task == 0:
            M = self._config["model_M"]
            self.model.fc.weight.data = torch.zeros(self._total_classes, M).cuda()
            self.model.fc.W_rand = torch.randn(self.model.feature_dim, M).cuda()
            self.W_rand = copy.deepcopy(self.model.fc.W_rand)
            self.Q = torch.zeros(M, self._total_classnum)
            self.G = torch.zeros(M, M)
            self.H = torch.zeros(0, M)
            self.Y = torch.zeros(0, self._total_classnum)
        else:
            self.model.fc.W_rand = self.W_rand
        
        self.model.backbone.load_state_dict(torch.load(self.backbone_checkpoint(self._cur_task)), strict=False)
        self.model.fc.use_RP = True
        
        trainset_CPs = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="test")
        train_loader_CPs = DataLoader(trainset_CPs, batch_size=512, shuffle=False, num_workers=4)
        
        fs = []
        ys = []
        with torch.no_grad():
            for _, (_, _, x, y) in tqdm(enumerate(train_loader_CPs)):
                x, y = x.cuda(), y.cuda()
                f = self.model.get_features(x)
                fs.append(f.cpu())
                ys.append(y.cpu())
                
        fs = torch.cat(fs, dim=0)
        ys = torch.cat(ys, dim=0)
        
        Y = F.one_hot(ys, self._total_classnum).float()
        H = F.relu(fs @ self.model.fc.W_rand.cpu())
        
        dG = H.T @ H
        diag_mask = torch.eye(dG.size(0), dtype=torch.bool, device=dG.device)
        dG = dG * (~diag_mask * 0.9 + diag_mask.float())
        
        self.Q += H.T @ Y
        self.G += dG
        
        self.H = torch.cat([self.H, H], dim=0)
        self.Y = torch.cat([self.Y, Y], dim=0)
        
        ridge = self.optimise_ridge_parameter(self.H, self.Y)
        Wo = torch.linalg.solve(self.G + ridge * torch.eye(self.G.size(dim=0)), self.Q).T
        self.model.fc.weight.data = Wo[0 : self._total_classes, :].cuda()
    
    def optimise_ridge_parameter(self, Features, Y):
        ridges = 10.0 ** np.arange(-8, 9)
        num_val_samples = int(Features.shape[0] * 0.8)
        losses = []
        Q_val = Features[0:num_val_samples, :].T @ Y[0:num_val_samples, :]
        G_val = Features[0:num_val_samples, :].T @ Features[0:num_val_samples, :]
        for ridge in ridges:
            Wo = torch.linalg.solve(
                G_val + ridge * torch.eye(G_val.size(dim=0)), Q_val
            ).T  # better nmerical stability than .inv
            Y_train_pred = Features[num_val_samples::, :] @ Wo.T
            losses.append(F.mse_loss(Y_train_pred, Y[num_val_samples::, :]))
        ridge = ridges[np.argmin(np.array(losses))]
        logger.info("Optimal lambda: " + str(ridge))
        return ridge
    
    def merge(self):
        base_params = torch.load(self.backbone_base())
        tasks_params = [torch.load(self.backbone_checkpoint(task)) for task in range(self._cur_task + 1)]
        backbone_params = merge(base_params, tasks_params, method=self._config["model_merge"], lamb=self._config["model_merge_coef"], topk=self._config["model_merge_topk"])
        self.model.backbone.load_state_dict(backbone_params, strict=False)

    def prepend_checkpoint(self):
        prepend = "seq_" if self._config["train_seq"] else ""
        prepend += f'{self._config["dataset_name"]}_{self._config["seed"]}_{self._config["model_backbone"]}'
        return prepend
    
    def backbone_checkpoint(self, task):
        path = f'/media/ellen/HardDisk/cl/logs/checkpoints/{self.prepend_checkpoint()}_backbone_{task}.pt'
        return path

    def backbone_base(self):
        return f'/media/ellen/HardDisk/cl/logs/checkpoints/{self.prepend_checkpoint()}_backbone_base.pt'
    
def set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
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

os.makedirs("/media/ellen/HardDisk/cl/logs/checkpoints", exist_ok=True) 

for dataset_name in ["imageneta"]:
    set_random(1)
    
    start_time = time.time()
    faa, ffm = [], []
    for seed in [1993]:
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
        logger.info("Class Order: ["+",".join([str(x) for x in data_manager._class_order])+"]")
        
        model_config = {
            "model_backbone": "vit_base_patch16_224_lora",
            "model_M": 10000,
            "model_merge": "ties",
            "model_merge_coef": 1.0,
            "model_merge_topk": 100,
        }
        config.update(model_config)
        
        train_config = {
            "train_epochs": 20,
            "train_batch_size": 48,
            "train_fit_lr": 1e-2,
            "train_fit_min_lr": 0.0,
            "train_fit_weight_decay": 5e-4,
            "train_seq": True
        }
        
        config.update(train_config)
        
        for item in config.items():
            logger.info(f"{item[0]}: {item[1]}")
        
        learner = Learner(config)
        learner.learn(data_manager)
        
        faa.append(learner._faa)
        ffm.append(learner._ffm)
    
    logger.info(f"FAA: {faa}, FFM: {ffm}")
    logger.info(f"End experiment in {time.time() - start_time}\n")