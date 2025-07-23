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


timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


def setup_logger(log_file=f"logs/{timestamp}_exp57.log"):
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG)

    logger.propagate = False

    # file_handler = logging.FileHandler(log_file)
    # file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # file_formatter = logging.Formatter('%(asctime)s - %(message)s')
    console_formatter = logging.Formatter("%(asctime)s [%(filename)s] => %(message)s")

    # file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)

    # logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = setup_logger()


def get_backbone(config):
    if "lora" in config["model_backbone"]:
        if config["model_backbone"] == "vit_base_patch16_224_lora":
            model = timm.create_model(
                "vit_base_patch16_224", pretrained=True, num_classes=0
            )
        elif config["model_backbone"] == "vit_base_patch16_224_in21k_lora":
            model = timm.create_model(
                "vit_base_patch16_224_in21k", pretrained=True, num_classes=0
            )
        model.requires_grad_(False)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["qkv"],
            lora_dropout=0.0,
            bias="none",
            init_lora_weights="gaussian",
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
            model = vision_transformer_adapter.vit_base_patch16_224_adapter(
                num_classes=0,
                global_pool=False,
                drop_path_rate=0.0,
                tuning_config=tuning_config,
            )
        elif config["model_backbone"] == "vit_base_patch16_224_in21k_adapter":
            model = vision_transformer_adapter.vit_base_patch16_224_in21k_adapter(
                num_classes=0,
                global_pool=False,
                drop_path_rate=0.0,
                tuning_config=tuning_config,
            )
        return model
    elif "ssf" in config["model_backbone"]:
        if config["model_backbone"] == "vit_base_patch16_224_ssf":
            model = timm.create_model(
                "vit_base_patch16_224_ssf", pretrained=True, num_classes=0
            )
        elif config["model_backbone"] == "vit_base_patch16_224_in21k_ssf":
            model = timm.create_model(
                "vit_base_patch16_224_in21k_ssf", pretrained=True, num_classes=0
            )

        for name, param in model.named_parameters():
            if "ssf_scale_" in name:
                nn.init.ones_(param)
            elif "ssf_shift_" in name:
                nn.init.zeros_(param)
            else:
                param.requires_grad_(False)
        return model


class RandomReplayBuffer:
    def __init__(self, capacity: int, decay=1.0):
        self._capacity = capacity
        self._buffer = []
        self._weights = []
        self._total_seen = 0
        self._decay = decay

    def add(self, x: torch.Tensor, z: torch.Tensor, z_neg: torch.Tensor, y: torch.Tensor):
        for i in range(y.size(0)):
            self._total_seen += 1
            entry = (x[i].cpu(), z[i].cpu(), z_neg[i].cpu(), y[i].cpu())

            if len(self._buffer) < self._capacity:
                self._buffer.append(entry)
                self._weights.append(1.0)
            else:
                # Prefer replacing entries with low weight
                probs = torch.tensor(self._weights, dtype=torch.float32)
                inv_probs = 1.0 / (probs + 1e-6)  # lower weight = higher chance
                inv_probs = inv_probs / inv_probs.sum()

                idx = torch.multinomial(inv_probs, 1).item()

                # Replace if current candidate weight is greater than chosen entry
                if 1.0 >= self._weights[idx]:
                    self._buffer[idx] = entry
                    self._weights[idx] = 1.0

    def __iter__(self, batch_size: int = 32):
        random.shuffle(self._buffer)
        for i in range(0, len(self._buffer), batch_size):
            batch = self._buffer[i : i + batch_size]
            x_batch = torch.stack([x for x, _, _, _ in batch])
            z_batch = torch.stack([z for _, z, _, _ in batch])
            z_neg_batch = torch.stack([z_neg for _, _, z_neg, _ in batch])
            y_batch = torch.tensor([y for _, _, _, y in batch], dtype=torch.long)
            yield x_batch, z_batch, z_neg_batch, y_batch

    @property
    def size(self):
        return len(self._buffer)


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
            weight = torch.cat(
                [
                    weight,
                    torch.zeros(
                        num_total_classes - nb_output, self.fc.weight.shape[1]
                    ).cuda(),
                ]
            )
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def update_head(self, num_classes, freeze_old=True):
        if self.head == None:
            self.head = ContinualLearnerHead(self.backbone.num_features, num_classes, with_norm=True)
        else:
            self.head.update(num_classes, freeze_old=freeze_old)
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
    mask = gamma_tvs == gamma
    tau_tvs = torch.stack([tv[0] for tv in trimmed_task_vectors], dim=0)
    mean_tvs = torch.where(mask, tau_tvs, torch.tensor(0.0, dtype=tau_tvs.dtype)).sum(
        dim=0
    ) / mask.sum(dim=0).clamp(min=1)

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
    ffd = np.max(max_acc_per_task[:-1] - final_acc_per_task[:-1]) - np.min(
        max_acc_per_task[:-1] - final_acc_per_task[:-1]
    )

    return faa, ffm, ffd


class Learner:
    def __init__(self, config):
        self._config = config
        self._known_classes = 0
        self._total_classes = 0
        self._class_increments = []
        self._cur_task = -1
        self.accuracy_matrix = []
        self._cls_to_task_idx = {}

        self.model = Model(config)
        self.model.cuda()
        self.model.eval()
        torch.save(self.model.get_backbone_trainable_params(), self.backbone_checkpoint())

        self._faa, self._ffm = 0, 0

    def learn(self, data_manager):
        self.data_manager = data_manager

        # self.buffer = ReplayBuffer(config["train_buffer_size"], data_manager.get_total_classnum())
        self.buffer = RandomReplayBuffer(
            config["train_buffer_size"]
        )

        num_tasks = data_manager.nb_tasks
        self._total_classnum = data_manager.get_total_classnum()
        self._slda_classifier = StreamingLDA(
            input_shape=self.model.feature_dim, num_classes=self._total_classnum
        )

        # for _ in range(num_tasks):
        #     self.model.update_head(10, freeze_old=False)
        
        self.model.cuda()

        for task in range(num_tasks):
            self.before_task(task, data_manager)
            self.train()
            self.eval()
            self.after_task()

    def before_task(self, task, data_manager):
        self._total_classes = self._known_classes + data_manager.get_task_size(task)
        self._class_increments.append((self._known_classes, self._total_classes - 1))
        self._cur_task = task

        for clz in range(self._known_classes, self._total_classes):
            self._cls_to_task_idx[clz] = self._cur_task

    def after_task(self):
        self._known_classes = self._total_classes

    def eval(self):
        test_set = self.data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=4)

        self.model.eval()

        y_true, y_pred = [], []
        with torch.no_grad():
            for _, (_, _, x, y) in tqdm(enumerate(test_loader)):
                x, y = x.cuda(), y.cuda()

                # logits = self.model(x)['logits']

                features = self.model.get_features(x)
                logits = self.model.head(features)["logits"]

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
        logger.info(f"Task {self._cur_task + 1} Evaluation")
        logger.info(f"Final Average Accuracy (FAA): {faa:.2f}")
        logger.info(f"Final Forgetting Measure (FFM): {ffm:.2f}")
        logger.info(f"Final Forgetting Discrepancy (FFD): {ffd:.2f}")

    def merge(self):
        if self._config["model_merge_base"] == "init":
            base_params = torch.load(self.backbone_checkpoint(-1))
        elif self._config["model_merge_base"] == "first":
            base_params = torch.load(self.backbone_checkpoint(0))

        task_params = [
            torch.load(self.backbone_checkpoint(session))
            for session in range(self._cur_task + 1)
        ]
        backbone_params = merge(
            base_params,
            task_params,
            method=self._config["model_merge"],
            lamb=self._config["model_merge_coef"],
            topk=self._config["model_merge_topk"],
        )
        self.model.backbone.load_state_dict(backbone_params, strict=False)

    def train(self):
        trainset = self.data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        train_loader = DataLoader(
            trainset,
            batch_size=self._config["train_batch_size"],
            shuffle=True,
            num_workers=4,
        )

        task_model = Model(self._config)
        task_model.backbone.load_state_dict(
            torch.load(self.backbone_checkpoint(self._cur_task - 1)),
            strict=False,
        )
        task_model.update_head(self._total_classes - self._known_classes)
        task_model.cuda()
        print(task_model)
        
        if not os.path.exists(self.backbone_checkpoint(self._cur_task)) or self._config["reset"]:
            trainset = self.data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="train")
            train_loader = DataLoader(trainset, batch_size=self._config["train_batch_size"], shuffle=True, num_workers=4)
            
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
                    info = f"Task {self._cur_task + 1}, Epoch {epoch + 1}/{epochs}, Loss: {total_loss / total:.4f}, Acc: {total_acc / total:.4f}"
                    logger.info(info)

            torch.save(task_model.get_backbone_trainable_params(), self.backbone_checkpoint(self._cur_task))
            torch.save(task_model.head.heads[-1].state_dict(), self.head_checkpoint(self._cur_task))
        else:
            task_model.backbone.load_state_dict(
                torch.load(self.backbone_checkpoint(self._cur_task)), strict=False
            )
                
        # add to replay buffer
        task_model.eval()
        with torch.no_grad():
            for _, batch in enumerate(train_loader):
                _, _, x, y = batch
                x, y = x.cuda(), y.cuda()
                task_model.backbone.load_state_dict(
                    torch.load(self.backbone_checkpoint(self._cur_task)), strict=False)
                features = task_model.get_features(x)
                
                if self._cur_task > 0:
                    neg_features = []
                    for task_id in range(self._cur_task):
                        backbone_path = self.backbone_checkpoint(task_id)
                        task_model.backbone.load_state_dict(torch.load(backbone_path), strict=False)
                        neg_features.append(task_model.get_features(x))
                    neg_features = torch.stack(neg_features, dim=0).mean(dim=0)
                else:
                    neg_features = torch.zeros_like(features)
                
                self.buffer.add(x, features, neg_features, y)
            
        logger.info(f"Buffer size: {self.buffer.size}")
    
        del task_model
        gc.collect()
        torch.cuda.empty_cache()
        
        if self._config["model_merge"] == "none":
            self.model.backbone.load_state_dict(
                torch.load(self.backbone_checkpoint(self._cur_task)), strict=False
            )
        else:
            print("Perform model merging...")
            self.merge()
        
        self.model.update_head(self._total_classes - self._known_classes, freeze_old=False)
        self.model.cuda()
        self.model.head.heads[self._cur_task].load_state_dict(
            torch.load(self.head_checkpoint(self._cur_task)), strict=True
        )
        print(self.model)
        
        optimizer = optim.SGD(self.model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0.0)

        for epoch in range(5):
            self.model.train()
            total_ce_loss = 0.0
            total_align_loss = 0.0
            total_combined_loss = 0.0
            total_correct = 0
            total_samples = 0

            for _, batch in enumerate(self.buffer.__iter__(batch_size=self._config["train_batch_size"])):
                x, z, z_neg, y = batch
                x, z, z_neg, y = x.cuda(), z.cuda(), z_neg.cuda(), y.cuda()

                f = self.model.get_features(x)
                logits = self.model.head(f)["logits"]

                ce = F.cross_entropy(logits, y)

                f_norm = F.normalize(f, dim=1)
                z_norm = F.normalize(z, dim=1)
                
                cos_sim = (f_norm * z_norm).sum(dim=1)  # shape: [batch]
                align = 1 - cos_sim.mean()
                    
                # align = F.mse_loss(f_norm, z_norm)
                
                # f_norm = F.normalize(f, dim=1)
                # z_norm = F.normalize(z, dim=1)
                # z_neg_norm = F.normalize(z_neg, dim=1)

                # # contrastive alignment: push toward z, pull away from z_neg
                # pos_sim = (f_norm * z_norm).sum(dim=1, keepdim=True)  # [B, 1]
                # neg_sim = (f_norm * z_neg_norm).sum(dim=1, keepdim=True)  # [B, 1]

                # sim_logits = torch.cat([pos_sim, neg_sim], dim=1) / 0.1  # temperature
                # align = F.cross_entropy(sim_logits, torch.zeros(f.size(0), dtype=torch.long).cuda())

                loss = ce + align

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_ce_loss += ce.item() * x.size(0)
                total_align_loss += align.item() * x.size(0)
                total_combined_loss += loss.item() * x.size(0)
                total_correct += (logits.argmax(dim=1) == y).sum().item()
                total_samples += x.size(0)

            scheduler.step()

            log = {
                "ce_loss": total_ce_loss / total_samples,
                "align_loss": total_align_loss / total_samples,
                "total_loss": total_combined_loss / total_samples,
                "acc": total_correct / total_samples,
            }

            logger.info(
                f"Task {self._cur_task + 1}, Epoch {epoch + 1}/5, "
                f"CE: {log['ce_loss']:.4f}, Align: {log['align_loss']:.4f}, "
                f"Loss: {log['total_loss']:.4f}, Acc: {log['acc']:.4f}"
            )
        

    def prefix(self):
        return f"{self._config['seed']}_{self._config['dataset_name']}_{self._config['dataset_num_task']}_\
            {self._config['model_backbone']}_{self._config['train_method']}_{self._config['model_merge']}_{self._config['model_merge_base']}"

    def backbone_checkpoint(self, task=-1):
        return f"/media/ellen/HardDisk/cl/logs/checkpoints/{self.prefix()}_meg_experts_backbone" + (f"_{task}.pt" if task >= 0 else "_base.pt")

    def head_checkpoint(self, task):
        return f"/media/ellen/HardDisk/cl/logs/checkpoints/{self.prefix()}_meg_experts_head_{task}.pt"


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
    # "cifar224": [(10, 10, 10)],
    "imagenetr": [(10, 20, 20)],
}

for model_backbone in ["vit_base_patch16_224_lora"]:
    for dataset_name in ["imagenetr"]:
        start_time = time.time()
        faa, ffm = [], []
        for seed in [1993]:
            set_random(1)

            config = {
                "seed": seed,
                "reset": False,
            }

            for dataset_num_task, dataset_init_cls, dataset_increment in data_table[
                dataset_name
            ]:
                dataset_config = {
                    "dataset_name": dataset_name,
                    "dataset_num_task": dataset_num_task,
                    "dataset_init_cls": dataset_init_cls,
                    "dataset_increment": dataset_increment,
                }
                config.update(dataset_config)

                data_manager = DataManager(
                    config["dataset_name"],
                    True,
                    config["seed"],
                    config["dataset_init_cls"],
                    config["dataset_increment"],
                    False,
                )

                for model_merge in ["ties"]:
                    model_config = {
                        "model_backbone": model_backbone,
                        "model_merge": model_merge,
                        "model_merge_base": "init",
                        "model_merge_coef": 1.0,
                        "model_merge_topk": 100,
                    }
                    config.update(model_config)

                    train_config = {
                        "train_epochs": 10,
                        "train_batch_size": 48,
                        "train_method": "last",
                        "train_buffer_size": 2000,
                    }

                    config.update(train_config)

                    for item in config.items():
                        logger.info(f"{item[0]}: {item[1]}")

                    learner = Learner(config)
                    learner.learn(data_manager)

        logger.info(f"End experiment in {time.time() - start_time}\n")
