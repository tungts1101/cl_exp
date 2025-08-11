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
os.makedirs("logs", exist_ok=True)

def setup_logger(log_file=f"logs/_exp60.log"):
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG)

    logger.propagate = False

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_formatter = logging.Formatter('%(asctime)s - %(message)s')
    console_formatter = logging.Formatter("%(asctime)s [%(filename)s] => %(message)s")

    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
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

from sklearn.cluster import KMeans
from collections import OrderedDict

class RandomReplayBuffer:
    def __init__(self, capacity: int, decay=1.0):
        self._capacity = capacity
        self._buffer = []
        self._weights = []
        self._total_seen = 0
        self._decay = decay

    def add(self, x: torch.Tensor, z: torch.Tensor, y: torch.Tensor):
        for i in range(y.size(0)):
            self._total_seen += 1
            entry = (x[i].cpu(), z[i].cpu(), y[i].cpu())

            if len(self._buffer) < self._capacity:
                self._buffer.append(entry)
                self._weights.append(1.0)
            else:
                is_selected = random.random() < (self._capacity / self._total_seen)
                if not is_selected:
                    continue
                
                probs = torch.tensor(self._weights, dtype=torch.float32)
                inv_probs = 1.0 / (probs + 1e-6)  # lower weight = higher chance
                inv_probs = inv_probs / inv_probs.sum()

                idx = torch.multinomial(inv_probs, 1).item()

                # Replace if current candidate weight is greater than chosen entry
                if 1.0 >= self._weights[idx]:
                    self._buffer[idx] = entry
                    self._weights[idx] = 1.0
    
    def update_weights(self):
        if not self._buffer:
            return
        
        self._weights = [w * self._decay for w in self._weights]

    def sample(self, batch_size: int=32):
        if not self._buffer:
            return None
        
        weights_np = np.array(self._weights, dtype=np.float32)
        probs = weights_np / weights_np.sum()
        indices = np.random.choice(len(self._buffer), size=min(batch_size, len(self._buffer)), p=probs)
        
        xs, zs, ys = zip(*[self._buffer[i] for i in indices])
        x_batch = torch.stack(xs)
        z_batch = torch.stack(zs)
        y_batch = torch.tensor(ys, dtype=torch.long)
        return x_batch, z_batch, y_batch
    
    def __iter__(self, batch_size: int = 32):
        random.shuffle(self._buffer)
        for i in range(0, len(self._buffer), batch_size):
            batch = self._buffer[i : i + batch_size]
            x_batch = torch.stack([x for x, _, _ in batch])
            z_batch = torch.stack([z for _, z, _ in batch])
            y_batch = torch.tensor([y for _, _, y in batch], dtype=torch.long)
            yield x_batch, z_batch, y_batch
    
    def split(self, K: int):
        zs = torch.stack([z for _, z, _ in self._buffer])
        kmeans = KMeans(n_clusters=K, random_state=0, n_init='auto')
        labels = kmeans.fit_predict(zs.cpu().numpy())
        
        clustered_sets = [[] for _ in range(K)]
        for idx, cluster_id in enumerate(labels):
            clustered_sets[cluster_id].append(self._buffer[idx])
        
        return clustered_sets
    
    # def split_by_cluster_means(self, cluster_means: torch.Tensor):
    #     """
    #     Args:
    #         cluster_means (torch.Tensor): shape (K, dim), list of cluster centers
    #     """
    #     zs = torch.stack([z for _, z, _ in self._buffer]).cuda()  # (N, dim)
    #     cluster_means = cluster_means.cuda()  # (K, dim)

    #     # Compute distance between each sample and each cluster center → (N, K)
    #     distances = torch.cdist(zs, cluster_means, p=2)  # Euclidean distance

    #     # Assign each sample to the nearest cluster
    #     nearest_clusters = distances.argmin(dim=1)  # (N,)

    #     clustered_sets = [[] for _ in range(len(cluster_means))]
    #     for idx, cluster_id in enumerate(nearest_clusters.cpu().tolist()):
    #         clustered_sets[cluster_id].append(self._buffer[idx])

    #     return clustered_sets
    
    def split_by_cluster_means(self, cluster_means: torch.Tensor, cluster_labels: torch.Tensor):
        """
        Args:
            cluster_means (torch.Tensor): (K, dim), cluster centers (prototypes)
            cluster_labels (torch.Tensor): (K,), corresponding labels for each prototype
        """
        zs = torch.stack([z for _, z, _ in self._buffer]).cuda()  # (N, dim)
        ys = torch.tensor([y.item() for _, _, y in self._buffer]).cuda()  # (N,)

        cluster_means = cluster_means.cuda()
        cluster_labels = cluster_labels.cuda()

        clustered_sets = [[] for _ in range(len(cluster_means))]

        # For each cluster (prototype label), select matching samples
        for cluster_id, cluster_label in enumerate(cluster_labels):
            mask = (ys == cluster_label)  # samples matching the prototype label
            if mask.sum() == 0:
                continue  # skip empty clusters

            matched_indices = mask.nonzero(as_tuple=True)[0]

            # Assign all samples with matching label to this cluster
            for idx in matched_indices:
                clustered_sets[cluster_id].append(self._buffer[idx.item()])

        return clustered_sets

    @property
    def size(self):
        return len(self._buffer)
    
    @property
    def size_by_class(self):
        class_counts = {}
        for _, _, y in self._buffer:
            y_value = y.item()
            class_counts[y_value] = class_counts.get(y_value, 0) + 1
        
        # Sort by key and return OrderedDict
        sorted_class_counts = OrderedDict(sorted(class_counts.items()))
        return sorted_class_counts


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


os.makedirs("/home/lis/checkpoints", exist_ok=True)



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
        self.buffer = RandomReplayBuffer(config["train_buffer_size"], 0.95)

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
            torch.load(self.backbone_checkpoint(task))
            for task in range(self._cur_task + 1)
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
        
        if not os.path.exists(self.backbone_checkpoint(self._cur_task)) or not os.path.exists(self.head_checkpoint(self._cur_task)) or self._config["reset"]:
            self.model.update_head(self._total_classes - self._known_classes, freeze_old=False)
            self.model.cuda()
            print(self.model)
            
            epochs = self._config["train_epochs"]
            
            # parameters = [
            #     {
            #         "params": [p for p in self.model.backbone.parameters() if p.requires_grad],
            #         "lr": 1e-4,
            #         "weight_decay": 5e-4,
            #     },
            #     {
            #         "params": [
            #             p
            #             for i, head in enumerate(self.model.head.heads)
            #             if i != self._cur_task
            #             for p in head.parameters()
            #             if p.requires_grad
            #         ],
            #         "lr": 1e-6,
            #         "weight_decay": 1e-6,
            #     },
            #     {
            #         "params": [
            #             p
            #             for p in self.model.head.heads[self._cur_task].parameters()
            #             if p.requires_grad
            #         ],
            #         "lr": 1e-4,
            #         "weight_decay": 5e-4,
            #     },
            # ]
            # optimizer = optim.AdamW(parameters, betas=(0.9, 0.999), eps=1e-8)
            
            parameters = [
                {
                    "params": [p for p in self.model.backbone.parameters() if p.requires_grad],
                    "lr": 1e-2,
                    "weight_decay": 5e-4,
                },
                {
                    "params": [
                        p
                        for i, head in enumerate(self.model.head.heads)
                        if i != self._cur_task
                        for p in head.parameters()
                        if p.requires_grad
                    ],
                    "lr": 1e-4,
                    "weight_decay": 1e-4,
                },
                {
                    "params": [
                        p
                        for p in self.model.head.heads[self._cur_task].parameters()
                        if p.requires_grad
                    ],
                    "lr": 1e-2,
                    "weight_decay": 5e-4,
                },
            ]
            optimizer = optim.SGD(parameters, momentum=0.9)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=1e-6
            )
            
            # second_parameters = [
            #     {
            #         "params": [
            #             p
            #             for i, head in enumerate(self.model.head.heads)
            #             if i != self._cur_task
            #             for p in head.parameters()
            #             if p.requires_grad
            #         ],
            #         "lr": 1e-4,
            #         "weight_decay": 1e-4,
            #     },
            # ]
            # second_optimizer = optim.SGD(second_parameters, momentum=0.9)
            # second_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            #     second_optimizer, T_max=epochs, eta_min=1e-6
            # )

            self.model.train()
            
            reg_weight = 0.1
            for epoch in range(epochs):
                total_ce_loss, total_reg_loss, total_acc, total = 0, 0, 0, 0

                for _, (_, x_aug, x, y) in enumerate(train_loader):
                    x_aug, x, y = x_aug.cuda(), x.cuda(), y.cuda()
                    
                    # y = torch.where(y - self._known_classes >= 0, y - self._known_classes, -100)

                    features = self.model.get_features(x)
                    grad_scales = [0.1 if i < self._cur_task else 1.0 for i in range(len(self.model.head.heads))]
                    logits = self.model.head(features, grad_scales)['logits'][:, self._known_classes:]
                    # logits = self.model.head(features)["logits"]
                    
                    ce_loss = F.cross_entropy(logits, y)

                    reg_loss = torch.tensor(0.0).cuda()
                    if self._cur_task > 0 and self.buffer.size > 0:
                        x_buf, f_buf, y_buf = self.buffer.sample(batch_size=x.size(0))
                        x_buf, f_buf, y_buf = x_buf.cuda(), f_buf.cuda(), y_buf.cuda()

                        f = self.model.get_features(x_buf)
                        logits_buf = self.model.head(f)['logits']
                        
                        probs_buf = F.softmax(logits_buf, dim=1)
                        entropy_buf = -(probs_buf * probs_buf.log()).sum(dim=1).mean()
                        
                        f_norm = F.normalize(f, dim=1)
                        z_norm = F.normalize(f_buf, dim=1)
                        pos_sim = (f_norm * z_norm).sum(dim=1)

                        reg_loss = entropy_buf + (1 - pos_sim.mean())

                    loss = ce_loss + reg_weight * reg_loss

                    optimizer.zero_grad()
                    # if self._cur_task > 0:
                    #     second_optimizer.zero_grad()
                    loss.backward()
                    
                    # head_params = list(self.model.head.parameters())
                    # head_grad_norm = torch.nn.utils.clip_grad_norm_(head_params, float('inf'))
                    # logger.info(f"Head Grad Norm before clipping: {head_grad_norm:.4f}")
        
                    # torch.nn.utils.clip_grad_norm_(self.model.head.parameters(), 5.0)
                    
                    # if self._cur_task > 0:
                    #     second_optimizer.step()
                    optimizer.step()

                    total_ce_loss += ce_loss.item() * len(y)
                    total_reg_loss += reg_loss.item() * len(y)
                    total_acc += (logits.argmax(dim=1) == y).sum().item()
                    total += len(y)

                scheduler.step()
                # if self._cur_task > 0:
                #     second_scheduler.step()

                logger.info(
                    f"Task {self._cur_task + 1}, Epoch {epoch + 1}/{epochs}, "
                    f"CE Loss: {total_ce_loss / total:.4f}, "
                    f"Reg Loss: {total_reg_loss / total:.4f}, "
                    f"Total Loss: {(total_ce_loss + reg_weight * total_reg_loss) / total:.4f}, "
                    f"Acc: {total_acc / total:.4f}"
                )


            torch.save(self.model.get_backbone_trainable_params(), self.backbone_checkpoint(self._cur_task))
            torch.save(self.model.head.heads[-1].state_dict(), self.head_checkpoint(self._cur_task))
        else:
            self.model.backbone.load_state_dict(
                torch.load(self.backbone_checkpoint(self._cur_task)), strict=False
            )
            self.model.cuda()
        
        # Add samples to replay buffer...
        self.model.eval()
        with torch.no_grad():
            for _, batch in enumerate(train_loader):
                _, _, x, y = batch
                x, y = x.cuda(), y.cuda()
                features = self.model.get_features(x)
                self.buffer.add(x, features, y)
        logger.info(f"Buffer size: {self.buffer.size}")
        logger.info(f"Buffer size by class: {self.buffer.size_by_class}")
        
        # if self._config["model_merge"] == "none":
        #     print(f"Load task model from {self.backbone_checkpoint(self._cur_task)}")
        #     self.model.backbone.load_state_dict(torch.load(self.backbone_checkpoint(self._cur_task)), strict=False)
        # else:
        #     print(f"Perform model merging with method {self._config['model_merge']}")
        #     self.merge()
        
    def prefix(self):
        return f"{self._config['seed']}_{self._config['dataset_name']}_{self._config['dataset_num_task']}_{self._config['model_backbone']}_{self._config['train_method']}_robust_training"

    def backbone_checkpoint(self, task=-1):
        return f"/home/lis/checkpoints/{self.prefix()}_backbone" + (f"_{task}.pt" if task >= 0 else "_base.pt")

    def head_checkpoint(self, task):
        return f"/home/lis/checkpoints/{self.prefix()}_head_{task}.pt"


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
        for seed in [1995]:
            set_random(1)

            config = {
                "seed": seed,
                "reset": True,
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
                
                train_config = {
                    "train_epochs": 1,
                    "train_batch_size": 48,
                    "train_method": "seq",
                    "train_buffer_size": 1000,
                    "train_split_K": 250
                }

                config.update(train_config)

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

                    for item in config.items():
                        logger.info(f"{item[0]}: {item[1]}")

                    learner = Learner(config)
                    learner.learn(data_manager)

        logger.info(f"End experiment in {time.time() - start_time}\n")
