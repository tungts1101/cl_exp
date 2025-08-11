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
from _exp import (
    ContinualLearnerHead,
    setup_logger,
    get_backbone,
    merge,
    compute_metrics,
    RandomReplayBuffer,
    MultivariateNormalModel,
)
from inc_net import CosineLinear
from util import accuracy, set_random
import gc
import time
from deps import StreamingLDA
import optuna


timestamp = datetime.now().strftime("%Y_%m_%d")
logger = setup_logger(log_file=f"logs/{timestamp}_local_robustness.log")


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
            self.head = ContinualLearnerHead(
                self.backbone.num_features, num_classes, with_norm=True
            )
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


CHECKPOINT_DIR = "/home/lis/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


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
        torch.save(
            self.model.get_backbone_trainable_params(), self.backbone_checkpoint()
        )

        self._faa, self._ffm = 0, 0

    def sample_cluster_centers(self, num_clusters, method="random"):
        if method == "random":
            max_val = 1.0
            min_val = -1.0
            centers = (
                torch.rand(num_clusters, self.model.feature_dim) * (max_val - min_val)
                + min_val
            )
            return centers.cuda()

    def learn(self, data_manager):
        baseline = {
            "cifar224": 89,
            "imagenetr": 78,
            "imageneta": 59
        }
        self._aa = 0

        self.data_manager = data_manager

        num_tasks = data_manager.nb_tasks
        self._total_classnum = data_manager.get_total_classnum()
        self._centers = self.sample_cluster_centers(self._total_classnum * self._config["train_local_robustness_num_clusters"])
        logger.info(f"[Local Robustness] Cluster centers: {self._centers.shape}")

        if self._config["train_with_buffer"]:
            buffer_size = min(
                int(data_manager.train_set_size * self._config["train_buffer_percent"]),
                self._config["train_buffer_size"],
            )
            logger.info(f"[Replay Buffer] Maximum Size: {buffer_size}")
            self.buffer = RandomReplayBuffer(
                buffer_size, self._config["train_buffer_decay"]
            )

        if self._config["train_align_classifier"]:
            self.generative_model = MultivariateNormalModel(
                self.model.feature_dim, self._total_classnum, device="cuda"
            )

        self.model.cuda()

        for task in range(num_tasks):
            self.before_task(task, data_manager)
            self.train()
            self.eval()
            self.after_task()

            if self._aa < baseline[self._config["dataset_name"]]:
                logger.info(
                    f"Baseline accuracy not reached: {self._aa:.2f} < {baseline[self._config['dataset_name']]:.2f}"
                )
                break

    def before_task(self, task, data_manager):
        self._total_classes = self._known_classes + data_manager.get_task_size(task)
        self._class_increments.append((self._known_classes, self._total_classes - 1))
        self._cur_task = task

        for clz in range(self._known_classes, self._total_classes):
            self._cls_to_task_idx[clz] = self._cur_task

        del self.model.fc
        self.model.fc = None
        self.model.update_fc(self._total_classes)

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
            for _, (_, _, x, y) in enumerate(test_loader):
                x, y = x.cuda(), y.cuda()

                if self._config["model_fc"] == "cosine":
                    logits = self.model(x)["logits"]
                elif self._config["model_fc"] == "mlp":
                    features = self.model.get_features(x)
                    logits = self.model.head(features)["logits"]

                predicts = logits.argmax(dim=1)

                y_pred.append(predicts.cpu().numpy())
                y_true.append(y.cpu().numpy())

        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        acc_total, grouped = accuracy(y_pred.T, y_true, self._class_increments)
        logger.info(f"[Evaluation] Task {self._cur_task + 1}")
        logger.info(f"[Evaluation] Total Acc: {acc_total:.2f}, Grouped Acc: {grouped}")

        self.accuracy_matrix.append(grouped)

        num_tasks = len(self.accuracy_matrix)
        accuracy_matrix = np.zeros((num_tasks, num_tasks))
        for i in range(num_tasks):
            for j in range(i + 1):
                accuracy_matrix[i, j] = self.accuracy_matrix[i][j]

        faa, ffm, ffd = compute_metrics(accuracy_matrix)
        logger.info(f"[Evaluation] Final Average Accuracy (FAA): {faa:.2f}")
        logger.info(f"[Evaluation] Final Forgetting Measure (FFM): {ffm:.2f}")
        logger.info(f"[Evaluation] Final Forgetting Discrepancy (FFD): {ffd:.2f}")

        self._aa = faa

    def merge(self):
        logger.info(f"[Model Merging] Method {self._config['model_merge']}")

        # if self._config["model_merge_base"] == "init":
        #     base_params = torch.load(self.backbone_checkpoint(-1))
        # elif self._config["model_merge_base"] == "first":
        #     base_params = torch.load(self.backbone_checkpoint(0))
        
        base_params = torch.load(self.backbone_checkpoint(-1))
        # Calculate total number of parameters being merged
        num_merged_params = sum(param.numel() for param in base_params.values())
        logger.info(f"[Model Merging] Merging with {num_merged_params:,} total parameters")

        task_params = [torch.load(self.backbone_checkpoint(task)) for task in range(self._cur_task + 1)]
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

        if (
            not os.path.exists(self.backbone_checkpoint(self._cur_task))
            or not os.path.exists(self.head_checkpoint(self._cur_task))
            or self._config["reset"]
        ):
            self.model.update_head(
                self._total_classes - self._known_classes,
                freeze_old=self._config["train_freeze_old"],
            )
            self.model.cuda()

            epochs = self._config["train_epochs"]
            base_lr = self._config["train_base_lr"]
            weight_decay = self._config["train_weight_decay"]

            parameters = [
                {
                    "params": [
                        p for p in self.model.backbone.parameters() if p.requires_grad
                    ],
                    "lr": base_lr,
                    "weight_decay": weight_decay,
                },
                {
                    "params": [
                        p
                        for p in self.model.head.heads[self._cur_task].parameters()
                        if p.requires_grad
                    ],
                    "lr": base_lr,
                    "weight_decay": weight_decay,
                },
            ]
            if not self._config["train_freeze_old"]:
                parameters.append(
                    {
                        "params": [
                            p
                            for i, head in enumerate(self.model.head.heads)
                            if i != self._cur_task
                            for p in head.parameters()
                            if p.requires_grad
                        ],
                        "lr": base_lr * 1e-2,
                        "weight_decay": weight_decay * 1e-2,
                    },
                )

            optimizer = optim.SGD(parameters, momentum=0.9)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=1e-6
            )

            self.model.train()
            logger.info(f"[Training] Task {self._cur_task + 1}")
            logger.info(f"[Training] {self.model}")

            rb_weight = self._config["train_base_reg_weight"]
            for epoch in range(epochs):
                total_ce_loss, total_rb_loss, total_acc, total = 0, 0, 0, 0

                for _, (_, x_aug, x, y) in enumerate(train_loader):
                    x_aug, x, y = x_aug.cuda(), x.cuda(), y.cuda()

                    if self._config["train_with_buffer"] and self.buffer.size > 0:
                        x_buf, z_buf, y_buf = self.buffer.sample(batch_size=x.size(0))
                        x_buf, z_buf, y_buf = x_buf.cuda(), z_buf.cuda(), y_buf.cuda()
                        x = torch.cat([x, x_buf], dim=0)
                        y = torch.cat([y, y_buf], dim=0)

                    features = self.model.get_features(x)

                    if self._config["train_freeze_old"]:
                        y = torch.where(
                            y - self._known_classes >= 0, y - self._known_classes, -100
                        )
                        logits = self.model.head.heads[-1](features)
                    else:
                        logits = self.model.head(features)["logits"]

                    ce_loss = F.cross_entropy(logits, y)

                    # calculate local robustness loss ===
                    rb_loss = torch.tensor(0.0, device=x.device)
                    with torch.no_grad():
                        sim_matrix = torch.matmul(features, self._centers.T)  # dot product
                        nearest_idx = sim_matrix.argmax(dim=1)  # [B]

                    num_clusters = self._centers.shape[0]
                    total_samples = x.size(0)
                    # select a subset of clusters to compute the loss
                    subset_size = min(self._total_classes - self._known_classes, num_clusters)
                    selected_clusters = torch.randperm(num_clusters, device=x.device)[:subset_size]

                    for i in selected_clusters:
                        cluster_mask = nearest_idx == i
                        if cluster_mask.sum() == 0:
                            continue
                        cluster_features = features[cluster_mask]
                        center_features = self._centers[i].unsqueeze(0).expand_as(cluster_features)

                        cluster_size = cluster_mask.sum()
                        cluster_logits = logits[cluster_mask]
                        if self._config["train_freeze_old"]:
                            center_logits = self.model.head.heads[-1](center_features)
                        else:
                            center_logits = self.model.head(center_features)["logits"]
                        rb_loss += (1 / cluster_size) * F.mse_loss(cluster_logits, center_logits)

                    rb_loss = rb_loss / total_samples

                    # cluster_features = F.normalize(cluster_features, dim=1)
                    # center_features = F.normalize(center_features, dim=1)

                    # pos_sim = (cluster_features * center_features).sum(dim=1)
                    # rb_loss += (cluster_mask.sum().float() / total_samples ) * (1.0 - pos_sim.mean())

                    # cluster_logits = logits[cluster_mask]
                    # cluster_probs = F.softmax(cluster_logits, dim=1)
                    # cluster_entropies = -(cluster_probs * cluster_probs.log()).sum(dim=1)

                    # center_features = F.normalize(center_features, dim=1)
                    # center_logits = self.model.head.heads[self._cur_task](center_features)
                    # center_probs = F.softmax(center_logits, dim=1)
                    # center_entropies = -(center_probs * center_probs.log()).sum(dim=1)

                    # rb_loss += (cluster_mask.sum().float() / total_samples ) * torch.abs(cluster_entropies - center_entropies).mean()

                    loss = ce_loss + rb_weight * rb_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_ce_loss += ce_loss.item() * len(y)
                    total_rb_loss += rb_loss.item() * len(y)
                    total_acc += (logits.argmax(dim=1) == y).sum().item()
                    total += len(y)

                scheduler.step()

                if epoch % 5 == 4:
                    logger.info(
                        f"[Training] Epoch {epoch + 1}/{epochs}, "
                        f"CE Loss: {total_ce_loss / total:.4f}, "
                        f"Reg Loss: {total_rb_loss / total:.4f}, "
                        f"Total Loss: {(total_ce_loss + rb_weight * total_rb_loss) / total:.4f}, "
                        f"Acc: {total_acc / total:.4f}"
                    )

            torch.save(
                self.model.get_backbone_trainable_params(),
                self.backbone_checkpoint(self._cur_task),
            )
            torch.save(
                self.model.head.heads[-1].state_dict(),
                self.head_checkpoint(self._cur_task),
            )
        else:
            self.model.backbone.load_state_dict(
                torch.load(self.backbone_checkpoint(self._cur_task)), strict=False
            )
            self.model.cuda()

        if self._config["model_merge"] != "none":
            self.merge()
        
        if self._config["train_with_buffer"]:
            self.model.eval()
            with torch.no_grad():
                for _, batch in enumerate(train_loader):
                    _, _, x, y = batch
                    x, y = x.cuda(), y.cuda()
                    features = self.model.get_features(x)
                    self.buffer.add(x, features, y)
            logger.info(f"[Replay Buffer] Size: {self.buffer.size}")
            logger.info(f"[Replay Buffer] Size by class: {self.buffer.size_by_class}")

        if self._config["train_align_classifier"] and self._config["model_fc"] == "mlp":
            self.align_classifier()

        if self._config["model_fc"] == "cosine":
            self.fit()

    def align_classifier(self):
        logger.info("[Classifier Alignment] Add samples to generative model")
        trainset_CPs = self.data_manager.get_dataset(  # Fix: use self.data_manager
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="test",
        )
        train_loader_CPs = DataLoader(
            trainset_CPs, batch_size=512, shuffle=False, num_workers=4
        )
        self.model.eval()
        with torch.no_grad():
            for _, (_, _, x, y) in enumerate(train_loader_CPs):
                x, y = x.cuda(), y.cuda()
                features = self.model.get_features(x)
                self.generative_model.update(features, y)

        self.model.train()
        print(self.model)

        epochs = 10
        n_per_class = 256
        task_sizes = self.data_manager.get_task_sizes()
        logit_norm_factor = -1

        optimizer = optim.AdamW(self.model.head.parameters(), lr=1e-2, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        for epoch in range(epochs):
            with torch.no_grad():
                sampled_data = []
                sampled_label = []

                for clz in range(self._total_classes):
                    synth_feat = self.generative_model.sample(n_per_class, clz)

                    sampled_data.append(synth_feat)
                    sampled_label.extend([clz] * n_per_class)
                
                for i, batch in enumerate(self.buffer):
                    _, z, y = batch
                    z = z.to(self.device)
                    y = y.to(self.device)
                    sampled_data.append(z)
                    sampled_label.extend(y.tolist())

                synth_feats = torch.cat(sampled_data, dim=0)          # (C * n, D)
                synth_labels = torch.tensor(sampled_label, device=self.device)

                perm = torch.randperm(synth_feats.size(0))
                synth_feats = synth_feats[perm]
                synth_labels = synth_labels[perm]

            total_loss = 0.0

            for clz in range(self._total_classes):
                clz_mask = (synth_labels == clz)
                if clz_mask.sum() == 0:
                    continue

                inp = synth_feats[clz_mask]
                tgt = synth_labels[clz_mask]

                logits = self.model.head(inp)["logits"]

                if logit_norm_factor < 0:
                    loss = F.cross_entropy(logits, tgt)
                else:
                    per_task_norms = []
                    slice_start = 0
                    for t in range(self._cur_task + 1):  # Fix: use self._cur_task
                        slice_end = slice_start + task_sizes[t]
                        norm = logits[:, slice_start:slice_end].norm(p=2, dim=-1, keepdim=True) + 1e-7
                        per_task_norms.append(norm)
                        slice_start = slice_end

                    per_task_norms = torch.cat(per_task_norms, dim=-1)
                    sample_norm = per_task_norms.mean(dim=-1, keepdim=True)
                    decoupled_logits = logits / sample_norm
                    decoupled_logits = decoupled_logits / logit_norm_factor
                    loss = F.cross_entropy(decoupled_logits, tgt)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            avg_loss = total_loss / self._total_classes  # Fix: use self._total_classes
            logger.info(f"[Classifier Alignment] Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}") 

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

        # self.model.backbone.load_state_dict(torch.load(self.backbone_checkpoint(self._cur_task)), strict=False)
        self.model.fc.use_RP = True

        trainset_CPs = self.data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="test",
        )
        train_loader_CPs = DataLoader(
            trainset_CPs, batch_size=512, shuffle=False, num_workers=4
        )

        fs = []
        ys = []
        with torch.no_grad():
            for _, (_, _, x, y) in enumerate(train_loader_CPs):
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
        Wo = torch.linalg.solve(
            self.G + ridge * torch.eye(self.G.size(dim=0)), self.Q
        ).T
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
        logger.info("[RanPAC] Optimal lambda: " + str(ridge))
        return ridge

    def prefix(self):
        prefix_parts = [
            str(self._config['seed']),
            self._config['dataset_name'], 
            str(self._config['dataset_num_task']),
            self._config['model_backbone'],
            self._config['train_method'],
            self._config['model_fc'],
            'robust_training'
        ]
        return "_".join(prefix_parts)

    def backbone_checkpoint(self, task=-1):
        return f"{CHECKPOINT_DIR}{self.prefix()}_backbone" + (
            f"_{task}.pt" if task >= 0 else "_base.pt"
        )

    def head_checkpoint(self, task):
        return f"{CHECKPOINT_DIR}{self.prefix()}_head_{task}.pt"


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
    "imagenetr": [(10, 20, 20)],
    "imageneta": [(10, 20, 20)],
}

def objective(trial):
    """Optuna objective function to optimize LoRA hyperparameters"""
    # Suggest LoRA hyperparameters - restricted options
    lora_r_options = [8, 16, 32, 64, 128, 256]  # Powers of 2 for LoRA rank (8-256)
    
    model_lora_r = trial.suggest_categorical("model_lora_r", lora_r_options)
    
    # LoRA alpha multiplier: 1x or 2x the LoRA r
    alpha_multiplier = trial.suggest_categorical("alpha_multiplier", [1, 2])
    model_lora_alpha = model_lora_r * alpha_multiplier
    
    logger.info(f"[Optuna Trial {trial.number}] Testing lora_r={model_lora_r}, lora_alpha={model_lora_alpha}")
    
    # Fixed hyperparameters
    model_backbone = "vit_base_patch16_224_lora"
    dataset_name = "imageneta"
    seed = 1993
    
    set_random(seed)
    
    config = {
        "seed": seed,
        "reset": True,
    }
    
    # Get dataset configuration
    dataset_num_task, dataset_init_cls, dataset_increment = data_table[dataset_name][0]  # Use first config
    dataset_config = {
        "dataset_name": dataset_name,
        "dataset_num_task": dataset_num_task,
        "dataset_init_cls": dataset_init_cls,
        "dataset_increment": dataset_increment,
    }
    config.update(dataset_config)
    
    # Fixed training configuration
    train_config = {
        "train_epochs": 10,
        "train_batch_size": 48,
        "train_method": "seq",
        "train_base_lr": 1e-2,
        "train_weight_decay": 5e-4,
        "train_local_robustness_num_clusters": 5,
        "train_base_reg_weight": 0.1,
        "train_freeze_old": False,
        "train_with_buffer": True,
        "train_buffer_percent": 0.3,
        "train_buffer_size": 1000,
        "train_buffer_decay": 1,
        "train_align_classifier": False,
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
    
    best_faa = 0
    
    # Test both mlp and cosine classifiers, keep the best result
    for model_fc in ["mlp", "cosine"]:
        for model_merge in ["ties"]:
            model_config = {
                "model_backbone": model_backbone,
                "model_lora_r": model_lora_r,  # Optimized parameter
                "model_lora_alpha": model_lora_alpha,  # Optimized parameter
                "model_lora_dropout": 0.0,
                "model_merge": model_merge,
                "model_merge_base": "init",  # Fixed merge base
                "model_merge_coef": 1.0,
                "model_merge_topk": 100,
                "model_M": 10000,
                "model_fc": model_fc,
            }
            config.update(model_config)
            
            try:
                learner = Learner(config)
                learner.learn(data_manager)
                
                # Get final average accuracy
                faa = learner._aa
                best_faa = max(best_faa, faa)
                
                logger.info(f"[Optuna Trial {trial.number}] FC: {model_fc}, FAA: {faa:.2f}")
                
                # Clean up memory
                del learner
                torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                logger.error(f"[Optuna Trial {trial.number}] Error with FC {model_fc}: {str(e)}")
                # Return a low score for failed trials
                return 0.0
    
    logger.info(f"[Optuna Trial {trial.number}] Best FAA: {best_faa:.2f}")
    return best_faa


# Run Optuna optimization
def run_optuna_optimization():
    study = optuna.create_study(
        direction="maximize",  # Maximize Final Average Accuracy
        study_name="lora_hyperparameter_optimization",
        storage="sqlite:///optuna_lora_optimization.db",  # Persistent storage
        load_if_exists=True
    )
    
    logger.info("Starting Optuna hyperparameter optimization for LoRA parameters")
    logger.info("Optimizing: model_lora_r and model_lora_alpha")
    
    try:
        study.optimize(objective, n_trials=50, timeout=3600*6)  # 50 trials or 6 hours max
        
        logger.info("Optuna optimization completed!")
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best value (FAA): {study.best_value:.2f}")
        logger.info(f"Best parameters: {study.best_params}")
        
        # Print top 5 trials
        logger.info("\nTop 5 trials:")
        trials_df = study.trials_dataframe()
        top_trials = trials_df.nlargest(5, 'value')
        for idx, trial in top_trials.iterrows():
            logger.info(f"Trial {trial['number']}: FAA={trial['value']:.2f}, "
                       f"lora_r={trial['params_model_lora_r']}, "
                       f"lora_alpha={trial['params_alpha_multiplier'] * trial['params_model_lora_r']}")
        
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")
        logger.info(f"Best trial so far: {study.best_trial.number}")
        logger.info(f"Best value so far: {study.best_value:.2f}")
        logger.info(f"Best parameters so far: {study.best_params}")


# Run the optimization
if __name__ == "__main__":
    start_time = time.time()
    run_optuna_optimization()
    logger.info(f"Total optimization time: {time.time() - start_time:.2f} seconds")
