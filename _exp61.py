import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import random
import os
from datetime import datetime
import copy
from generative_model import MultivariateNormalGenerator
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
from _exp import (
    ContinualLearnerHead,
    setup_logger,
    get_backbone,
    merge,
    compute_metrics,
    RandomReplayBuffer,
    weight_init
)
from inc_net import CosineLinear
from util import accuracy, set_random
import gc
import time
import cma

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
            weight = torch.cat([
                weight,
                torch.zeros(num_total_classes - nb_output, self.fc.weight.shape[1]).cuda(),
            ])
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
    def __init__(self, config, logger):
        self._config = config
        self._logger = logger
        self._known_classes = 0
        self._total_classes = 0
        self._class_increments = []
        self._cur_task = -1
        self.mlp_matrix = []
        self.nme_matrix = []
        self._cls_to_task_idx = {}

        self.model = Model(config)
        self.model.cuda()
        torch.save(self.model.get_backbone_trainable_params(), self.backbone_checkpoint())
        self._faa, self._ffm = 0, 0
        
        # Initialize feature generator for CMA-ES alignment
        self.feature_generator = None

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
            "cifar224": 70,
            "imagenetr": 78,
            "imageneta": 59,
            "cub": 85,
            "omnibenchmark": 73,
            "vtab": 87
        }
        self._aa = 0
        self.data_manager = data_manager
        num_tasks = data_manager.nb_tasks
        self._total_classnum = data_manager.get_total_classnum()

        if self._config["train_robustness"]:
            self._centers = self.sample_cluster_centers(self._total_classnum * self._config["train_local_robustness_num_clusters"])
            self._logger.info(f"[Local Robustness] Cluster centers: {self._centers.shape}")
            
            self._logger.info(f"[Replay Buffer] Total dataset size: {data_manager.train_set_size}")
            buffer_size = max(
                int(data_manager.train_set_size * self._config["train_buffer_percent"]),
                self._config["train_buffer_size"],
            )
            self._logger.info(f"[Replay Buffer] Maximum Size: {buffer_size}")
            self.buffer = RandomReplayBuffer(
                buffer_size, self._config["train_buffer_decay"], seed=self._config["seed"]
            )
        else:
            self.buffer = None

        self.model.cuda()

        for task in range(num_tasks):
            self.before_task(task, data_manager)
            self.train()
            self.eval()
            self.after_task()

            if self._aa < baseline[self._config["dataset_name"]]:
                self._logger.info(
                    f"Baseline accuracy not reached: {self._aa:.2f} < {baseline[self._config['dataset_name']]:.2f}"
                )
                break

    def before_task(self, task, data_manager):
        self._total_classes = self._known_classes + data_manager.get_task_size(task)
        self._class_increments.append((self._known_classes, self._total_classes - 1))
        self._cur_task = task

        for clz in range(self._known_classes, self._total_classes):
            self._cls_to_task_idx[clz] = self._cur_task

        if self._config["model_use_ranpac"]:
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

        results = {}
        for fc_type in self._config["model_classifier"]:
            y_true, y_pred = [], []
            with torch.no_grad():
                for _, (_, _, x, y) in enumerate(test_loader):
                    x, y = x.cuda(), y.cuda()

                    if fc_type == "nme":
                        logits = self.model(x)["logits"]
                    elif fc_type == "mlp":
                        features = self.model.get_features(x)
                        logits = self.model.head(features)["logits"]

                    predicts = logits.argmax(dim=1)
                    y_pred.append(predicts.cpu().numpy())
                    y_true.append(y.cpu().numpy())

            y_pred = np.concatenate(y_pred)
            y_true = np.concatenate(y_true)
            acc_total, grouped = accuracy(y_pred.T, y_true, self._class_increments)
            
            results[fc_type] = {"acc_total": acc_total, "grouped": grouped}
            self._logger.info(f"[Evaluation {fc_type.upper()}] Task {self._cur_task + 1}")
            self._logger.info(f"[Evaluation {fc_type.upper()}] Total Acc: {acc_total:.2f}, Grouped Acc: {grouped}")

        if "mlp" in self._config["model_classifier"]:
            self.mlp_matrix.append(results["mlp"]["grouped"])
            num_tasks = len(self.mlp_matrix)
            mlp_accuracy_matrix = np.zeros((num_tasks, num_tasks))
            for i in range(num_tasks):
                for j in range(i + 1):
                    mlp_accuracy_matrix[i, j] = self.mlp_matrix[i][j]

            mlp_faa, mlp_ffm, mlp_ffd, mlp_asa = compute_metrics(mlp_accuracy_matrix)
            self._mlp_faa = mlp_faa
            self._mlp_asa = mlp_asa
            self._logger.info(f"[Evaluation MLP] FAA: {mlp_faa:.2f}, FFM: {mlp_ffm:.2f}, FFD: {mlp_ffd:.2f}, ASA: {mlp_asa:.2f}")
        else:
            self._mlp_faa = 0.0
            self._mlp_asa = 0.0
        
        if "nme" in self._config["model_classifier"]:
            self.nme_matrix.append(results["nme"]["grouped"])
            num_tasks = len(self.nme_matrix)
            nme_accuracy_matrix = np.zeros((num_tasks, num_tasks))
            for i in range(num_tasks):
                for j in range(i + 1):
                    nme_accuracy_matrix[i, j] = self.nme_matrix[i][j]
            nme_faa, nme_ffm, nme_ffd, nme_asa = compute_metrics(nme_accuracy_matrix)
            self._nme_faa = nme_faa
            self._nme_asa = nme_asa
            self._logger.info(f"[Evaluation NME] FAA: {nme_faa:.2f}, FFM: {nme_ffm:.2f}, FFD: {nme_ffd:.2f}, ASA: {nme_asa:.2f}")
        else:
            self._nme_faa = 0.0
            self._nme_asa = 0.0
        
        self._aa = max(self._mlp_faa, self._nme_faa)

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
            freeze_old = not self._config["train_robustness"]
            self.model.update_head(
                self._total_classes - self._known_classes,
                freeze_old=freeze_old,
            )
            self.model.cuda()

            epochs = self._config["train_epochs"]
            base_lr = self._config["train_base_lr"]
            weight_decay = self._config["train_weight_decay"]

            parameters = [
                {
                    "params": [p for p in self.model.backbone.parameters() if p.requires_grad],
                    "lr": base_lr,
                    "weight_decay": weight_decay,
                },
                {
                    "params": [p for p in self.model.head.heads[self._cur_task].parameters() if p.requires_grad],
                    "lr": base_lr,
                    "weight_decay": weight_decay,
                },
            ]
            
            if not freeze_old:
                parameters.append({
                    "params": [
                        p for i, head in enumerate(self.model.head.heads)
                        if i != self._cur_task
                        for p in head.parameters()
                        if p.requires_grad
                    ],
                    "lr": base_lr * 1e-2,
                    "weight_decay": weight_decay * 1e-2,
                })

            optimizer = optim.SGD(parameters, lr=base_lr, momentum=0.9, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

            self.model.train()
            self._logger.info(f"[Training] Task {self._cur_task + 1}")
            self._logger.info(f"[Training] {self.model}")

            rb_weight = self._config["train_base_reg_weight"]
            for epoch in range(epochs):
                total_ce_loss, total_rb_loss, total_acc, total = 0, 0, 0, 0

                for _, (_, x_aug, x, y) in enumerate(train_loader):
                    x_aug, x, y = x_aug.cuda(), x.cuda(), y.cuda()

                    if self._config["train_robustness"] and self.buffer.size > 0:
                        x_buf, z_buf, y_buf = self.buffer.sample(batch_size=x.size(0))
                        x_buf, z_buf, y_buf = x_buf.cuda(), z_buf.cuda(), y_buf.cuda()
                        x = torch.cat([x, x_buf], dim=0)
                        y = torch.cat([y, y_buf], dim=0)

                    features = self.model.get_features(x)

                    if freeze_old:
                        y = torch.where(y - self._known_classes >= 0, y - self._known_classes, -100)
                        logits = self.model.head.heads[-1](features)
                    else:
                        logits = self.model.head(features)["logits"]

                    ce_loss = F.cross_entropy(logits, y)
                    rb_loss = torch.tensor(0.0, device=x.device)

                    if self._config["train_robustness"] and self.buffer.size > 0:
                        with torch.no_grad():
                            sim_matrix = torch.matmul(features, self._centers.T)
                            nearest_idx = sim_matrix.argmax(dim=1)

                        num_clusters = self._centers.shape[0]
                        total_samples = x.size(0)
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
                            center_logits = self.model.head(center_features)["logits"]
                            rb_loss += ((1 / cluster_size) * F.mse_loss(cluster_logits, center_logits)) / total_samples

                            # cluster_features = F.normalize(cluster_features, dim=1)
                            # center_features = F.normalize(center_features, dim=1)

                            # pos_sim = (cluster_features * center_features).sum(dim=1)
                            # rb_loss += ( cluster_mask.sum().float() / total_samples ) * (1.0 - pos_sim.mean())

                            # cluster_logits = logits[cluster_mask]
                            # cluster_probs = F.softmax(cluster_logits, dim=1)
                            # cluster_entropies = -(cluster_probs * cluster_probs.log()).sum(dim=1)

                            # # center_features = F.normalize(center_features, dim=1)
                            # center_logits = self.model.head(center_features)["logits"]
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
                    self._logger.info(
                        f"[Training] Epoch {epoch + 1}/{epochs}, "
                        f"CE Loss: {total_ce_loss / total:.4f}, "
                        f"Reg Loss: {total_rb_loss / total:.4f}, "
                        f"Total Loss: {(total_ce_loss + rb_weight * total_rb_loss) / total:.4f}, "
                        f"Acc: {total_acc / total:.4f}"
                    )

            torch.save(self.model.get_backbone_trainable_params(), self.backbone_checkpoint(self._cur_task))
            torch.save(self.model.head.heads[-1].state_dict(), self.head_checkpoint(self._cur_task))
        else:
            self.model.backbone.load_state_dict(
                torch.load(self.backbone_checkpoint(self._cur_task)), strict=False
            )
            freeze_old = not self._config["train_robustness"]
            self.model.update_head(
                self._total_classes - self._known_classes,
                freeze_old=freeze_old,
            )
            self.model.head.heads[-1].load_state_dict(
                torch.load(self.head_checkpoint(self._cur_task)), strict=True
            )
            self.model.cuda()

        if self._config["model_merge"] != "none":
            self.merge()
        
        if self._config["train_robustness"]:
            self.model.eval()
            with torch.no_grad():
                for _, batch in enumerate(train_loader):
                    _, _, x, y = batch
                    x, y = x.cuda(), y.cuda()
                    features = self.model.get_features(x)
                    self.buffer.add(x, features, y)
            self._logger.info(f"[Replay Buffer] Size: {self.buffer.size}")

        if self._config["train_ca"]:
            # if self.buffer is None:
            #     buffer_size = max(
            #         int(self.data_manager.train_set_size * self._config["train_buffer_percent"]),
            #         self._config["train_buffer_size"],
            #     )
            #     self._logger.info(f"[Replay Buffer] Maximum Size: {buffer_size}")
            #     self.buffer = RandomReplayBuffer(
            #         buffer_size, self._config["train_buffer_decay"], seed=self._config["seed"]
            #     )

            # self.model.eval()
            # with torch.no_grad():
            #     for _, batch in enumerate(train_loader):
            #         _, _, x, y = batch
            #         x, y = x.cuda(), y.cuda()
            #         features = self.model.get_features(x)
            #         self.buffer.add(x, features, y)
            # self._logger.info(f"[Replay Buffer] Size: {self.buffer.size}")

            self.align_classifier()

        # if "nme" in self._config["model_classifier"]:
        #     self.fit()

        if self.buffer is not None:
            self.buffer.update_weights()

    def align_classifier(self):
        # ----------------------------
        # 0) Build / update feature generator from *new* classes only
        # ----------------------------
        if self.feature_generator is None:
            self.feature_generator = MultivariateNormalGenerator(
                self.model.feature_dim, self._total_classnum, device="cuda"
            )

        for class_idx in range(self._known_classes, self._total_classes):
            trainset_cls = self.data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
            )
            cls_loader = DataLoader(trainset_cls, batch_size=512, shuffle=False, num_workers=4)

            features_list, labels_list = [], []
            self.model.eval()
            with torch.no_grad():
                for _, (_, _, x, y) in enumerate(cls_loader):
                    x, y = x.cuda(), y.cuda()
                    feats = self.model.get_features(x)
                    features_list.append(feats)
                    labels_list.append(y)

            if features_list:
                all_features = torch.cat(features_list, dim=0)
                all_labels = torch.cat(labels_list, dim=0)
                self.feature_generator.update(all_features, all_labels)

        if self._cur_task == 0:
            return  # No alignment needed for first task

        self._logger.info("[Alignment] Starting NES head alignment with synthetic data")

        # ----------------------------
        # 1) Generate a *balanced* synthetic set (equal per class), no magnitude skew
        # ----------------------------
        sampled_data, sampled_label = [], []
        num_sampled_pcls = self._config.get("train_ca_samples_per_class", 100)

        for c_id in range(self._total_classes):
            if self.feature_generator.can_sample(c_id):
                try:
                    feats = self.feature_generator.sample(num_sampled_pcls, c_id)  # [K, D]
                    feats = F.layer_norm(feats, (feats.shape[-1],))                # reduce magnitude bias
                    sampled_data.append(feats)
                    sampled_label.extend([c_id] * num_sampled_pcls)
                except Exception as e:
                    self._logger.warning(f"[Alignment] Failed to sample from class {c_id}: {e}")
                    continue

        if not sampled_data:
            self._logger.warning("No samples generated, skipping classifier alignment")
            return

        all_features = torch.cat(sampled_data, dim=0).float().cuda(non_blocking=True)
        all_labels = torch.tensor(sampled_label).long().cuda(non_blocking=True)
        self._logger.info(f"[Alignment] Generated synthetic dataset: {all_features.shape[0]} samples, {self._total_classes} classes")

        # ----------------------------
        # 2) Head → class range mapping and stitched logits (task-free inference)
        #    We assume self._class_increments = [(s0,e0), (s1,e1), ...] for seen tasks.
        # ----------------------------
        head_class_ranges = {t: (rng[0], rng[1]) for t, rng in enumerate(self._class_increments[: self._cur_task + 1])}
        num_global_classes = self._total_classes

        @torch.no_grad()
        def stitched_logits_from_heads(features):
            """Compute a global logit matrix by stitching per-head outputs into global class indices."""
            B = features.size(0)
            out = features.new_full((B, num_global_classes), float("-inf"))
            for t, head in enumerate(self.model.head.heads):
                if t > self._cur_task:
                    continue
                c_lo, c_hi = head_class_ranges[t]
                logits_t = head(features)  # [B, C_t]
                out[:, c_lo: c_hi + 1] = logits_t
            return out

        # ----------------------------
        # 3) Record baseline per-class accuracy BEFORE optimization (guardrail)
        # ----------------------------
        baseline_guard = bool(self._config.get("train_ca_guardrail", True))
        epsilon_drop = float(self._config.get("train_ca_guardrail_eps", 0.20))  # tolerate 20% abs drop for recent task improvement
        alpha_guard = float(self._config.get("train_ca_guardrail_alpha", 1.0))  # penalty weight

        self._ca_baseline_acc_per_class = None
        if baseline_guard:
            with torch.no_grad():
                base_logits = stitched_logits_from_heads(all_features)
                self._ca_baseline_acc_per_class = {}
                for c in range(num_global_classes):
                    m = (all_labels == c)
                    if m.any():
                        acc_c = (base_logits[m].argmax(1) == all_labels[m]).float().mean().item()
                        self._ca_baseline_acc_per_class[c] = acc_c
            self._logger.info(f"[Alignment] Baseline guard ON. Recorded {len(self._ca_baseline_acc_per_class)} class baselines.")

        # ----------------------------
        # 4) Build initial genome from *all heads* (weights + biases)
        # ----------------------------
        head_params = []
        param_shapes = []
        head_param_slices = {}  # per-head slices for trust region penalty
        offset = 0

        for task_idx in range(self._cur_task + 1):
            head = self.model.head.heads[task_idx]
            head_start = offset
            for p in head.parameters():
                arr = p.data.detach().cpu().numpy().flatten()
                head_params.append(arr)
                param_shapes.append(p.shape)
                offset += arr.size
            head_end = offset
            head_param_slices[task_idx] = (head_start, head_end)

        theta = np.concatenate(head_params).astype(np.float32)  # NES mean (current params)
        original_solution = theta.copy()

        # ----------------------------
        # 5) Per-head trust region strengths
        # ----------------------------
        lambda_cfg = self._config.get("train_ca_lambda_head", {})
        default_old = float(self._config.get("train_ca_lambda_old_default", 1e-4))
        default_cur = float(self._config.get("train_ca_lambda_cur_default", 1e-5))

        lambda_task = {}
        for t in range(self._cur_task + 1):
            lam = lambda_cfg.get(str(t), lambda_cfg.get(t, default_cur if t == self._cur_task else default_old))
            lambda_task[t] = float(lam)

        # ----------------------------
        # 6) Macro-class accuracy helper
        # ----------------------------
        @torch.no_grad()
        def macro_class_accuracy(logits, labels):
            accs = []
            for c in range(num_global_classes):
                m = (labels == c)
                if m.any():
                    acc_c = (logits[m].argmax(1) == labels[m]).float().mean().item()
                    accs.append(acc_c)
            return float(np.mean(accs)) if accs else 0.0, accs  # (macro, per-class list)

        # ----------------------------
        # 7) Objective (returns scalar *loss* to MINIMIZE)
        # ----------------------------
        def objective_function(params_flat: np.ndarray) -> float:
            try:
                # Install candidate weights (snap + restore pattern)
                param_idx = 0
                original_params_snap = {}
                for task_idx in range(self._cur_task + 1):
                    head = self.model.head.heads[task_idx]
                    original_params_snap[task_idx] = []
                    for _, p in head.named_parameters():
                        original_params_snap[task_idx].append(p.data.clone())
                        sz = p.numel()
                        new_data = params_flat[param_idx:param_idx + sz]
                        param_idx += sz
                        p.data = torch.from_numpy(new_data).float().cuda().view(p.shape)

                # Forward: stitched logits (no routing) and macro-class reward
                with torch.no_grad():
                    logits = stitched_logits_from_heads(all_features)
                    macro_acc, acc_list = macro_class_accuracy(logits, all_labels)
                    
                    # Optional: Entropy regularization to prevent overconfident predictions
                    entropy_reg = 0.0
                    if self._config.get("train_ca_entropy_reg", False):
                        probs = torch.softmax(logits, dim=1)
                        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
                        entropy_weight = float(self._config.get("train_ca_entropy_weight", 0.1))
                        entropy_reg = -entropy_weight * entropy.item()  # Negative because we want to maximize entropy

                # Per-head MSE trust region (mean-squared movement)
                delta = params_flat - original_solution
                prox = 0.0
                for t in range(self._cur_task + 1):
                    s, e = head_param_slices[t]
                    dt = delta[s:e]
                    n_t = max((e - s), 1)
                    
                    # Option 1: Standard L2 trust region (current)
                    l2_penalty = lambda_task[t] * float(np.dot(dt, dt) / n_t)
                    
                    # Option 2: Exponential decay penalty (prioritize recent tasks)
                    age_factor = float(self._config.get("train_ca_age_decay", 1.0)) ** (self._cur_task - t)
                    exp_penalty = lambda_task[t] * age_factor * float(np.dot(dt, dt) / n_t)
                    
                    # Option 3: Adaptive penalty based on task performance
                    c_lo, c_hi = head_class_ranges[t]
                    if c_hi < len(acc_list):
                        # Get mean accuracy for this task's classes
                        task_acc = np.mean(acc_list[c_lo:c_hi+1]) if c_hi+1 <= len(acc_list) else np.mean(acc_list[c_lo:])
                        # Higher penalty for worse performing tasks (inverse relationship)
                        performance_weight = 1.0 / max(task_acc, 0.1)  # Avoid division by zero
                        adaptive_penalty = lambda_task[t] * performance_weight * float(np.dot(dt, dt) / n_t)
                    else:
                        adaptive_penalty = l2_penalty  # Fallback to standard penalty
                    
                    # Choose penalty type
                    penalty_type = self._config.get("train_ca_penalty_type", "standard")
                    if penalty_type == "exponential":
                        prox += exp_penalty
                    elif penalty_type == "adaptive":
                        prox += adaptive_penalty
                    else:  # standard
                        prox += l2_penalty

                # Guardrail: penalize per-class drops on former tasks
                guard = 0.0
                if baseline_guard and self._ca_baseline_acc_per_class is not None:
                    for t in range(self._cur_task):  # only former tasks
                        c_lo, c_hi = head_class_ranges[t]
                        for c in range(c_lo, c_hi + 1):
                            base = self._ca_baseline_acc_per_class.get(c, None)
                            if base is not None:
                                now = acc_list[c] if c < len(acc_list) else 0.0
                                drop = (base - now) - epsilon_drop
                                if drop > 0.0:
                                    guard += drop

                loss = -(macro_acc) + prox + entropy_reg + (alpha_guard * guard if baseline_guard else 0.0)

                # Restore original weights
                for task_idx in range(self._cur_task + 1):
                    head = self.model.head.heads[task_idx]
                    for p, w in zip(head.parameters(), original_params_snap[task_idx]):
                        p.data = w

                return float(loss)

            except Exception as e:
                self._logger.warning(f"[Alignment] Error in objective function: {e}")
                return 1.0


        # ---------- Build per-parameter sigma_vec (diagonal σ) with adaptive scheduling ----------
        base_sigma_init = float(self._config.get("train_ca_nes_sigma_init", 1e-3))  # Start large
        base_sigma_final = float(self._config.get("train_ca_nes_sigma_final", 1e-4))  # End small
        sigma_min  = float(self._config.get("train_ca_nes_sigma_min", 1e-3))
        sigma_max  = float(self._config.get("train_ca_nes_sigma_max", 1e-1))
        sigma_decay_type = self._config.get("train_ca_nes_sigma_decay", "constant")  # "exponential", "linear", "cosine"

        # Map trust-region strengths (lambda_task) to per-head sigma scales.
        # Intuition: stronger trust (bigger lambda) -> smaller exploration σ
        def lambda_to_scale(lmb):
            return 1.0 / np.sqrt(max(lmb, 1e-12))

        # Build initial sigma_vec (will be updated each iteration)
        sigma_vec_base = np.empty_like(theta, dtype=np.float32)
        for t in range(self._cur_task + 1):
            s, e = head_param_slices[t]
            scale_t = lambda_to_scale(lambda_task[t])
            sigma_vec_base[s:e] = scale_t  # Store scale factors

        def get_sigma_vec(iteration, total_iters):
            """Compute adaptive sigma for current iteration"""
            # Global sigma schedule
            if sigma_decay_type == "exponential":
                # Exponential decay: σ(t) = σ_init * (σ_final/σ_init)^(t/T)
                decay_factor = (base_sigma_final / base_sigma_init) ** (iteration / max(total_iters - 1, 1))
                current_base_sigma = base_sigma_init * decay_factor
            elif sigma_decay_type == "linear":
                # Linear decay: σ(t) = σ_init - (σ_init - σ_final) * t/T
                alpha = iteration / max(total_iters - 1, 1)
                current_base_sigma = base_sigma_init * (1 - alpha) + base_sigma_final * alpha
            elif sigma_decay_type == "cosine":
                # Cosine annealing: σ(t) = σ_final + 0.5 * (σ_init - σ_final) * (1 + cos(π*t/T))
                alpha = iteration / max(total_iters - 1, 1)
                current_base_sigma = base_sigma_final + 0.5 * (base_sigma_init - base_sigma_final) * (1 + np.cos(np.pi * alpha))
            else:  # "constant"
                current_base_sigma = base_sigma_init
                
            # Apply per-head scaling and clipping
            sigma_vec = current_base_sigma * sigma_vec_base
            return np.clip(sigma_vec, sigma_min, sigma_max).astype(np.float32)

        # ---------- NES loop with adaptive per-parameter σ (vanilla ES + trust-region in objective) ----------
        lr    = float(self._config.get("train_ca_nes_lr", 0.05))
        iters = int(self._config.get("train_ca_nes_iterations", 50))
        pop   = int(self._config.get("train_ca_nes_popsize", 30))  # antithetic PAIRS

        self._logger.info(f"[Alignment][NES-diag] iters={iters}, pop={pop}, lr={lr}")
        self._logger.info(f"[Alignment][NES-diag] sigma schedule: {sigma_decay_type}, init={base_sigma_init:.3e}, final={base_sigma_final:.3e}")

        best_theta = theta.copy()
        best_loss  = objective_function(best_theta)
        no_improve_steps = 0
        patience = int(self._config.get("train_ca_nes_patience", 10))

        for it in range(iters):
            # Get adaptive sigma for current iteration
            sigma_vec = get_sigma_vec(it, iters)
            
            # Antithetic noise (pop, d)
            eps = np.random.randn(pop, theta.size).astype(np.float32)

            # Evaluate θ ± σ_vec * ε_i  (elementwise)
            losses_pos = np.empty(pop, dtype=np.float32)
            losses_neg = np.empty(pop, dtype=np.float32)
            for i in range(pop):
                step = sigma_vec * eps[i]                     # elementwise
                losses_pos[i] = objective_function(theta + step)
                losses_neg[i] = objective_function(theta - step)

            # Vanilla ES gradient with diagonal σ:
            #   grad ≈ (1/(2M)) Σ (ΔL_i * ε_i / σ_vec)
            deltaL = (losses_pos - losses_neg)[:, None]       # (pop,1)
            grad = (deltaL * (eps / np.maximum(sigma_vec, 1e-12))).mean(axis=0) / 2.0  # (d,)
            theta = theta - lr * grad.astype(np.float32)

            # Track best
            cur_loss = objective_function(theta)
            if cur_loss < best_loss - 1e-6:
                best_loss = cur_loss
                best_theta = theta.copy()
                no_improve_steps = 0
            else:
                no_improve_steps += 1

            # Log with sigma info
            if it % 10 == 0 or it == iters - 1:
                # quick metric snapshot (same as your existing logging block)
                param_idx = 0
                snap = {}
                for t in range(self._cur_task + 1):
                    head = self.model.head.heads[t]
                    snap[t] = [p.data.clone() for p in head.parameters()]
                    for p in head.parameters():
                        sz = p.numel()
                        new_data = best_theta[param_idx:param_idx + sz]
                        param_idx += sz
                        p.data = torch.from_numpy(new_data).float().cuda().view(p.shape)
                with torch.no_grad():
                    logits = stitched_logits_from_heads(all_features)
                    macro_acc, _ = macro_class_accuracy(logits, all_labels)
                for t in range(self._cur_task + 1):
                    head = self.model.head.heads[t]
                    for p, w in zip(head.parameters(), snap[t]):
                        p.data = w
                
                current_sigma_range = f"[{sigma_vec.min():.3e}, {sigma_vec.max():.3e}]"
                self._logger.info(f"[Alignment][NES-diag] iter {it}: best_loss={best_loss:.6f}, macro_acc={macro_acc:.4f}, σ_range={current_sigma_range}")

            if no_improve_steps >= patience:
                self._logger.info(f"[Alignment][NES-diag] Early stop at iter {it} (no improvement for {patience}).")
                break


        # ----------------------------
        # 9) Apply best solution
        # ----------------------------
        param_idx = 0
        for t in range(self._cur_task + 1):
            head = self.model.head.heads[t]
            for p in head.parameters():
                sz = p.numel()
                new_data = best_theta[param_idx:param_idx + sz]
                param_idx += sz
                p.data = torch.from_numpy(new_data).float().cuda().view(p.shape)

        with torch.no_grad():
            logits = stitched_logits_from_heads(all_features)
            final_macro, _ = macro_class_accuracy(logits, all_labels)
        self._logger.info(f"[Alignment][NES] Completed. Final Macro-Class Accuracy: {final_macro:.4f}")


        # """NES-based classifier alignment with Fisher (natural) gradient.
        # - Uses stitched logits (no task-id).
        # - Optimizes head params via antithetic NES with Fisher-preconditioned update.
        # - Keeps optional baseline guardrail to protect old classes.
        # """
        # import numpy as np
        # import torch
        # import torch.nn.functional as F
        # from torch.utils.data import DataLoader

        # # ----------------------------
        # # 0) Build / update feature generator from *new* classes only
        # # ----------------------------
        # if self.feature_generator is None:
        #     self.feature_generator = MultivariateNormalGenerator(
        #         self.model.feature_dim, self._total_classnum, device="cuda"
        #     )

        # for class_idx in range(self._known_classes, self._total_classes):
        #     trainset_cls = self.data_manager.get_dataset(
        #         np.arange(class_idx, class_idx + 1),
        #         source="train",
        #         mode="test",
        #     )
        #     cls_loader = DataLoader(trainset_cls, batch_size=512, shuffle=False, num_workers=4)

        #     features_list, labels_list = [], []
        #     self.model.eval()
        #     with torch.no_grad():
        #         for _, (_, _, x, y) in enumerate(cls_loader):
        #             x, y = x.cuda(), y.cuda()
        #             feats = self.model.get_features(x)
        #             features_list.append(feats)
        #             labels_list.append(y)

        #     if features_list:
        #         all_features = torch.cat(features_list, dim=0)
        #         all_labels = torch.cat(labels_list, dim=0)
        #         self.feature_generator.update(all_features, all_labels)

        # if self._cur_task == 0:
        #     return  # No alignment needed for first task

        # self._logger.info("[Alignment] Starting NES (natural gradient) head alignment with synthetic data")

        # # ----------------------------
        # # 1) Generate a *balanced* synthetic set (equal per class), no magnitude skew
        # # ----------------------------
        # sampled_data, sampled_label = [], []
        # num_sampled_pcls = self._config.get("train_ca_samples_per_class", 50)

        # for c_id in range(self._total_classes):
        #     if self.feature_generator.can_sample(c_id):
        #         try:
        #             feats = self.feature_generator.sample(num_sampled_pcls, c_id)  # [K, D]
        #             feats = F.layer_norm(feats, (feats.shape[-1],))                # reduce magnitude bias
        #             sampled_data.append(feats)
        #             sampled_label.extend([c_id] * num_sampled_pcls)
        #         except Exception as e:
        #             self._logger.warning(f"[Alignment] Failed to sample from class {c_id}: {e}")
        #             continue

        # if not sampled_data:
        #     self._logger.warning("No samples generated, skipping classifier alignment")
        #     return

        # all_features = torch.cat(sampled_data, dim=0).float().cuda(non_blocking=True)
        # all_labels = torch.tensor(sampled_label).long().cuda(non_blocking=True)
        # self._logger.info(f"[Alignment] Generated synthetic dataset: {all_features.shape[0]} samples, {self._total_classes} classes")

        # # ----------------------------
        # # 2) Head → class range mapping and stitched logits (task-free inference)
        # #    Assumes self._class_increments = [(s0,e0), (s1,e1), ...] for seen tasks.
        # # ----------------------------
        # head_class_ranges = {t: (rng[0], rng[1]) for t, rng in enumerate(self._class_increments[: self._cur_task + 1])}
        # num_global_classes = self._total_classes

        # @torch.no_grad()
        # def stitched_logits_from_heads(features):
        #     """Compute a global logit matrix by stitching per-head outputs into global class indices."""
        #     B = features.size(0)
        #     out = features.new_full((B, num_global_classes), float("-inf"))
        #     for t, head in enumerate(self.model.head.heads):
        #         if t > self._cur_task:
        #             continue
        #         c_lo, c_hi = head_class_ranges[t]
        #         logits_t = head(features)  # [B, C_t]
        #         out[:, c_lo: c_hi + 1] = logits_t
        #     return out

        # # ----------------------------
        # # 3) Record baseline per-class accuracy BEFORE optimization (guardrail)
        # # ----------------------------
        # baseline_guard = bool(self._config.get("train_ca_guardrail", False))
        # epsilon_drop = float(self._config.get("train_ca_guardrail_eps", 0.01))  # tolerate 1% abs drop
        # alpha_guard = float(self._config.get("train_ca_guardrail_alpha", 1.0))  # penalty weight

        # self._ca_baseline_acc_per_class = None
        # if baseline_guard:
        #     with torch.no_grad():
        #         base_logits = stitched_logits_from_heads(all_features)
        #         self._ca_baseline_acc_per_class = {}
        #         for c in range(num_global_classes):
        #             m = (all_labels == c)
        #             if m.any():
        #                 acc_c = (base_logits[m].argmax(1) == all_labels[m]).float().mean().item()
        #                 self._ca_baseline_acc_per_class[c] = acc_c
        #     self._logger.info(f"[Alignment] Baseline guard ON. Recorded {len(self._ca_baseline_acc_per_class)} class baselines.")

        # # ----------------------------
        # # 4) Build initial genome from *all heads* (weights + biases)
        # # ----------------------------
        # head_params = []
        # param_shapes = []
        # for task_idx in range(self._cur_task + 1):
        #     head = self.model.head.heads[task_idx]
        #     for p in head.parameters():
        #         head_params.append(p.data.detach().cpu().numpy().flatten())
        #         param_shapes.append(p.shape)

        # theta = np.concatenate(head_params).astype(np.float32)  # NES mean (current params)

        # # ----------------------------
        # # 5) Macro-class accuracy helper
        # # ----------------------------
        # @torch.no_grad()
        # def macro_class_accuracy(logits, labels):
        #     accs = []
        #     for c in range(num_global_classes):
        #         m = (labels == c)
        #         if m.any():
        #             acc_c = (logits[m].argmax(1) == labels[m]).float().mean().item()
        #             accs.append(acc_c)
        #     return float(np.mean(accs)) if accs else 0.0, accs  # (macro, per-class list)

        # # ----------------------------
        # # 6) Objective (returns scalar *loss* to MINIMIZE)
        # #     Natural gradient means: DO NOT add L2 trust-region here.
        # # ----------------------------
        # def objective_function(params_flat: np.ndarray) -> float:
        #     try:
        #         # Install candidate weights (snap + restore pattern)
        #         param_idx = 0
        #         original_params_snap = {}
        #         for task_idx in range(self._cur_task + 1):
        #             head = self.model.head.heads[task_idx]
        #             original_params_snap[task_idx] = []
        #             for _, p in head.named_parameters():
        #                 original_params_snap[task_idx].append(p.data.clone())
        #                 sz = p.numel()
        #                 new_data = params_flat[param_idx:param_idx + sz]
        #                 param_idx += sz
        #                 p.data = torch.from_numpy(new_data).float().cuda().view(p.shape)

        #         # Forward: stitched logits (no routing) and macro-class reward
        #         with torch.no_grad():
        #             logits = stitched_logits_from_heads(all_features)
        #             macro_acc, acc_list = macro_class_accuracy(logits, all_labels)

        #         # Guardrail: penalize per-class drops on former tasks
        #         guard = 0.0
        #         if baseline_guard and self._ca_baseline_acc_per_class is not None:
        #             for t in range(self._cur_task):  # only former tasks
        #                 c_lo, c_hi = head_class_ranges[t]
        #                 for c in range(c_lo, c_hi + 1):
        #                     base = self._ca_baseline_acc_per_class.get(c, None)
        #                     if base is not None:
        #                         now = acc_list[c] if c < len(acc_list) else 0.0
        #                         drop = (base - now) - epsilon_drop
        #                         if drop > 0.0:
        #                             guard += drop

        #         loss = -(macro_acc) + (alpha_guard * guard if baseline_guard else 0.0)

        #         # Restore original weights
        #         for task_idx in range(self._cur_task + 1):
        #             head = self.model.head.heads[task_idx]
        #             for p, w in zip(head.parameters(), original_params_snap[task_idx]):
        #                 p.data = w

        #         return float(loss)

        #     except Exception as e:
        #         self._logger.warning(f"[Alignment] Error in objective function: {e}")
        #         return 1.0

        # # ----------------------------
        # # 7) NES loop (antithetic sampling, Fisher/natural gradient update)
        # # ----------------------------
        # sigma = float(self._config.get("train_ca_nes_sigma", 0.03))   # exploration std
        # lr    = float(self._config.get("train_ca_nes_lr", 0.08))      # learning rate (slightly smaller; natural step is larger)
        # iters = int(self._config.get("train_ca_nes_iterations", 60))
        # pop   = int(self._config.get("train_ca_nes_popsize", 30))     # number of *pairs*

        # self._logger.info(f"[Alignment][NES-NG] iters={iters}, popsize(pairs)={pop}, sigma={sigma}, lr={lr}")

        # best_theta = theta.copy()
        # best_loss = objective_function(best_theta)

        # no_improve_steps = 0
        # patience = int(self._config.get("train_ca_nes_patience", 10))

        # for it in range(iters):
        #     # Antithetic perturbations
        #     eps = np.random.randn(pop, theta.size).astype(np.float32)

        #     # Evaluate losses for θ±σ·ε
        #     losses_pos = np.empty(pop, dtype=np.float32)
        #     losses_neg = np.empty(pop, dtype=np.float32)
        #     for i in range(pop):
        #         losses_pos[i] = objective_function(theta + sigma * eps[i])
        #         losses_neg[i] = objective_function(theta - sigma * eps[i])

        #     # ---- Fisher (natural) gradient wrt mean:
        #     # g_nat ≈ (1/(2M)) Σ (L+ - L-) * ε       (note: NO division by σ)
        #     deltaL = (losses_pos - losses_neg)[:, None]      # (pop, 1)
        #     g_nat  = (deltaL * eps).mean(axis=0) / 2.0       # (d,)

        #     # Update params (gradient descent on loss)
        #     theta = theta - lr * g_nat.astype(np.float32)

        #     # Track best
        #     cur_loss = objective_function(theta)
        #     if cur_loss < best_loss - 1e-6:
        #         best_loss = cur_loss
        #         best_theta = theta.copy()
        #         no_improve_steps = 0
        #     else:
        #         no_improve_steps += 1

        #     # Periodic logging
        #     if it % 10 == 0 or it == iters - 1:
        #         # Temporarily install best to report macro acc
        #         param_idx = 0
        #         snap = {}
        #         for t in range(self._cur_task + 1):
        #             head = self.model.head.heads[t]
        #             snap[t] = [p.data.clone() for p in head.parameters()]
        #             for p in head.parameters():
        #                 sz = p.numel()
        #                 new_data = best_theta[param_idx:param_idx + sz]
        #                 param_idx += sz
        #                 p.data = torch.from_numpy(new_data).float().cuda().view(p.shape)
        #         with torch.no_grad():
        #             logits = stitched_logits_from_heads(all_features)
        #             macro_acc, _ = macro_class_accuracy(logits, all_labels)
        #         # restore
        #         for t in range(self._cur_task + 1):
        #             head = self.model.head.heads[t]
        #             for p, w in zip(head.parameters(), snap[t]):
        #                 p.data = w
        #         self._logger.info(f"[Alignment][NES-NG] iter {it}: best_loss={best_loss:.6f}, macro_acc={macro_acc:.4f}")

        #     # Early stop on plateau
        #     if no_improve_steps >= patience:
        #         self._logger.info(f"[Alignment][NES-NG] Early stop at iter {it} (no improvement for {patience} steps).")
        #         break

        # # ----------------------------
        # # 8) Apply best solution
        # # ----------------------------
        # param_idx = 0
        # for t in range(self._cur_task + 1):
        #     head = self.model.head.heads[t]
        #     for p in head.parameters():
        #         sz = p.numel()
        #         new_data = best_theta[param_idx:param_idx + sz]
        #         param_idx += sz
        #         p.data = torch.from_numpy(new_data).float().cuda().view(p.shape)

        # with torch.no_grad():
        #     logits = stitched_logits_from_heads(all_features)
        #     final_macro, _ = macro_class_accuracy(logits, all_labels)
        # self._logger.info(f"[Alignment][NES-NG] Completed. Final Macro-Class Accuracy: {final_macro:.4f}")


    def merge(self):
        self._logger.info(f"[Merging] Method {self._config['model_merge']}")
        if self._config["model_merge_base"] == "init":
            base_params = torch.load(self.backbone_checkpoint(-1))
        elif self._config["model_merge_base"] == "first":
            base_params = torch.load(self.backbone_checkpoint(0))
        
        task_params = [torch.load(self.backbone_checkpoint(task)) for task in range(self._cur_task + 1)]
        backbone_params = merge(
            base_params,
            task_params,
            method=self._config["model_merge"],
            lamb=self._config["model_merge_coef"],
            topk=self._config["model_merge_topk"],
        )
        self.model.backbone.load_state_dict(backbone_params, strict=False)

    def fit(self):
        trainset_CPs = self.data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="test",
        )
        train_loader_CPs = DataLoader(trainset_CPs, batch_size=512, shuffle=False, num_workers=4)
        
        if self._config["model_use_ranpac"]:
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

            self.model.fc.use_RP = True

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
            Wo = torch.linalg.solve(self.G + ridge * torch.eye(self.G.size(dim=0)), self.Q).T
            self.model.fc.weight.data = Wo[0 : self._total_classes, :].cuda()
        else:
            self.model.update_fc(self._total_classes)
            embedding_list = []
            label_list = []
            with torch.no_grad():
                for i, batch in enumerate(train_loader_CPs):
                    (_, _, data, label) = batch
                    data = data.cuda()
                    label = label.cuda()
                    embedding = self.model.get_features(data)
                    embedding_list.append(embedding.cpu())
                    label_list.append(label.cpu())

            embedding_list = torch.cat(embedding_list, dim=0)
            label_list = torch.cat(label_list, dim=0)

            class_list = np.unique(label_list)
            for class_index in class_list:
                data_index = (label_list == class_index).nonzero().squeeze(-1)
                embedding = embedding_list[data_index]
                proto = embedding.mean(0)
                self.model.fc.weight.data[class_index] = proto

    def optimise_ridge_parameter(self, Features, Y):
        ridges = 10.0 ** np.arange(-8, 9)
        num_val_samples = int(Features.shape[0] * 0.8)
        losses = []
        Q_val = Features[0:num_val_samples, :].T @ Y[0:num_val_samples, :]
        G_val = Features[0:num_val_samples, :].T @ Features[0:num_val_samples, :]
        for ridge in ridges:
            Wo = torch.linalg.solve(G_val + ridge * torch.eye(G_val.size(dim=0)), Q_val).T
            Y_train_pred = Features[num_val_samples::, :] @ Wo.T
            losses.append(F.mse_loss(Y_train_pred, Y[num_val_samples::, :]))
        ridge = ridges[np.argmin(np.array(losses))]
        self._logger.info("[RanPAC] Optimal lambda: " + str(ridge))
        return ridge

    def prefix(self):
        prefix_parts = [
            str(self._config['seed']),
            self._config['dataset_name'], 
            str(self._config['dataset_num_task']),
            self._config['model_backbone'],
            self._config['train_method'],
            self._config['model_merge'],
            self._config['model_merge_base'],
            'robust_training'
        ]
        return "_".join(prefix_parts)

    def backbone_checkpoint(self, task=-1):
        return f"{CHECKPOINT_DIR}/{self.prefix()}_backbone" + (
            f"_{task}.pt" if task >= 0 else "_base.pt"
        )

    def head_checkpoint(self, task):
        return f"{CHECKPOINT_DIR}/{self.prefix()}_head_{task}.pt"


def set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Dataset configuration table - change dataset_name to switch experiments
DATASET_CONFIGS = {
    "cifar224": {
        "dataset_num_task": 10,
        "dataset_init_cls": 10,
        "dataset_increment": 10,
    },
    "imagenetr": {
        "dataset_num_task": 10,
        "dataset_init_cls": 20,
        "dataset_increment": 20,
    },
    "imageneta": {
        "dataset_num_task": 10,
        "dataset_init_cls": 20,
        "dataset_increment": 20,
    },
    "cub": {
        "dataset_num_task": 10,
        "dataset_init_cls": 20,
        "dataset_increment": 20,
    },
    "omnibenchmark": {
        "dataset_num_task": 10,
        "dataset_init_cls": 30,
        "dataset_increment": 30,
    },
    "vtab": {
        "dataset_num_task": 5,
        "dataset_init_cls": 10,
        "dataset_increment": 10,
    },
}

BASELINE_CONFIG = {
    "seed": 1993,
    "reset": True,
    
    # Dataset config
    "dataset_name": "imagenetr",
    
    # Training config
    "train_method": "seq",
    "train_epochs": 10,
    "train_batch_size": 48,
    "train_base_lr": 1e-2,
    "train_weight_decay": 5e-4,
    
    # Robustness training
    "train_robustness": True,
    "train_local_robustness_num_clusters": 5,
    "train_base_reg_weight": 1,
    "train_buffer_percent": 0.1,
    "train_buffer_size": 200,
    "train_buffer_decay": 1,
    
    # Classifier alignment
    "train_ca": True,
    "train_ca_samples_per_class": 100,
    "train_ca_lambda_old_default": 1e-4,  # Increased for tighter control (less scaling)
    "train_ca_lambda_cur_default": 1e-5,  # Increased for tighter control (less scaling)
    "train_ca_nes_lr": 0.05,
    "train_ca_nes_iterations": 100,
    "train_ca_nes_popsize": 50,
    "train_ca_nes_patience": 10,

    "train_ca_entropy_reg": True,
    
    "train_ca_nes_sigma_init": 1e-3,        # Reduced initial sigma (from 1e-3)
    "train_ca_nes_sigma_final": 1e-4,       # Reduced final sigma (from 1e-4)
    "train_ca_nes_sigma_decay": "exponential",   # "exponential", "linear", "cosine", "constant"
    "train_ca_nes_sigma_min": 1e-3,          
    "train_ca_nes_sigma_max": 2e-1,          

    "train_ca_guardrail": True,
    "train_ca_guardrail_eps": 0.20,  # 20% drop tolerance
    "train_ca_guardrail_alpha": 1.0,  # penalty weight
    
    "train_ca_penalty_type": "exponential",  # "standard", "exponential", "adaptive"
    "train_ca_age_decay": 1.2,  # Should be > 1.0 for exponential decay (older tasks get higher penalties)
    
    # Model config
    "model_backbone": "vit_base_patch16_224_lora",
    "model_classifier": ["mlp"],
    "model_lora_r": 256,
    "model_lora_alpha": 512,
    "model_lora_dropout": 0.0,
    "model_merge": "ties",
    "model_merge_base": "init",
    "model_merge_coef": 1.0,
    "model_merge_topk": 70,
    "model_use_ranpac": True,
    "model_M": 10000,
}


def run_experiment():
    """Run simplified experiment with baseline configuration"""
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    logger = setup_logger(f"exp_{timestamp}.log")
    
    set_random(BASELINE_CONFIG["seed"])
    
    # Get dataset-specific configuration
    dataset_name = BASELINE_CONFIG["dataset_name"]
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")
    
    dataset_config = DATASET_CONFIGS[dataset_name]
    
    # Update BASELINE_CONFIG with dataset-specific values
    config_with_dataset = BASELINE_CONFIG.copy()
    config_with_dataset.update({
        "dataset_num_task": dataset_config["dataset_num_task"],
        "dataset_init_cls": dataset_config["dataset_init_cls"], 
        "dataset_increment": dataset_config["dataset_increment"],
    })
    
    # Initialize data manager
    data_manager = DataManager(
        dataset_name,
        True,
        BASELINE_CONFIG["seed"],
        dataset_config["dataset_init_cls"],
        dataset_config["dataset_increment"],
        False,
    )
    
    try:
        learner = Learner(config_with_dataset, logger)
        learner.learn(data_manager)
        
        # Get final metrics
        mlp_faa = learner._mlp_faa
        mlp_asa = learner._mlp_asa
        nme_faa = learner._nme_faa
        nme_asa = learner._nme_asa
        
        logger.info(f"[Final Results]")
        logger.info(f"  MLP - FAA: {mlp_faa:.2f}, ASA: {mlp_asa:.2f}")
        logger.info(f"  NME - FAA: {nme_faa:.2f}, ASA: {nme_asa:.2f}")

        # Clean up memory
        del learner
        torch.cuda.empty_cache()
        gc.collect()
        
        return {
            "MLP_FAA": mlp_faa,
            "MLP_ASA": mlp_asa,
            "NME_FAA": nme_faa,
            "NME_ASA": nme_asa,
        }
        
    except Exception as e:
        logger.error(f"Experiment failed with error: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"error": str(e)}


if __name__ == "__main__":
    results = run_experiment()
    print(f"Experiment Results: {results}")
