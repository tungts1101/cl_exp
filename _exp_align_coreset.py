import copy
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import os
from datetime import datetime
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
from _exp import (
    ContinualLearnerHead,
    setup_logger,
    get_backbone,
    merge,
    compute_metrics,
)
from util import accuracy, set_random
import gc
import time


timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
logger = setup_logger(f"logs/{timestamp}_exp_loss_kd_coreset_clean.log")


class CoresetDataset(Dataset):
    """
    Coreset dataset that selects samples by a simple, targeted score:
      score = CE(y, p_m) + kd_lambda * KL(p_old || p_m^T)
    where:
      - p_old is the prediction from the LAST (pre-merge) backbone + current head
      - p_m   is the prediction from the MERGED backbone + current head
    Higher score = more room to improve label loss while staying close to old model.
    """

    def __init__(self, memory_percentage=0.1, total_train_size=None, kd_lambda=0.5, kd_temperature=2.0):
        """
        Initialize CE+KD coreset dataset.

        Args:
            memory_percentage (float): Percentage of training data to keep (e.g., 0.1 = 10%)
            total_train_size (int): Total number of training samples (to compute fixed budget)
            kd_lambda (float): Weight for KD loss in the selection score
            kd_temperature (float): Temperature used in KD term
        """
        self.memory_percentage = memory_percentage
        self.total_train_size = total_train_size
        self.kd_lambda = kd_lambda
        self.kd_temperature = kd_temperature

        # Calculate fixed budget based on total training size
        if total_train_size is not None:
            self.total_budget = int(total_train_size * memory_percentage)
        else:
            self.total_budget = 0  # Will be set later when total_train_size is known

        self.buffer = []  # List of (sample, label) tuples
        self._buffer_scores = torch.tensor([])  # scores aligned with buffer entries (on CPU)
        self.total_samples_seen = 0  # Total samples seen across all tasks

    def set_total_budget(self, total_train_size):
        """Set the total budget based on total training size"""
        self.total_train_size = total_train_size
        self.total_budget = int(total_train_size * self.memory_percentage)

    @torch.no_grad()
    def loss_kd_update(self, samples, labels, last_backbone_model, merged_backbone_model, head_model):
        """
        Update coreset using CE + KD priority:
          score = CE(y, p_m) + kd_lambda * KL(p_old || p_m^T)

        Args:
            samples (torch.Tensor): (batch_size, C, H, W)
            labels (torch.Tensor): (batch_size,)
            last_backbone_model (nn.Module): Backbone BEFORE merging (old model)
            merged_backbone_model (nn.Module): Backbone AFTER merging
            head_model (nn.Module): Current multi-head classifier; we use it for logits
        """
        # Normalize input to CPU tensors for buffer storage; compute on GPU as needed
        if isinstance(samples, torch.Tensor):
            samples = samples.detach().cpu()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu()

        # If we have more samples than needed to fill initially, we will:
        #  - Fill to capacity with the first part
        #  - THEN process the remainder with replacement logic (no early-return bias)
        remaining_samples = samples
        remaining_labels = labels

        # Fast-path fill if buffer not yet full
        if len(self.buffer) < self.total_budget:
            remaining_budget = self.total_budget - len(self.buffer)
            num_to_add = min(remaining_budget, len(samples))

            # Add directly (scores will be computed right after for these)
            for i in range(num_to_add):
                self.buffer.append((samples[i], labels[i]))
                self.total_samples_seen += 1

            # Remainder of the batch to process with scoring/replacement
            if num_to_add < len(samples):
                remaining_samples = samples[num_to_add:]
                remaining_labels = labels[num_to_add:]
            else:
                remaining_samples = torch.empty((0, *samples.shape[1:]), dtype=samples.dtype)
                remaining_labels = torch.empty((0,), dtype=labels.dtype)

        # Compute/update scores for all buffer entries ONCE before considering replacements
        # (We only need updated scores if we just appended new entries without scores yet.)
        device = next(merged_backbone_model.parameters()).device
        last_backbone_model.eval()
        merged_backbone_model.eval()
        head_model.eval()

        # Helper to compute CE+KD scores for a batch (returns score tensor on CPU)
        def compute_scores(x_cpu, y_cpu):
            if x_cpu.numel() == 0:
                return torch.tensor([], dtype=torch.float32)

            x = x_cpu.to(device, non_blocking=True)
            y = y_cpu.to(device, non_blocking=True)

            # Old model predictions (pre-merge backbone + current head)
            old_feats = last_backbone_model(x)
            old_logits = head_model(old_feats)["logits"]
            p_old = F.softmax(old_logits, dim=1)

            # Merged model predictions (merged backbone + current head)
            merged_feats = merged_backbone_model(x)
            merged_logits = head_model(merged_feats)["logits"]
            # CE(y, p_m)
            ce = F.cross_entropy(merged_logits, y, reduction="none")
            # KD: KL(p_old || p_m^T)
            T = self.kd_temperature
            kd = F.kl_div(F.log_softmax(merged_logits / T, dim=1), p_old, reduction="none").sum(dim=1)
            # NOTE: Standard KD often multiplies by T^2; for selection scoring that scaling is not essential,
            # but you can uncomment the next line if you prefer:
            # kd = (T * T) * kd

            score = ce + self.kd_lambda * kd
            return score.detach().cpu()

        # If we appended without scores, recompute scores for ENTIRE buffer to keep things consistent
        # (buffer size is at most total_budget, which is fixed and manageable)
        if len(self.buffer) > 0:
            buf_samples = torch.stack([s for s, _ in self.buffer])
            buf_labels = torch.stack([l for _, l in self.buffer])
            self._buffer_scores = compute_scores(buf_samples, buf_labels)
        else:
            self._buffer_scores = torch.tensor([], dtype=torch.float32)

        # Now process any remaining samples with replacement logic based on scores
        batch_size = 64
        for i in range(0, len(remaining_samples), batch_size):
            batch_x_cpu = remaining_samples[i:i + batch_size]
            batch_y_cpu = remaining_labels[i:i + batch_size]

            batch_scores = compute_scores(batch_x_cpu, batch_y_cpu)

            # For each candidate in the batch, attempt to add/replace by highest-score policy
            for j, (x_cpu, y_cpu, sc) in enumerate(zip(batch_x_cpu, batch_y_cpu, batch_scores)):
                self.total_samples_seen += 1

                if len(self.buffer) < self.total_budget:
                    # Shouldn't typically occur here (handled above), but keep safe
                    self.buffer.append((x_cpu, y_cpu))
                    if self._buffer_scores.numel() == 0:
                        self._buffer_scores = sc.unsqueeze(0)
                    else:
                        self._buffer_scores = torch.cat([self._buffer_scores, sc.unsqueeze(0)], dim=0)
                else:
                    # Replace the current minimum-score element if this one has higher score
                    min_idx = int(self._buffer_scores.argmin().item())
                    if sc.item() > self._buffer_scores[min_idx].item():
                        self.buffer[min_idx] = (x_cpu, y_cpu)
                        self._buffer_scores[min_idx] = sc

        # Ensure CPU frees large temporaries
        torch.cuda.empty_cache()

    def __len__(self):
        """Return total number of samples in coreset"""
        return len(self.buffer)

    def __getitem__(self, idx):
        """Get sample by index"""
        if idx >= len(self.buffer):
            raise IndexError(f"Index {idx} out of range for coreset of size {len(self.buffer)}")
        return self.buffer[idx]

    def clear(self):
        """Clear all stored samples"""
        self.buffer.clear()
        self._buffer_scores = torch.tensor([])

    def clear_class(self, class_id):
        """Clear samples for a specific class"""
        new_buffer = []
        new_scores = []
        for (sample, label), sc in zip(self.buffer, self._buffer_scores):
            if int(label.item()) != class_id:
                new_buffer.append((sample, label))
                new_scores.append(sc)
        self.buffer = new_buffer
        self._buffer_scores = torch.stack(new_scores) if len(new_scores) > 0 else torch.tensor([])

    def __repr__(self):
        """Print informative representation of the coreset"""
        if not self.buffer:
            budget_info = f"budget={self.total_budget}" if self.total_budget > 0 else "budget=not_set"
            return "CoresetDataset(empty, memory_percentage={:.1%}, {}, strategy=loss_kd)".format(
                self.memory_percentage, budget_info)

        total_samples = len(self.buffer)
        budget_utilization = f"{total_samples}/{self.total_budget}" if self.total_budget > 0 else f"{total_samples}/unknown"

        # Count samples per class
        class_counts = {}
        for sample, label in self.buffer:
            class_id = int(label.item())
            class_counts[class_id] = class_counts.get(class_id, 0) + 1

        num_classes = len(class_counts)

        # Format class distribution in a single line
        class_dist_str = "{" + ", ".join([f"{class_id}: {class_counts[class_id]}" for class_id in sorted(class_counts.keys())]) + "}"

        info_lines = [
            f"CoresetDataset(",
            f"  total_samples={total_samples},",
            f"  budget_utilization={budget_utilization},",
            f"  num_classes={num_classes},",
            f"  memory_percentage={self.memory_percentage:.1%},",
            f"  total_train_size={self.total_train_size},",
            f"  total_samples_seen={self.total_samples_seen},",
            f"  selection_strategy='loss_kd',",
            f"  kd_lambda={self.kd_lambda}, kd_temperature={self.kd_temperature},",
            f"  class_distribution={class_dist_str}",
            f")"
        ]

        return "\n".join(info_lines)


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self.backbone = get_backbone(config)
        self.head = None

    @property
    def feature_dim(self):
        return self.backbone.num_features

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
        y = self.head(f)
        return y

    def __repr__(self):
        trainable_params = count_parameters(self, trainable=True)
        total_params = count_parameters(self)
        return f"Model(trainable_params={trainable_params}, total_params={total_params}, percentage={trainable_params * 100 / total_params:.2f})"


CHECKPOINT_DIR = "/media/ellen/HardDisk/cl/logs/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


class Learner:
    def __init__(self, config):
        self._config = config
        self._known_classes = 0
        self._total_classes = 0
        self._class_increments = []
        self._cur_task = -1
        self.mlp_matrix = []  # For MLP classifier evaluation
        self._cls_to_task_idx = {}

        self.model = Model(config)
        self.model.cuda()
        torch.save(
            self.model.get_backbone_trainable_params(), self.backbone_checkpoint()
        )

        self._faa, self._ffm = 0, 0

    def learn(self, data_manager):
        baseline = {
            "cifar224": 80,
            "imagenetr": 78,
            "imageneta": 59,
            "cub": 85,
            "omnibenchmark": 73,
            "vtab": 87
        }
        self._aa = 0

        self.data_manager = data_manager

        self.coreset = CoresetDataset(
            memory_percentage=self._config.get("coreset_memory_percentage", 0.1),
            total_train_size=data_manager.total_train_size,
            kd_lambda=self._config.get("kd_lambda", 0.5),
            kd_temperature=self._config.get("kd_temperature", 2.0)
        )
        logger.info(f"[Coreset] Initialized with memory_percentage={self._config.get('coreset_memory_percentage', 0.1):.1%}, "
                    f"strategy='loss_kd', kd_lambda={self._config.get('kd_lambda', 0.5)}, T={self._config.get('kd_temperature', 2.0)}")

        num_tasks = data_manager.nb_tasks
        self._total_classnum = data_manager.get_total_classnum()

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

    def after_task(self):
        self._known_classes = self._total_classes

    def eval(self):
        test_set = self.data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=4)

        self.model.eval()

        # Evaluate MLP classifier only
        y_true, y_pred = [], []
        with torch.no_grad():
            for _, (_, _, x, y) in enumerate(test_loader):
                x, y = x.cuda(), y.cuda()

                logits = self.model(x)["logits"]

                predicts = logits.argmax(dim=1)
                y_pred.append(predicts.cpu().numpy())
                y_true.append(y.cpu().numpy())

        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        acc_total, grouped = accuracy(y_pred.T, y_true, self._class_increments)

        logger.info(f"[Evaluation MLP] Task {self._cur_task + 1}")
        logger.info(f"[Evaluation MLP] Total Acc: {acc_total:.2f}, Grouped Acc: {grouped}")

        self.mlp_matrix.append(grouped)
        num_tasks = len(self.mlp_matrix)
        mlp_accuracy_matrix = np.zeros((num_tasks, num_tasks))
        for i in range(num_tasks):
            for j in range(i + 1):
                mlp_accuracy_matrix[i, j] = self.mlp_matrix[i][j]

        mlp_faa, mlp_ffm, mlp_ffd, mlp_asa = compute_metrics(mlp_accuracy_matrix)
        self._mlp_faa = mlp_faa
        self._mlp_asa = mlp_asa
        logger.info(f"[Evaluation MLP] FAA: {mlp_faa:.2f}, FFM: {mlp_ffm:.2f}, FFD: {mlp_ffd:.2f}, ASA: {mlp_asa:.2f}")

        self._aa = mlp_faa

    def merge(self):
        logger.info(f"[Merging] Method {self._config['model_merge']}")
        base_params = torch.load(self.backbone_checkpoint(-1))
        num_merged_params = sum(param.numel() for param in base_params.values())
        logger.info(f"[Merging] Merging with {num_merged_params:,} total parameters")

        task_params = [torch.load(self.backbone_checkpoint(task)) for task in range(self._cur_task + 1)]
        backbone_params = merge(
            base_params,
            task_params,
            method=self._config["model_merge"],
            lamb=self._config["model_merge_coef"],
            topk=self._config["model_merge_topk"],
        )
        self.model.backbone.load_state_dict(backbone_params, strict=False)

    def train_standard(self, train_loader, optimizer, scheduler, epochs):
        """Standard training without any augmentation"""
        self.model.train()

        for epoch in range(epochs):
            total_loss, total_acc, total = 0, 0, 0

            for _, (_, x_aug, x, y) in enumerate(train_loader):
                x_aug, x, y = x_aug.cuda(), x.cuda(), y.cuda()

                features = self.model.get_features(x)
                logits = self.model.head(features)["logits"]
                loss = F.cross_entropy(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(y)
                total_acc += (logits.argmax(dim=1) == y).sum().item()
                total += len(y)

            scheduler.step()

            if epoch % 5 == 4 or epoch == epochs - 1:
                logger.info(
                    f"[Training] Epoch {epoch + 1}/{epochs}, "
                    f"Loss: {total_loss / total:.4f}, "
                    f"Acc: {total_acc / total:.4f}"
                )

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
            freeze_old = False  # Always use standard training with unfrozen heads

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
            if not freeze_old:
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

            if self._config.get("train_optim", "sgd") == "adamw":
                optimizer = optim.AdamW(parameters, lr=base_lr, weight_decay=weight_decay)
            elif self._config.get("train_optim", "sgd") == "sgd":
                optimizer = optim.SGD(
                    parameters, lr=base_lr, momentum=0.9, weight_decay=weight_decay
                )
            else:
                raise ValueError(f"Unknown optimizer")

            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=1e-6
            )

            logger.info(f"[Training] Task {self._cur_task + 1}")
            logger.info(f"[Training] {self.model}")

            # Standard training
            self.train_standard(train_loader, optimizer, scheduler, epochs)

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
            self.model.update_head(
                self._total_classes - self._known_classes,
                freeze_old=True,
            )
            self.model.head.heads[-1].load_state_dict(
                torch.load(self.head_checkpoint(self._cur_task)), strict=True
            )
            self.model.cuda()

        # Store backbone state before merging for CE+KD calculation
        backbone_before_merge = None
        if self._cur_task > 0 and self._config["model_merge"] != "none":
            backbone_before_merge = copy.deepcopy(self.model.backbone)
            backbone_before_merge.eval()

        if self._config["model_merge"] != "none":
            self.merge()

        # Update coreset using CE+KD priority (for t>0 with old backbone available)
        if self._cur_task > 0 and backbone_before_merge is not None:
            self.update_coreset_with_loss_kd(backbone_before_merge)
        else:
            # For first task, use simple addition since KD to old isn't defined
            self.update_coreset_simple()

        # Clean up
        if backbone_before_merge is not None:
            del backbone_before_merge
            torch.cuda.empty_cache()

        # Perform classifier alignment using coreset
        if self._config.get("use_coreset_alignment", True) and self._cur_task > 0:
            self.align_classifier_with_coreset()

    def update_coreset_simple(self):
        """Simple coreset update for first task (no old model available)"""
        logger.info(f"[Coreset] Simple update for task {self._cur_task + 1} (first task)")

        # Get training data for current task
        trainset = self.data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="test",  # Use test mode to get clean samples without augmentation
        )
        train_loader = DataLoader(trainset, batch_size=256, shuffle=False, num_workers=4)

        # Collect all samples from current task
        all_samples = []
        all_labels = []

        with torch.no_grad():
            for _, (_, _, x, y) in enumerate(train_loader):
                all_samples.append(x.cpu())
                all_labels.append(y.cpu())

        # Concatenate all samples and labels
        all_samples = torch.cat(all_samples, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Add samples directly to coreset (simple addition for first task)
        for i in range(len(all_samples)):
            if len(self.coreset) < self.coreset.total_budget:
                self.coreset.buffer.append((all_samples[i], all_labels[i]))
                self.coreset.total_samples_seen += 1

        logger.info(f"[Coreset] Updated coreset for first task:")
        logger.info(repr(self.coreset))

    def update_coreset_with_loss_kd(self, backbone_before_merge):
        """Update coreset with samples from current task using CE+KD priority"""
        logger.info(f"[Coreset] Updating coreset with loss+KD for task {self._cur_task + 1}")

        # Get training data for current task
        trainset = self.data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="test",  # Use test mode to get clean samples without augmentation
        )
        train_loader = DataLoader(trainset, batch_size=256, shuffle=False, num_workers=4)

        # Process all samples from current task for score calculation
        for _, (_, _, x, y) in enumerate(train_loader):
            x, y = x.cuda(), y.cuda()

            # Update coreset using CE + KD between old backbone and merged backbone
            self.coreset.loss_kd_update(
                x.cpu(), y.cpu(),
                backbone_before_merge,
                self.model.backbone,
                self.model.head
            )

        logger.info(f"[Coreset] Updated coreset with loss+KD:")
        logger.info(repr(self.coreset))

    def align_classifier_with_coreset(self):
        """Align classifier using coreset samples (CE only on selected buffer)"""
        if len(self.coreset) == 0:
            logger.warning("[Coreset Alignment] No samples in coreset, skipping alignment")
            return

        logger.info(f"[Coreset Alignment] Starting classifier alignment using coreset")

        # Configuration for alignment training
        align_epochs = self._config.get("align_epochs", 10)
        align_lr = self._config.get("align_lr", 1e-3)
        align_batch_size = self._config.get("align_batch_size", 64)

        # Create DataLoader from coreset
        coreset_loader = DataLoader(
            self.coreset,
            batch_size=align_batch_size,
            shuffle=True,
            num_workers=0  # Use 0 for custom dataset
        )

        for p in self.model.head.parameters():
            p.requires_grad = True

        head_params = [p for p in self.model.head.parameters() if p.requires_grad]
        total_head_params = sum(p.numel() for p in head_params)
        logger.info(f"[Alignment] Training {total_head_params:,} head parameters ({len(head_params)} tensors)")
        optimizer = optim.AdamW(head_params, lr=align_lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=align_epochs)

        self.model.train()

        for epoch in range(align_epochs):
            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            for batch_samples, batch_labels in coreset_loader:
                batch_samples = batch_samples.cuda()
                batch_labels = batch_labels.cuda()

                batch_features = self.model.get_features(batch_samples)
                logits = self.model.head(batch_features)["logits"]

                logits_truncated = logits[:, :self._total_classes]

                loss = F.cross_entropy(logits_truncated, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(batch_labels)
                total_correct += (logits_truncated.argmax(dim=1) == batch_labels).sum().item()
                total_samples += len(batch_labels)

            scheduler.step()

            avg_loss = total_loss / total_samples
            avg_acc = total_correct / total_samples

            if (epoch + 1) % 5 == 0 or epoch == align_epochs - 1:
                logger.info(
                    f"[Coreset Alignment] Epoch {epoch + 1}/{align_epochs}, "
                    f"Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}"
                )

        logger.info(f"[Coreset Alignment] Completed alignment for task {self._cur_task + 1}")

    def prefix(self):
        prefix_parts = [
            str(self._config['seed']),
            self._config['dataset_name'],
            str(self._config['dataset_num_task']),
            self._config['model_backbone'],
            self._config['train_method'],
            self._config['model_merge'],
            'coreset',
            'loss_kd_clean'
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


def run_experiment():
    """Run experiment with CE+KD coreset (clean version)"""

    model_backbone = "vit_base_patch16_224_lora"
    seed = 1993
    dataset_name = "imagenetr"

    set_random(seed)

    # Complete configuration
    config = {
        # Basic config
        "seed": seed,
        "reset": True,

        # Dataset configuration
        "dataset_name": dataset_name,
        "dataset_num_task": 10,
        "dataset_init_cls": 20,
        "dataset_increment": 20,

        # Training configuration
        "train_method": "seq",
        "train_epochs": 10,
        "train_batch_size": 48,
        "train_base_lr": 1e-2,
        "train_weight_decay": 5e-4,
        "train_optim": "sgd",  # Optimizer: "sgd" or "adamw"

        # Model configuration
        "model_backbone": model_backbone,
        "model_lora_r": 512,
        "model_lora_alpha": 1024,
        "model_lora_dropout": 0.0,
        "model_merge": "ties",
        "model_merge_coef": 1.0,
        "model_merge_topk": 100,

        # Coreset configuration
        "use_coreset_alignment": True,
        "coreset_memory_percentage": 0.1,  # 10% of training data
        "align_epochs": 5,
        "align_lr": 1e-3,
        "align_batch_size": 64,

        # NEW: KD selection knobs
        "kd_lambda": 0.5,
        "kd_temperature": 2.0,
    }

    data_manager = DataManager(
        config["dataset_name"],
        True,
        config["seed"],
        config["dataset_init_cls"],
        config["dataset_increment"],
        False,
    )

    try:
        logger.info(f"[Experiment] Starting experiment with CE+KD coreset (clean version)")
        learner = Learner(config)
        learner.learn(data_manager)

        mlp_faa = learner._mlp_faa
        mlp_asa = learner._mlp_asa

        result = {
            "strategy": "loss_kd_clean",
            "MLP_FAA": mlp_faa,
            "MLP_ASA": mlp_asa,
        }

        logger.info(f"[Experiment Results] CE+KD Coreset (Clean)")
        logger.info(f"  MLP - FAA: {mlp_faa:.2f}, ASA: {mlp_asa:.2f}")

        # Clean up memory
        del learner
        torch.cuda.empty_cache()
        gc.collect()

        return result

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"[Experiment] Detailed Error:")
        logger.error(f"Exception Type: {type(e).__name__}")
        logger.error(f"Exception Message: {str(e)}")
        logger.error(f"Full Traceback:\n{error_details}")

        return {
            "strategy": "loss_kd_clean",
            "MLP_FAA": 0.0,
            "MLP_ASA": 0.0,
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": error_details,
        }


# Run the experiment
if __name__ == "__main__":
    start_time = time.time()
    result = run_experiment()
    total_time = time.time() - start_time

    if "error" not in result:
        logger.info(f"✓ Experiment completed successfully in {total_time:.2f}s")
        logger.info(f"  MLP - FAA: {result['MLP_FAA']:.2f}, ASA: {result['MLP_ASA']:.2f}")
    else:
        logger.error(f"✗ Experiment failed: {result['error']}")

    logger.info(f"Total experiment time: {total_time:.2f} seconds")
    logger.info(f"Generated log file: logs/{timestamp}_exp_loss_kd_coreset_clean.log")
