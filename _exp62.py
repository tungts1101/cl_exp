import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import random
import os
from datetime import datetime
import copy
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
from _exp import (
    ContinualLearnerHead,
    setup_logger,
    get_backbone,
    merge,
    compute_metrics,
)
from generative_model import MultivariateNormalGenerator
from inc_net import CosineLinear
from util import accuracy, set_random
import gc
import time


timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
logger = setup_logger(f"nes_backbone_evolution_{timestamp}.log")


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
        self.fc_matrix = []  # For FC classifier evaluation
        self._cls_to_task_idx = {}

        self.model = Model(config)
        self.model.cuda()
        torch.save(
            self.model.get_backbone_trainable_params(), self.backbone_checkpoint()
        )

        self._faa, self._ffm = 0, 0
        
        # Initialize feature generator for NES evolution
        self.feature_generator = None
        
        # RanPAC variables
        if self._config.get("model_use_ranpac", False):
            self.W_rand = None
            self.Q = None
            self.G = None
            self.H = None
            self.Y = None

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

        # Initialize feature generator for NES if enabled
        if self._config.get("train_nes_backbone", False):
            self.feature_generator = MultivariateNormalGenerator(
                self.model.feature_dim, self._total_classnum, device="cuda"
            )
            self._logger.info("[NES Backbone] Using MultivariateNormal feature generator")
        else:
            self.feature_generator = None

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

        # Update head for new task (like in local_robustness.py)
        self.model.update_head(
            self._total_classes - self._known_classes,
            freeze_old=True,
        )

    def after_task(self):
        self._known_classes = self._total_classes

    def eval(self):
        test_set = self.data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=4)

        self.model.eval()

        # Evaluate only FC classifier (head is used only for training)
        y_true, y_pred = [], []
        with torch.no_grad():
            for _, (_, _, x, y) in enumerate(test_loader):
                x, y = x.cuda(), y.cuda()
                logits = self.model(x)  # FC classifier
                predicts = logits.argmax(dim=1)
                y_pred.append(predicts.cpu().numpy())
                y_true.append(y.cpu().numpy())

        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        acc_total, grouped = accuracy(y_pred.T, y_true, self._class_increments)
        
        self._logger.info(f"[Evaluation FC] Task {self._cur_task + 1}")
        self._logger.info(f"[Evaluation FC] Total Acc: {acc_total:.2f}, Grouped Acc: {grouped}")

        # Update FC classifier matrix and compute metrics
        self.fc_matrix.append(grouped)
        
        num_tasks = len(self.fc_matrix)
        
        # FC metrics
        fc_accuracy_matrix = np.zeros((num_tasks, num_tasks))
        for i in range(num_tasks):
            for j in range(i + 1):
                fc_accuracy_matrix[i, j] = self.fc_matrix[i][j]

        fc_faa, fc_ffm, fc_ffd, fc_asa = compute_metrics(fc_accuracy_matrix)
        self._fc_faa = fc_faa
        self._fc_asa = fc_asa
        self._fc_grouped = grouped  # Store the grouped accuracies for current task
        self._logger.info(f"[Evaluation FC] FAA: {fc_faa:.2f}, FFM: {fc_ffm:.2f}, FFD: {fc_ffd:.2f}, ASA: {fc_asa:.2f}")
        
        self._aa = self._fc_faa

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
            or self._config["reset"]
        ):
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
                        p for p in self.model.fc.parameters() if p.requires_grad
                    ],
                    "lr": base_lr,
                    "weight_decay": weight_decay,
                },
                {
                    # Only train the current task's head (freeze_old=True ensures old heads are frozen)
                    "params": [
                        p
                        for p in self.model.head.heads[self._cur_task].parameters()
                        if p.requires_grad
                    ],
                    "lr": base_lr,
                    "weight_decay": weight_decay,
                }
            ]

            optimizer = optim.SGD(
                parameters, lr=base_lr, momentum=0.9, weight_decay=weight_decay
            )

            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=1e-6
            )

            self.model.train()
            self._logger.info(f"[Training] Task {self._cur_task + 1}")
            self._logger.info(f"[Training] {self.model}")

            for epoch in range(epochs):
                total_ce_loss, total_acc, total = 0, 0, 0

                for _, (_, x_aug, x, y) in enumerate(train_loader):
                    x_aug, x, y = x_aug.cuda(), x.cuda(), y.cuda()

                    # Train both FC and MLP head
                    # FC loss
                    fc_logits = self.model(x)
                    fc_loss = F.cross_entropy(fc_logits, y)
                    
                    # MLP head loss (task-specific head)
                    features = self.model.get_features(x)
                    # Adjust labels for task-specific head (offset by known classes)
                    y_head = torch.where(
                        y - self._known_classes >= 0, y - self._known_classes, -100
                    )
                    head_logits = self.model.head.heads[-1](features)
                    head_loss = F.cross_entropy(head_logits, y_head)
                    
                    # Total loss
                    ce_loss = fc_loss + head_loss

                    optimizer.zero_grad()
                    ce_loss.backward()
                    optimizer.step()

                    total_ce_loss += ce_loss.item() * len(y)
                    total_acc += (fc_logits.argmax(dim=1) == y).sum().item()
                    total += len(y)

                scheduler.step()

                if epoch % 5 == 4 or epoch == epochs - 1:
                    self._logger.info(
                        f"[Training] Epoch {epoch + 1}/{epochs}, "
                        f"CE Loss: {total_ce_loss / total:.4f}, "
                        f"Acc: {total_acc / total:.4f}"
                    )

            torch.save(
                self.model.get_backbone_trainable_params(),
                self.backbone_checkpoint(self._cur_task),
            )
        else:
            self.model.backbone.load_state_dict(
                torch.load(self.backbone_checkpoint(self._cur_task)), strict=False
            )
            self.model.cuda()

        # Perform model merging if configured
        if self._config["model_merge"] != "none":
            self.merge()
        
        # Perform NES backbone evolution if enabled
        if self._config.get("train_nes_backbone", False):
            self.evolve_backbone_with_nes()
        
        # Fit FC classifier using prototype-based approach or RanPAC
        self.fit()

    def merge(self):
        self._logger.info(f"[Merging] Method {self._config['model_merge']}")
        base_params = torch.load(self.backbone_checkpoint(-1))
        num_merged_params = sum(param.numel() for param in base_params.values())
        self._logger.info(f"[Merging] Merging with {num_merged_params:,} total parameters")

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
        """Fit FC classifier using prototype-based approach or RanPAC"""
        trainset_CPs = self.data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="test",
        )
        train_loader_CPs = DataLoader(
            trainset_CPs, batch_size=512, shuffle=False, num_workers=4
        )
        
        if self._config.get("model_use_ranpac", False):
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
            Wo = torch.linalg.solve(
                self.G + ridge * torch.eye(self.G.size(dim=0)), self.Q
            ).T
            self.model.fc.weight.data = Wo[0 : self._total_classes, :].cuda()
            
            self._logger.info(f"[RanPAC] Fitted classifier for task {self._cur_task + 1}")
        else:
            # Standard prototype-based approach
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
                
            self._logger.info(f"[Prototype] Fitted classifier for task {self._cur_task + 1}")

    def optimise_ridge_parameter(self, Features, Y):
        ridges = 10.0 ** np.arange(-8, 9)
        num_val_samples = int(Features.shape[0] * 0.8)
        losses = []
        Q_val = Features[0:num_val_samples, :].T @ Y[0:num_val_samples, :]
        G_val = Features[0:num_val_samples, :].T @ Features[0:num_val_samples, :]
        for ridge in ridges:
            Wo = torch.linalg.solve(
                G_val + ridge * torch.eye(G_val.size(dim=0)), Q_val
            ).T
            Y_train_pred = Features[num_val_samples::, :] @ Wo.T
            losses.append(F.mse_loss(Y_train_pred, Y[num_val_samples::, :]))
        ridge = ridges[np.argmin(np.array(losses))]
        self._logger.info("[RanPAC] Optimal lambda: " + str(ridge))
        return ridge

    def evolve_backbone_with_nes(self):
        """Evolve backbone parameters using NES after merging"""
        if self.feature_generator is None:
            self.feature_generator = MultivariateNormalGenerator(
                self.model.feature_dim, self._total_classnum, device="cuda"
            )

        # Update feature generator with current task classes
        for class_idx in range(self._known_classes, self._total_classes):
            trainset_cls = self.data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
            )
            cls_loader = DataLoader(trainset_cls, batch_size=512, shuffle=False, num_workers=4)
            
            features_list = []
            labels_list = []
            self.model.eval()
            with torch.no_grad():
                for _, (_, _, x, y) in enumerate(cls_loader):
                    x, y = x.cuda(), y.cuda()
                    features = self.model.get_features(x)
                    features_list.append(features.cpu())
                    labels_list.append(y.cpu())
            
            if features_list:
                all_features = torch.cat(features_list, dim=0)
                all_labels = torch.cat(labels_list, dim=0)
                
                self.feature_generator.update(all_features, all_labels)
                self._logger.info(f"[NES Backbone] Updated feature generator for class {class_idx} with {all_features.shape[0]} samples")
            else:
                self._logger.warning(f"[NES Backbone] Class {class_idx}: no features extracted")

        if self._cur_task == 0:
            self._logger.info("[NES Backbone] Skipping evolution for first task")
            return

        self._logger.info("[NES Backbone] Starting backbone evolution with synthetic data")

        # Generate synthetic data for all classes
        sampled_data = []
        sampled_label = []
        num_sampled_pcls = self._config.get("train_nes_samples_per_class", 100)

        for c_id in range(self._total_classes):
            if self.feature_generator.can_sample(c_id):
                try:
                    feats = self.feature_generator.sample(num_sampled_pcls, c_id)
                    feats = F.layer_norm(feats, (feats.shape[-1],))
                    sampled_data.append(feats)
                    sampled_label.extend([c_id] * num_sampled_pcls)
                except Exception as e:
                    self._logger.warning(f"[NES Backbone] Failed to sample from class {c_id}: {e}")
                    continue

        if not sampled_data:
            self._logger.warning("[NES Backbone] No samples generated, skipping backbone evolution")
            return

        all_features = torch.cat(sampled_data, dim=0).float().cuda(non_blocking=True)
        all_labels = torch.tensor(sampled_label).long().cuda(non_blocking=True)
        self._logger.info(f"[NES Backbone] Generated synthetic dataset: {all_features.shape[0]} samples, {self._total_classes} classes")

        # Build initial genome from backbone parameters
        backbone_params = []
        param_shapes = []
        for name, param in self.model.backbone.named_parameters():
            if param.requires_grad:
                backbone_params.append(param.data.detach().cpu().numpy().flatten())
                param_shapes.append(param.shape)

        theta = np.concatenate(backbone_params).astype(np.float32)
        original_solution = theta.copy()

        # Macro-class accuracy helper
        @torch.no_grad()
        def macro_class_accuracy(features, labels):
            logits = self.model.fc(features)
            accs = []
            for c in range(self._total_classes):
                m = (labels == c)
                if m.any():
                    acc_c = (logits[m].argmax(1) == labels[m]).float().mean().item()
                    accs.append(acc_c)
            return float(np.mean(accs)) if accs else 0.0

        # Objective function (returns scalar loss to MINIMIZE)
        def objective_function(params_flat: np.ndarray) -> float:
            try:
                # Install candidate weights (snap + restore pattern)
                param_idx = 0
                original_params_snap = []
                for name, param in self.model.backbone.named_parameters():
                    if param.requires_grad:
                        original_params_snap.append(param.data.clone())
                        sz = param.numel()
                        new_data = params_flat[param_idx:param_idx + sz]
                        param_idx += sz
                        param.data = torch.from_numpy(new_data).float().cuda().view(param.shape)

                # Forward pass and compute macro accuracy
                with torch.no_grad():
                    features = self.model.get_features(all_features)
                    macro_acc = macro_class_accuracy(features, all_labels)

                # Compute trust region penalty (L2 regularization)
                delta = params_flat - original_solution
                trust_region_penalty = self._config.get("train_nes_lambda", 1e-4) * float(np.dot(delta, delta))

                loss = -(macro_acc) + trust_region_penalty

                # Restore original weights
                param_idx = 0
                for name, param in self.model.backbone.named_parameters():
                    if param.requires_grad:
                        param.data = original_params_snap[param_idx]
                        param_idx += 1

                return float(loss)

            except Exception as e:
                self._logger.warning(f"[NES Backbone] Error in objective function: {e}")
                return 1.0

        # NES optimization loop
        sigma = float(self._config.get("train_nes_sigma", 0.01))
        lr = float(self._config.get("train_nes_lr", 0.05))
        iters = int(self._config.get("train_nes_iterations", 50))
        pop = int(self._config.get("train_nes_popsize", 20))

        self._logger.info(f"[NES Backbone] iters={iters}, pop={pop}, sigma={sigma}, lr={lr}")

        best_theta = theta.copy()
        best_loss = objective_function(best_theta)
        no_improve_steps = 0
        patience = int(self._config.get("train_nes_patience", 10))

        for it in range(iters):
            # Antithetic noise
            eps = np.random.randn(pop, theta.size).astype(np.float32)

            # Evaluate θ ± σ * ε_i
            losses_pos = np.empty(pop, dtype=np.float32)
            losses_neg = np.empty(pop, dtype=np.float32)
            for i in range(pop):
                step = sigma * eps[i]
                losses_pos[i] = objective_function(theta + step)
                losses_neg[i] = objective_function(theta - step)

            # NES gradient
            deltaL = (losses_pos - losses_neg)[:, None]
            grad = (deltaL * eps).mean(axis=0) / (2.0 * sigma)
            theta = theta - lr * grad.astype(np.float32)

            # Track best
            cur_loss = objective_function(theta)
            if cur_loss < best_loss - 1e-6:
                best_loss = cur_loss
                best_theta = theta.copy()
                no_improve_steps = 0
            else:
                no_improve_steps += 1

            # Log progress
            if it % 10 == 0 or it == iters - 1:
                # Compute current accuracy with best parameters
                param_idx = 0
                snap = []
                for name, param in self.model.backbone.named_parameters():
                    if param.requires_grad:
                        snap.append(param.data.clone())
                        sz = param.numel()
                        new_data = best_theta[param_idx:param_idx + sz]
                        param_idx += sz
                        param.data = torch.from_numpy(new_data).float().cuda().view(param.shape)
                
                with torch.no_grad():
                    features = self.model.get_features(all_features)
                    macro_acc = macro_class_accuracy(features, all_labels)
                
                # Restore
                param_idx = 0
                for name, param in self.model.backbone.named_parameters():
                    if param.requires_grad:
                        param.data = snap[param_idx]
                        param_idx += 1
                        
                self._logger.info(f"[NES Backbone] iter {it}: best_loss={best_loss:.6f}, macro_acc={macro_acc:.4f}")

            if no_improve_steps >= patience:
                self._logger.info(f"[NES Backbone] Early stop at iter {it} (no improvement for {patience} steps).")
                break

        # Apply best solution
        param_idx = 0
        for name, param in self.model.backbone.named_parameters():
            if param.requires_grad:
                sz = param.numel()
                new_data = best_theta[param_idx:param_idx + sz]
                param_idx += sz
                param.data = torch.from_numpy(new_data).float().cuda().view(param.shape)

        with torch.no_grad():
            features = self.model.get_features(all_features)
            final_macro = macro_class_accuracy(features, all_labels)
        self._logger.info(f"[NES Backbone] Completed. Final Macro-Class Accuracy: {final_macro:.4f}")

        # Save evolved backbone
        torch.save(
            self.model.get_backbone_trainable_params(),
            self.backbone_checkpoint(f"{self._cur_task}_evolved"),
        )

    def prefix(self):
        prefix_parts = [
            str(self._config['seed']),
            self._config['dataset_name'], 
            str(self._config['dataset_num_task']),
            self._config['model_backbone'],
            self._config['train_method'],
            self._config['model_merge'],
            'nes_backbone_evolution'
        ]
        return "_".join(prefix_parts)

    def backbone_checkpoint(self, task=-1):
        if isinstance(task, str):
            return f"{CHECKPOINT_DIR}/{self.prefix()}_backbone_{task}.pt"
        return f"{CHECKPOINT_DIR}/{self.prefix()}_backbone" + (
            f"_{task}.pt" if task >= 0 else "_base.pt"
        )


def set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Dataset configuration table
data_table = {
    # "cars": [(10, 20, 20)],
    # "cifar224": [(10, 10, 10)],
    "imagenetr": [(10, 20, 20)],
    # "imageneta": [(10, 20, 20)],
    # "cub": [(10, 20, 20)],
    # "omnibenchmark": [(10, 30, 30)],
    # "vtab": [(5, 10, 10)],
}


def run_experiments():
    seeds = [1993]  # Run only one seed
    
    # Test only one configuration
    experiment_configs = [
        # Test: NES backbone evolution with standard parameters
        {"train_nes_backbone": True, "train_nes_sigma": 0.01, "train_nes_lr": 0.05, "train_nes_lambda": 1e-4},
    ]
    
    all_results = {}
    
    # Iterate through all seeds
    for seed in seeds:
        logger.info(f"{'='*120}")
        logger.info(f"STARTING SEED: {seed}")
        logger.info(f"{'='*120}")
        
        set_random(seed)
        seed_results = {}
        
        # Iterate through all datasets
        for dataset_name in data_table.keys():
            logger.info(f"{'='*100}")
            logger.info(f"SEED {seed} | DATASET: {dataset_name.upper()}")
            logger.info(f"{'='*100}")
            
            dataset_num_task = data_table[dataset_name][0][0]
            dataset_init_cls = data_table[dataset_name][0][1]
            dataset_increment = data_table[dataset_name][0][2]
            
            base_config = {
                "seed": seed,
                "reset": True,

                "dataset_name": dataset_name,
                "dataset_num_task": dataset_num_task,
                "dataset_init_cls": dataset_init_cls,
                "dataset_increment": dataset_increment,
                
                "train_method": "seq",
                "train_epochs": 10,
                "train_batch_size": 48,
                "train_optim": "sgd",
                "train_base_lr": 1e-2,
                "train_weight_decay": 5e-4,
                
                # NES backbone evolution parameters
                "train_nes_samples_per_class": 100,
                "train_nes_iterations": 50,
                "train_nes_popsize": 20,
                "train_nes_patience": 10,
                
                "model_backbone": "vit_base_patch16_224_lora",
                "model_lora_r": 128,
                "model_lora_alpha": 256,
                "model_lora_dropout": 0.0,
                "model_merge": "ties",
                "model_merge_coef": 1.0,
                "model_merge_topk": 100,
                
                # RanPAC parameters
                "model_use_ranpac": True,
                "model_M": 10000,
            }
            
            dataset_results = []
            
            for i, experiment_config in enumerate(experiment_configs):
                logger.info(f"{'-'*60}")
                logger.info(f"Seed: {seed} | Dataset: {dataset_name} | Experiment {i+1}/{len(experiment_configs)}")
                logger.info(f"Configuration: {experiment_config}")
                logger.info(f"{'-'*60}")
                
                # Merge base config with experiment-specific config
                config = {**base_config, **experiment_config}
                
                try:
                    # Create data manager
                    data_manager = DataManager(
                        config["dataset_name"],
                        True,  # shuffle
                        config["seed"],
                        config["dataset_init_cls"],
                        config["dataset_increment"],
                        False,  # validation
                    )
                    
                    learner = Learner(config, logger)
                    
                    # Run the full learning process
                    learner.learn(data_manager)
                    
                    # Get final results
                    result = {
                        'seed': seed,
                        'dataset': dataset_name,
                        'config_id': i + 1,
                        'experiment_config': experiment_config,
                        'train_nes_backbone': config.get('train_nes_backbone', False),
                        'nes_sigma': config.get('train_nes_sigma', 'N/A'),
                        'fc_faa': learner._fc_faa,
                        'fc_asa': learner._fc_asa,
                        'fc_grouped': learner._fc_grouped,
                    }
                    dataset_results.append(result)
                    
                    logger.info(f"Seed {seed} | {dataset_name} - Experiment {i+1} Results:")
                    logger.info(f"  FC FAA: {learner._fc_faa:.2f}%")
                    logger.info(f"  FC ASA: {learner._fc_asa:.2f}%")
                    logger.info(f"  FC Grouped (per-task): {[f'{acc:.2f}' for acc in learner._fc_grouped]}")
                    
                    # Clean up memory
                    del learner
                    del data_manager
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    logger.error(f"[Seed {seed} | {dataset_name} - Experiment {i+1}] Detailed Error:")
                    logger.error(f"Exception Type: {type(e).__name__}")
                    logger.error(f"Exception Message: {str(e)}")
                    logger.error(f"Full Traceback:\n{error_details}")
                    
                    result = {
                        'seed': seed,
                        'dataset': dataset_name,
                        'config_id': i + 1,
                        'experiment_config': experiment_config,
                        'train_nes_backbone': config.get('train_nes_backbone', False),
                        'nes_sigma': config.get('train_nes_sigma', 'N/A'),
                        'fc_faa': 0.0,
                        'fc_asa': 0.0,
                        'fc_grouped': [],
                        'error': str(e),
                        'error_type': type(e).__name__,
                    }
                    dataset_results.append(result)
                    
                    # Clean up memory even on error
                    torch.cuda.empty_cache()
                    gc.collect()
            
            seed_results[dataset_name] = dataset_results
            
            # Dataset-specific summary for this seed
            logger.info(f"{'='*60}")
            logger.info(f"SEED {seed} | DATASET {dataset_name.upper()} SUMMARY")
            logger.info(f"{'='*60}")
            
            for result in dataset_results:
                logger.info(f"Config {result['config_id']}: {result['experiment_config']}")
                if 'error' in result:
                    logger.info(f"  Status: FAILED - {result['error']}")
                else:
                    logger.info(f"  Status: SUCCESS")
                    logger.info(f"  FC FAA: {result['fc_faa']:.2f}%")
                    logger.info(f"  FC ASA: {result['fc_asa']:.2f}%")
                    logger.info(f"  FC Grouped: {[f'{acc:.2f}' for acc in result['fc_grouped']]}")
        
        all_results[seed] = seed_results
        
        # Seed-specific summary
        logger.info(f"{'='*100}")
        logger.info(f"SEED {seed} OVERALL SUMMARY")
        logger.info(f"{'='*100}")
        
        for dataset_name, dataset_results in seed_results.items():
            logger.info(f"{dataset_name.upper()}:")
            successful_results = [r for r in dataset_results if 'error' not in r]
            
            if successful_results:
                best_result = max(successful_results, key=lambda x: x['fc_faa'])
                logger.info(f"  Best Configuration: Config {best_result['config_id']}")
                logger.info(f"    Configuration: {best_result['experiment_config']}")
                logger.info(f"    Best FC FAA: {best_result['fc_faa']:.2f}%")
                logger.info(f"    Best FC ASA: {best_result['fc_asa']:.2f}%")
                
                # Performance comparison for this dataset
                baseline_result = next((r for r in successful_results if not r['train_nes_backbone']), None)
                if baseline_result:
                    logger.info(f"  Performance Analysis:")
                    logger.info(f"    Baseline (no NES): FAA={baseline_result['fc_faa']:.2f}%")
                    logger.info(f"      Per-task accuracies: {[f'{acc:.2f}' for acc in baseline_result['fc_grouped']]}")
                    
                    for result in successful_results:
                        if result['train_nes_backbone']:
                            improvement = result['fc_faa'] - baseline_result['fc_faa']
                            logger.info(f"    Config {result['config_id']} (σ={result['nes_sigma']}): "
                                      f"FAA={result['fc_faa']:.2f}% ({improvement:+.2f}%)")
                            logger.info(f"      Per-task accuracies: {[f'{acc:.2f}' for acc in result['fc_grouped']]}")
            else:
                logger.info(f"  No successful experiments for {dataset_name}")
    
    # Final multi-seed summary
    logger.info(f"{'='*120}")
    logger.info("MULTI-SEED EXPERIMENT SUMMARY")
    logger.info(f"{'='*120}")
    
    # Calculate average performance across seeds for each dataset and configuration
    for dataset_name in data_table.keys():
        logger.info(f"{dataset_name.upper()} - AVERAGE ACROSS SEEDS:")
        
        for config_id in [1]:  # Only one config now
            config_results = []
            for seed in seeds:
                if dataset_name in all_results[seed]:
                    seed_config_results = [r for r in all_results[seed][dataset_name] 
                                         if r['config_id'] == config_id and 'error' not in r]
                    config_results.extend(seed_config_results)
            
            if config_results:
                avg_faa = sum(r['fc_faa'] for r in config_results) / len(config_results)
                avg_asa = sum(r['fc_asa'] for r in config_results) / len(config_results)
                std_faa = (sum((r['fc_faa'] - avg_faa)**2 for r in config_results) / len(config_results))**0.5
                config_desc = config_results[0]['experiment_config']
                
                logger.info(f"  Config {config_id} {config_desc}:")
                logger.info(f"    FAA: {avg_faa:.2f}% ± {std_faa:.2f}% (n={len(config_results)})")
                logger.info(f"    ASA: {avg_asa:.2f}%")
                
                # Show individual seed results
                for seed in seeds:
                    seed_results = [r for r in config_results if r['seed'] == seed]
                    if seed_results:
                        seed_faa = seed_results[0]['fc_faa']
                        logger.info(f"      Seed {seed}: FAA={seed_faa:.2f}%")
            else:
                logger.info(f"  Config {config_id}: No successful results")
    
    return all_results, logger


# Run the experiments
if __name__ == "__main__":
    start_time = time.time()
    all_results, logger = run_experiments()
    total_time = time.time() - start_time
    
    logger.info(f"{'='*120}")
    logger.info("FINAL SUMMARY")
    logger.info(f"{'='*120}")
    logger.info(f"Total experiment time: {total_time:.2f} seconds")
    
    # Count total experiments across all seeds
    total_experiments = 0
    successful_experiments = 0
    
    for seed_results in all_results.values():
        for dataset_results in seed_results.values():
            total_experiments += len(dataset_results)
            successful_experiments += len([r for r in dataset_results if 'error' not in r])
    
    logger.info(f"Total experiments: {total_experiments}")
    logger.info(f"Successful experiments: {successful_experiments}")
    logger.info(f"Failed experiments: {total_experiments - successful_experiments}")
    
    if successful_experiments > 0:
        logger.info(f"Success rate: {successful_experiments / total_experiments * 100:.1f}%")
