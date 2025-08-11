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
    RandomReplayBuffer,
    weight_init
)
from generative_model import MultivariateNormalGenerator, VAEFeatureGenerator
from inc_net import CosineLinear
from util import accuracy, set_random
import gc
import time


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
    def __init__(self, config, logger):
        self._config = config
        self._logger = logger
        self._known_classes = 0
        self._total_classes = 0
        self._class_increments = []
        self._cur_task = -1
        self.mlp_matrix = []  # For MLP classifier evaluation
        self.nme_matrix = []  # For Cosine classifier evaluation
        self._cls_to_task_idx = {}

        self.model = Model(config)
        self.model.cuda()
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

        if self._config["train_ca"]:
            if self._config["train_ca_method"] == "mul_norm":
                self.feature_generator = MultivariateNormalGenerator(
                    self.model.feature_dim, self._total_classnum, device="cuda"
                )
                self._logger.info("[Alignment] Using MultivariateNormal feature generator")
            elif self._config["train_ca_method"] == "vae":
                self.feature_generator = None
                self._logger.info("[Alignment] Using VAE feature generator (will be initialized during training)")
            else:
                self._logger.warning("[Alignment] Using legacy method")

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

        # Evaluate both MLP and Cosine classifiers
        results = {}
        if self._config["model_ensemble"]:
            self._logger.info("[Evaluation] Using model ensemble")
        
        for fc_type in self._config["model_classifier"]:
            y_true, y_pred = [], []
            with torch.no_grad():
                for _, (_, _, x, y) in enumerate(test_loader):
                    x, y = x.cuda(), y.cuda()

                    if fc_type == "nme":
                        logits = self.model(x)["logits"]  # nme classifier through fc layer
                    elif fc_type == "mlp":
                        features = self.model.get_features(x)
                        mlp_logits = self.model.head(features)["logits"]  # MLP classifier through head
                        nme_logits = self.model.fc(features)["logits"]
                        
                        mlp_probs = F.softmax(mlp_logits, dim=1)
                        nme_probs = F.softmax(nme_logits, dim=1)
                        
                        max_probs = torch.maximum(mlp_probs, nme_probs)
                        logits = torch.log(max_probs + 1e-8)  # Add small epsilon to avoid log(0)
                    
                    if self._config["model_ensemble"]:
                        # Save current backbone state by copying the actual parameter values
                        current_backbone_state = {}
                        for name, param in self.model.backbone.named_parameters():
                            if param.requires_grad:
                                current_backbone_state[name] = param.data.clone()
                        
                        # Load first task backbone for ensemble
                        self.model.backbone.load_state_dict(
                            torch.load(self.backbone_checkpoint(0)), strict=False
                        )
                        features_first = self.model.get_features(x)
                        if fc_type == "nme":
                            first_task_logits = self.model.fc(features_first)["logits"]  # Use features, not x directly
                        elif fc_type == "mlp":
                            first_task_logits = self.model.head(features_first)["logits"]
                        
                        logits += first_task_logits
                        
                        # Restore current backbone state
                        for name, param in self.model.backbone.named_parameters():
                            if param.requires_grad and name in current_backbone_state:
                                param.data.copy_(current_backbone_state[name])

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
            self._logger.info("[Evaluation MLP] Not evaluated, MLP classifier not used")
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
            self._logger.info("[Evaluation NME] Not evaluated, NME classifier not used")
            self._nme_faa = 0.0
            self._nme_asa = 0.0
        
        self._aa = max(self._mlp_faa, self._nme_faa)

    def merge(self):
        self._logger.info(f"[Merging] Method {self._config['model_merge']}")
        if self._config["model_merge_base"] == "init":
            base_params = torch.load(self.backbone_checkpoint(-1))
        elif self._config["model_merge_base"] == "first":
            base_params = torch.load(self.backbone_checkpoint(0))
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

            if self._config["train_optim"] == "adamw":
                optimizer = optim.AdamW(parameters, lr=base_lr, weight_decay=weight_decay)
            elif self._config["train_optim"] == "sgd":
                optimizer = optim.SGD(
                    parameters, lr=base_lr, momentum=0.9, weight_decay=weight_decay
                )
            else:
                raise ValueError(f"Unknown optimizer: {self._config['train_optim']}")

            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=1e-6
            )

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
                        y = torch.where(
                            y - self._known_classes >= 0, y - self._known_classes, -100
                        )
                        logits = self.model.head.heads[-1](features)
                    else:
                        logits = self.model.head(features)["logits"]

                    ce_loss = F.cross_entropy(logits, y)

                    # calculate local robustness loss ===
                    rb_loss = torch.tensor(0.0, device=x.device)

                    if self._config["train_robustness"] and self.buffer.size > 0:
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

                            # cluster_size = cluster_mask.sum()
                            # cluster_logits = logits[cluster_mask]
                            # center_logits = self.model.head(center_features)["logits"]
                            # rb_loss += ((1 / cluster_size) * F.mse_loss(cluster_logits, center_logits)) / total_samples

                            cluster_features = F.normalize(cluster_features, dim=1)
                            center_features = F.normalize(center_features, dim=1)

                            pos_sim = (cluster_features * center_features).sum(dim=1)
                            rb_loss += ( cluster_mask.sum().float() / total_samples ) * (1.0 - pos_sim.mean())

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
            self._logger.info(f"[Replay Buffer] Size by class: {self.buffer.size_by_class}")

        if self._config["train_ca"]:
            if self._config["train_ca_method"] == "vae":
                self.train_generative_model()
            self.align_classifier()

        if "nme" in self._config["model_classifier"]:
            self.fit()

        if self.buffer is not None:
            self._logger.info(f"[Replay Buffer] Updating weights")
            self.buffer.update_weights()

    def train_generative_model(self):
        trainset = self.data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="test",
        )
        train_loader = DataLoader(
            trainset,
            batch_size=self._config["train_batch_size"],
            shuffle=True,
            num_workers=4,
        )

        # ──────────────────────────────────────────────────────────────────────────────
        def vae_loss(recon, x, mu, logvar, beta=1.0):
            recon_loss = F.mse_loss(recon, x, reduction="mean")
            # recon_loss = F.smooth_l1_loss(recon, x, reduction="mean", beta=1.0)
            kl         = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            return recon_loss + beta * kl, recon_loss, kl
        
        @torch.no_grad()
        def estimate_stats(backbone: nn.Module, loader: DataLoader, device: torch.device):
            backbone.eval()
            s1, s2, n = 0.0, 0.0, 0
            for _, _, imgs, _ in loader:
                f = backbone(imgs.to(device))
                n += f.size(0)
                s1 += f.sum(0)
                s2 += (f ** 2).sum(0)
            mean = s1 / n
            std  = (s2 / n - mean ** 2).sqrt()
            return mean, std.clamp_min_(1e-6)
        
        def freeze_model(model):
            for param in model.parameters():
                param.requires_grad = False
            return model
        
        def ema_update(target_model, source_model, beta):
            for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
                target_param.data = beta * target_param.data + (1.0 - beta) * source_param.data
        # ──────────────────────────────────────────────────────────────────────────────
        

        if self._cur_task == 0:
            generative_model_config = {
                "generative_latent_dim": 256,
                "generative_hidden_dim": 512,
                "generative_feature_dim": self.model.feature_dim,
                "generative_num_classes": self._total_classnum,
            }
            self._logger.info(f"[Generative Model] Initializing with config: {generative_model_config}")
            from generative_model import ConditionalVAE
            g = ConditionalVAE(generative_model_config)
            self._g = g.cuda()
            self._g.apply(weight_init)
        else:
            g_old = copy.deepcopy(self._g)
            freeze_model(g_old)
            g = self._g
  
        self.model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        feat_mean, feat_std = estimate_stats(self.model.backbone, train_loader, device)

        train_generative_epochs = 50
        optimizer_V = torch.optim.AdamW(g.parameters(), lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-5)
        scheduler   = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer_V, T_0=5, T_mult=2, eta_min=1e-5)

        cycle_steps = 10 * len(train_loader)      # β saw-tooth every 10 epochs
        global_step = 0

        for epoch in range(train_generative_epochs):
            log_loss = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "aug": 0.0}

            g.train()
            for i, batch in enumerate(train_loader):
                global_step += 1
                phase = (global_step % cycle_steps) / cycle_steps
                beta  = min(1.0, phase)                # 0→1 cyclical warm-up

                _, _, imgs, targets = batch
                imgs, targets = imgs.cuda(), targets.cuda()

                with torch.no_grad():
                    real_feat = self.model.get_features(imgs)  # (B, feat_dim)
                    real_feat = (real_feat - feat_mean) / feat_std       # fixed norm

                y1h   = F.one_hot(targets, num_classes=self._total_classnum).float()
                recon, mu, logvar = g(real_feat, y1h)
                loss, rec_l, kl_l = vae_loss(recon, real_feat, mu, logvar, beta=beta)

                # # (optional) alignment w/ previous generator
                # if self._cur_task > 0:
                #     B = 256
                #     z      = torch.randn(B, generative_latent_dim, device=self.device)
                #     prev_y = torch.randint(0, self._known_classes, (B,), device=self.device)
                #     prev_y1h = F.one_hot(prev_y, self._total_classnum).float()
                #     with torch.no_grad():
                #         pre_feat = g_old.dec(z, prev_y1h)
                #     cur_feat = g.dec(z, prev_y1h)
                #     loss_aug = F.mse_loss(cur_feat, pre_feat)
                #     loss += 10.0 * loss_aug

                optimizer_V.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(g.parameters(), 5.0)
                optimizer_V.step()
                scheduler.step(epoch + i / len(train_loader))

                log_loss["loss"] += loss.item()
                log_loss["recon"] += rec_l.item()
                log_loss["kl"] += kl_l.item()
                # log_loss["aug"] += loss_aug.item() if self._cur_task > 0 else 0.0

            for key in log_loss:
                log_loss[key] /= len(train_loader)
            
            self._logger.info(
                f"[Generative Model] Epoch [{epoch + 1}/{train_generative_epochs}], "
                f"Recon Loss: {log_loss['recon']:.4f}, "
                f"KL Loss: {log_loss['kl']:.4f}, "
                f"Aug Loss: {log_loss['aug']:.4f}, "
                f"Total Loss: {log_loss['loss']:.4f}"
            )
        
        if self._cur_task > 0:
            self._logger.info("[Generative Model] Aligning with previous generator")
            ema_update(g, g_old, beta=0.5)
            del g_old
            gc.collect()

        # torch.save(g.state_dict(), "checkpoints/vae.pth")
        
        # Initialize VAE feature generator after training
        if self.feature_generator is None:
            self.feature_generator = VAEFeatureGenerator(
                g, self.model.feature_dim, self._total_classnum, device="cuda"
            )
            self._logger.info("[Alignment] VAE feature generator initialized")

    def align_classifier(self):
        """Align classifier using the selected feature generator method"""
        if self._config["train_ca_method"] == "mul_norm":
            self._align_classifier_with_generator()
        elif self._config["train_ca_method"] == "vae":
            self._classifier_alignment_with_vae()
        else:
            self._align_classifier_legacy()
    
    def _align_classifier_with_generator(self):
        self._logger.info("[Alignment] Using feature generator for classifier alignment")
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
                    features_list.append(features)
                    labels_list.append(y)
            
            if features_list:
                all_features = torch.cat(features_list, dim=0)
                all_labels = torch.cat(labels_list, dim=0)
                
                self.feature_generator.update(all_features, all_labels)
                # self._logger.info(f"[Alignment] Updated feature generator for class {class_idx} with {all_features.shape[0]} samples")
        
        self._run_classifier_alignment_with_generator()
    
    def _classifier_alignment_with_vae(self):
        self._logger.info("[Alignment] Using VAE for classifier alignment")
        
        # Update VAE feature generator (track which classes have been seen)
        for class_idx in range(self._known_classes, self._total_classes):
            # Create dummy data to mark classes as seen
            dummy_features = torch.zeros(1, self.model.feature_dim, device="cuda")
            dummy_labels = torch.tensor([class_idx], device="cuda")
            self.feature_generator.update(dummy_features, dummy_labels)
        
        # Perform classifier alignment using VAE
        self._run_classifier_alignment_with_generator()
    
    def _run_classifier_alignment_with_generator(self):
        if self._config["reset"] or not os.path.exists(self.head_ca_checkpoint(self._cur_task)):
            if self._cur_task == 0:
                torch.save(self.model.head.state_dict(), self.head_ca_checkpoint(self._cur_task))
                return
            self._logger.info("[Alignment] Starting classifier alignment training")
            
            run_epochs = self._config.get("train_ca_epochs", 10)
            crct_num = self._total_classes
            task_sizes = [self._total_classes - self._known_classes] if self._cur_task == 0 else self.data_manager.get_task_sizes()
            
            # for p in self.model.head.parameters():
            #     p.requires_grad = True
            param_list = [p for p in self.model.head.parameters() if p.requires_grad]
            total_head_params = sum(p.numel() for p in param_list)
            self._logger.info(f"[Alignment] Training {total_head_params:,} head parameters ({len(param_list)} tensors)")
            optimizer = optim.SGD(param_list, lr=self._config.get("train_ca_lr", 1e-2), momentum=0.9, 
                                weight_decay=self._config.get("train_ca_weight_decay", 5e-4))
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)

            self.model.train()
            
            for epoch in range(run_epochs):
                losses = 0.
                
                sampled_data = []
                sampled_label = []
                num_sampled_pcls = 256
                
                # Sample from feature generator for each class
                for c_id in range(crct_num):
                    if self.feature_generator.can_sample(c_id):
                        try:
                            # Apply task-dependent decay similar to SLCA
                            t_id = c_id // (self._total_classes // (self._cur_task + 1)) if self._cur_task > 0 else 0
                            decay = (t_id + 1) / (self._cur_task + 1) * 0.1
                            
                            # Sample features from the generator
                            sampled_features = self.feature_generator.sample(num_sampled_pcls, c_id)
                            
                            # Apply decay factor to the features (similar to the original implementation)
                            sampled_features = sampled_features * (0.9 + decay)
                            
                            sampled_data.append(sampled_features)
                            sampled_label.extend([c_id] * num_sampled_pcls)
                            
                        except Exception as e:
                            self._logger.warning(f"[Alignment] Failed to sample from class {c_id}: {e}")
                            # Fallback: skip this class
                            continue

                if not sampled_data:
                    self._logger.warning("No samples generated, skipping classifier alignment")
                    return

                sampled_data = torch.cat(sampled_data, dim=0).float().cuda()
                sampled_label = torch.tensor(sampled_label).long().cuda()

                # Shuffle the data
                sf_indexes = torch.randperm(sampled_data.size(0))
                sampled_data = sampled_data[sf_indexes]
                sampled_label = sampled_label[sf_indexes]

                # Train per class to follow SLCA methodology
                samples_per_class = len(sampled_data) // crct_num
                
                for _iter in range(crct_num):
                    start_idx = _iter * samples_per_class
                    end_idx = min((_iter + 1) * samples_per_class, len(sampled_data))
                    
                    if start_idx >= end_idx:
                        continue
                        
                    inp = sampled_data[start_idx:end_idx]
                    tgt = sampled_label[start_idx:end_idx]
                    
                    # Forward pass through head only
                    logits = self.model.head(inp)["logits"]

                    # Apply logit normalization if configured
                    logit_norm_factor = self._config.get("train_ca_logit_norm", None)
                    if logit_norm_factor is not None and logit_norm_factor > 0:
                        # Per-task normalization similar to SLCA
                        per_task_norm = []
                        prev_t_size = 0
                        
                        for _ti in range(self._cur_task + 1):
                            if _ti < len(task_sizes):
                                cur_t_size = prev_t_size + task_sizes[_ti]
                            else:
                                cur_t_size = prev_t_size + (self._total_classes - prev_t_size)
                                
                            temp_norm = torch.norm(logits[:, prev_t_size:cur_t_size], p=2, dim=-1, keepdim=True) + 1e-7
                            per_task_norm.append(temp_norm)
                            prev_t_size = cur_t_size
                        
                        per_task_norm = torch.cat(per_task_norm, dim=-1)
                        norms = per_task_norm.mean(dim=-1, keepdim=True)
                        
                        decoupled_logits = torch.div(logits[:, :crct_num], norms) / logit_norm_factor
                        loss = F.cross_entropy(decoupled_logits, tgt)
                    else:
                        loss = F.cross_entropy(logits[:, :crct_num], tgt)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses += loss.item()

                scheduler.step()
                
                # Evaluate alignment progress
                if (epoch + 1) % 5 == 0 or epoch == run_epochs - 1:
                    test_acc = self._compute_alignment_accuracy()
                    info = f'[Alignment] Task {self._cur_task + 1} Epoch {epoch + 1}/{run_epochs} => Loss {losses/crct_num:.3f}, Test_acc {test_acc:.3f}'
                    self._logger.info(info)
            
            torch.save(self.model.head.state_dict(), self.head_ca_checkpoint(self._cur_task))
        else:
            self._logger.info("[Alignment] Loading trained classifier")
            self.model.head.load_state_dict(
                torch.load(self.head_ca_checkpoint(self._cur_task)), strict=True
            )
    
    def _align_classifier_legacy(self):
        self._logger.info("[Alignment] Computing class statistics")
        
        # First, compute class means and covariances for generative sampling
        if not hasattr(self, '_class_means') or self._class_means is None:
            self._class_means = {}
            self._class_covs = {}
        
        # Update class statistics for current task classes
        for class_idx in range(self._known_classes, self._total_classes):
            trainset_cls = self.data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
            )
            cls_loader = DataLoader(trainset_cls, batch_size=512, shuffle=False, num_workers=4)
            
            features_list = []
            self.model.eval()
            with torch.no_grad():
                for _, (_, _, x, y) in enumerate(cls_loader):
                    x = x.cuda()
                    features = self.model.get_features(x)
                    features_list.append(features.cpu())
            
            if features_list:
                all_features = torch.cat(features_list, dim=0)
                class_mean = all_features.mean(dim=0)
                
                # Compute covariance matrix and ensure it's positive definite
                if all_features.shape[0] > 1:  # Need at least 2 samples for covariance
                    class_cov = torch.cov(all_features.T)
                    
                    # Method 1: Add larger regularization to ensure positive definiteness
                    min_eigenval = torch.linalg.eigvals(class_cov).real.min()
                    if min_eigenval <= 0:
                        reg_term = abs(min_eigenval) + 1e-3
                        self._logger.info(f"[Alignment] Class {class_idx}: regularizing covariance (min_eig={min_eigenval:.6f}, reg={reg_term:.6f})")
                    else:
                        reg_term = 1e-4
                    class_cov = class_cov + torch.eye(all_features.shape[1]) * reg_term
                    
                    # Verify positive definiteness
                    try:
                        torch.linalg.cholesky(class_cov)
                    except RuntimeError:
                        # Fallback: use diagonal covariance
                        self._logger.warning(f"[Alignment] Class {class_idx}: using diagonal covariance as fallback")
                        class_var = all_features.var(dim=0, unbiased=True)
                        class_cov = torch.diag(class_var + 1e-3)
                else:
                    # Single sample: use identity covariance
                    self._logger.warning(f"[Alignment] Class {class_idx}: only one sample, using identity covariance")
                    class_cov = torch.eye(all_features.shape[1]) * 1e-2
                
                self._class_means[class_idx] = class_mean
                self._class_covs[class_idx] = class_cov
                self._logger.info(f"[Alignment] Class {class_idx}: computed stats from {all_features.shape[0]} samples")
            else:
                self._logger.warning(f"[Alignment] Class {class_idx}: no features extracted")

        if self._config["reset"] or not os.path.exists(self.head_ca_checkpoint(self._cur_task)):
            if self._cur_task == 0:
                torch.save(self.model.head.state_dict(), self.head_ca_checkpoint(self._cur_task))
                return
            self._logger.info("[Alignment] Starting compact classifier training")
                
            run_epochs = self._config.get("train_ca_epochs", 10)
            crct_num = self._total_classes
            task_sizes = [self._total_classes - self._known_classes] if self._cur_task == 0 else self.data_manager.get_task_sizes()
            
            for p in self.model.head.parameters():
                p.requires_grad = True
            param_list = [p for p in self.model.head.parameters() if p.requires_grad]
            total_head_params = sum(p.numel() for p in param_list)
            self._logger.info(f"[Alignment] Training {total_head_params:,} head parameters ({len(param_list)} tensors)")
            optimizer = optim.SGD(param_list, lr=self._config.get("train_ca_lr", 1e-2), momentum=0.9, 
                                weight_decay=self._config.get("train_ca_weight_decay", 5e-4))
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)

            self.model.train()  # Set to training mode for classifier alignment
            
            for epoch in range(run_epochs):
                losses = 0.
                
                sampled_data = []
                sampled_label = []
                num_sampled_pcls = 256
                
                # Sample from class distributions
                for c_id in range(crct_num):
                    if c_id in self._class_means:
                        # Apply task-dependent decay similar to SLCA
                        t_id = c_id // (self._total_classes // (self._cur_task + 1)) if self._cur_task > 0 else 0
                        decay = (t_id + 1) / (self._cur_task + 1) * 0.1
                        
                        cls_mean = self._class_means[c_id].cuda() * (0.9 + decay)
                        cls_cov = self._class_covs[c_id].cuda()
                        
                        # Sample from multivariate normal distribution with error handling
                        try:
                            from torch.distributions.multivariate_normal import MultivariateNormal
                            m = MultivariateNormal(cls_mean.float(), cls_cov.float())
                            sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                            sampled_data.append(sampled_data_single)
                            sampled_label.extend([c_id] * num_sampled_pcls)
                        except Exception as e:
                            self._logger.warning(f"[Alignment] Failed to sample from class {c_id}: {e}")
                            # Fallback: sample from simple normal around mean
                            noise_scale = 0.1
                            noise = torch.randn(num_sampled_pcls, cls_mean.shape[0]).cuda() * noise_scale
                            sampled_data_single = cls_mean.unsqueeze(0) + noise
                            sampled_data.append(sampled_data_single)
                            sampled_label.extend([c_id] * num_sampled_pcls)

                if not sampled_data:
                    self._logger.warning("No samples generated, skipping classifier alignment")
                    return

                sampled_data = torch.cat(sampled_data, dim=0).float().cuda()
                sampled_label = torch.tensor(sampled_label).long().cuda()

                # Shuffle the data
                sf_indexes = torch.randperm(sampled_data.size(0))
                sampled_data = sampled_data[sf_indexes]
                sampled_label = sampled_label[sf_indexes]

                # Train per class to follow SLCA methodology
                samples_per_class = len(sampled_data) // crct_num
                
                for _iter in range(crct_num):
                    start_idx = _iter * samples_per_class
                    end_idx = min((_iter + 1) * samples_per_class, len(sampled_data))
                    
                    if start_idx >= end_idx:
                        continue
                        
                    inp = sampled_data[start_idx:end_idx]
                    tgt = sampled_label[start_idx:end_idx]
                    
                    logits = self.model.head(inp)["logits"]
                    logit_norm_factor = self._config.get("train_ca_logit_norm", None)
                    if logit_norm_factor is not None and logit_norm_factor > 0:
                        per_task_norm = []
                        prev_t_size = 0
                        
                        for _ti in range(self._cur_task + 1):
                            if _ti < len(task_sizes):
                                cur_t_size = prev_t_size + task_sizes[_ti]
                            else:
                                cur_t_size = prev_t_size + (self._total_classes - prev_t_size)
                                
                            temp_norm = torch.norm(logits[:, prev_t_size:cur_t_size], p=2, dim=-1, keepdim=True) + 1e-7
                            per_task_norm.append(temp_norm)
                            prev_t_size = cur_t_size
                        
                        per_task_norm = torch.cat(per_task_norm, dim=-1)
                        norms = per_task_norm.mean(dim=-1, keepdim=True)
                        
                        decoupled_logits = torch.div(logits[:, :crct_num], norms) / logit_norm_factor
                        loss = F.cross_entropy(decoupled_logits, tgt)
                    else:
                        loss = F.cross_entropy(logits[:, :crct_num], tgt)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses += loss.item()

                scheduler.step()
                
                # Evaluate alignment progress
                if (epoch + 1) % 5 == 0 or epoch == run_epochs - 1:
                    test_acc = self._compute_alignment_accuracy()
                    info = f'[Alignment] Task {self._cur_task + 1} Epoch {epoch + 1}/{run_epochs} => Loss {losses/crct_num:.3f}, Test_acc {test_acc:.3f}'
                    self._logger.info(info)
            
            self._logger.info(f"[Alignment] Classifier alignment completed for task {self._cur_task + 1}")
            torch.save(self.model.head.state_dict(), self.head_ca_checkpoint(self._cur_task))
        else:
            self._logger.info("[Alignment] Loading trained classifier")
            self.model.head.load_state_dict(
                torch.load(self.head_ca_checkpoint(self._cur_task)), strict=True
            )

    def _compute_alignment_accuracy(self):
        """Compute test accuracy during classifier alignment"""
        test_set = self.data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=4)
        
        self.model.eval()
        correct, total = 0, 0
        
        with torch.no_grad():
            for _, (_, _, x, y) in enumerate(test_loader):
                x, y = x.cuda(), y.cuda()
                features = self.model.get_features(x)
                logits = self.model.head(features)["logits"]
                
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        return 100.0 * correct / total if total > 0 else 0.0 

    def fit(self):
        trainset_CPs = self.data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="test",
        )
        train_loader_CPs = DataLoader(
            trainset_CPs, batch_size=512, shuffle=False, num_workers=4
        )
        
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

            # self.model.backbone.load_state_dict(torch.load(self.backbone_checkpoint(self._cur_task)), strict=False)
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
            Wo = torch.linalg.solve(
                G_val + ridge * torch.eye(G_val.size(dim=0)), Q_val
            ).T  # better nmerical stability than .inv
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

    def head_ca_checkpoint(self, task):
        return f"{CHECKPOINT_DIR}/{self.prefix()}_head_ca_{task}.pt"

def set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

data_table = {
    "cifar224": [(10, 10, 10)],
    # "imagenetr": [(10, 20, 20)],
    # "imageneta": [(10, 20, 20)],
    # "cub": [(10, 20, 20)],
    # "omnibenchmark": [(10, 30, 30)],
    # "vtab": [(5, 10, 10)],
}

def run_single_experiment(dataset_name, config_name, experiment_config, logger):
    """Run experiment for a single dataset with specific configuration"""
    
    # Fixed hyperparameters
    model_backbone = "vit_base_patch16_224_lora"
    seed = 1993
    
    set_random(seed)
    
    config = {
        "seed": seed,
        "reset": True,
    }
    
    # Get dataset configuration
    dataset_num_task, dataset_init_cls, dataset_increment = data_table[dataset_name][0]
    dataset_config = {
        "dataset_name": dataset_name,
        "dataset_num_task": dataset_num_task,
        "dataset_init_cls": dataset_init_cls,
        "dataset_increment": dataset_increment,
    }
    config.update(dataset_config)
    
    # Base training configuration
    train_config = {
        "train_method": "seq",
        "train_epochs": 10,
        "train_batch_size": 48,
        "train_optim": "sgd",
        "train_base_lr": 1e-2,
        "train_weight_decay": 5e-4,

        "train_local_robustness_num_clusters": 5,
        
        "train_base_reg_weight": 1,
        "train_buffer_percent": 0.1,
        "train_buffer_size": 200,
        "train_buffer_decay": 0.95,

        "train_ca_epochs": 10,
        "train_ca_lr": 1e-2,
        "train_ca_logit_norm": 0.1,
        "train_ca_weight_decay": 5e-4,
        "train_ca_method": "mul_norm"
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
    
    model_config = {
        "model_backbone": model_backbone,
        "model_classifier": ["mlp", "nme"],
        "model_lora_r": 256,
        "model_lora_alpha": 512,
        "model_lora_dropout": 0.0,
        "model_merge_base": "init",
        "model_merge_coef": 1.0,
        "model_merge_topk": 100,
        "model_M": 10000,
    }
    config.update(model_config)

    # Update with experiment-specific configuration
    config.update(experiment_config)
    
    experiment_name = f"{dataset_name}_{config_name}"
    
    try:
        learner = Learner(config, logger)
        learner.learn(data_manager)
        
        # Get metrics for both classifiers
        mlp_faa = learner._mlp_faa
        mlp_asa = learner._mlp_asa
        nme_faa = learner._nme_faa
        nme_asa = learner._nme_asa
        
        result = {
            "MLP_FAA": mlp_faa,
            "MLP_ASA": mlp_asa,
            "NME_FAA": nme_faa,
            "NME_ASA": nme_asa,
            "config": experiment_config
        }
        
        logger.info(f"[Experiment {experiment_name}]")
        logger.info(f"  Configuration: {experiment_config}")
        logger.info(f"  MLP - FAA: {mlp_faa:.2f}, ASA: {mlp_asa:.2f}")
        logger.info(f"  NME - FAA: {nme_faa:.2f}, ASA: {nme_asa:.2f}")

        # Clean up memory
        del learner
        torch.cuda.empty_cache()
        gc.collect()
        
        return result
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"[Experiment {experiment_name}] Detailed Error:")
        logger.error(f"Exception Type: {type(e).__name__}")
        logger.error(f"Exception Message: {str(e)}")
        logger.error(f"Full Traceback:\n{error_details}")
        
        return {
            "MLP_FAA": 0.0,
            "MLP_ASA": 0.0,
            "NME_FAA": 0.0,
            "NME_ASA": 0.0,
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": error_details,
            "config": experiment_config
        }


def run_experiments():
    """Run experiments for all datasets with 4 different configurations"""
    timestamp = datetime.now().strftime("%Y_%m_%d")
    
    # Define 4 experimental configurations
    experiment_configs = {
        # "robustness_only": {
        #     "train_robustness": True,
        #     "train_ca": False,
        #     "model_merge": "none",
        #     "model_use_ranpac": False,
        #     "model_ensemble": False
        # },
        # "ca_only": {
        #     "train_robustness": False,
        #     "train_ca": True,
        #     "model_merge": "none",
        #     "model_use_ranpac": False,
        #     "model_ensemble": False
        # },
        # "merge_only": {
        #     "train_robustness": False,
        #     "train_ca": False,
        #     "model_merge": "ties",
        #     "model_use_ranpac": None,
        #     "model_ensemble": False
        # },
        # "ranpac_only": {
        #     "train_robustness": False,
        #     "train_ca": False,
        #     "model_merge": "none",
        #     "model_use_ranpac": True,
        #     "model_ensemble": False
        # },
        # "merge_ensemble": {
        #     "train_robustness": False,
        #     "train_ca": False,
        #     "model_merge": "ties", # ensemble requires model merging
        #     "model_use_ranpac": False,
        #     "model_ensemble": True
        # },
        # "merge_robustness": {
        #     "train_robustness": True,
        #     "train_ca": False,
        #     "model_merge": "ties",
        #     "model_use_ranpac": False,
        #     "model_ensemble": False
        # },
        # "wo_robustness": {
        #     "train_robustness": False,
        #     "train_ca": True,
        #     "model_merge": "ties",
        #     "model_use_ranpac": True,
        #     "model_ensemble": True
        # },
        "baseline": {
            "train_robustness": False,
            "train_ca": True,
            "model_merge": "max",
            "model_use_ranpac": True,
            "model_ensemble": False
        },
    }
    
    all_results = {}
    
    for dataset_name in data_table.keys():
        print(f"\n{'='*60}")
        print(f"Starting experiments for dataset: {dataset_name}")
        print(f"{'='*60}")
        
        dataset_results = {}
        
        for config_name, config in experiment_configs.items():
            print(f"\n  Running configuration: {config_name}")
            print(f"  Config: {config}")
            
            # Create unique logger for each dataset-config combination
            log_file = f"logs/{timestamp}_{dataset_name}_{config_name}.log"
            logger = create_unique_logger(f"logger_{dataset_name}_{config_name}", log_file)
            
            logger.info(f"Starting experiment: {dataset_name} - {config_name}")
            logger.info(f"Configuration: {config}")
            experiment_start_time = time.time()
            
            result = run_single_experiment(dataset_name, config_name, config, logger)
            
            experiment_end_time = time.time()
            logger.info(f"Experiment {dataset_name}_{config_name} time: {experiment_end_time - experiment_start_time:.2f} seconds")
            
            dataset_results[config_name] = result
            
            # Print progress to console
            if "error" not in result:
                print(f"    ✓ {config_name} completed successfully")
                print(f"      MLP - FAA: {result['MLP_FAA']:.2f}, ASA: {result['MLP_ASA']:.2f}")
                print(f"      NME - FAA: {result['NME_FAA']:.2f}, ASA: {result['NME_ASA']:.2f}")
            else:
                print(f"    ✗ {config_name} failed: {result['error']}")
        
        all_results[dataset_name] = dataset_results
        
        # Print dataset summary
        print(f"\n  Dataset {dataset_name} summary:")
        for config_name, result in dataset_results.items():
            if "error" not in result:
                print(f"    {config_name}: MLP_FAA={result['MLP_FAA']:.2f}, NME_FAA={result['NME_FAA']:.2f}")
            else:
                print(f"    {config_name}: FAILED")
    
    # Create comprehensive summary logger
    summary_log_file = f"logs/{timestamp}_FULL_SUMMARY.log"
    summary_logger = create_unique_logger("logger_full_summary", summary_log_file)
    
    # Final comprehensive summary
    summary_logger.info("\n" + "="*80)
    summary_logger.info("COMPREHENSIVE EXPERIMENT SUMMARY")
    summary_logger.info("="*80)
    
    for dataset_name, dataset_results in all_results.items():
        summary_logger.info(f"\nDataset: {dataset_name}")
        summary_logger.info("-" * 40)
        
        for config_name, result in dataset_results.items():
            if "error" not in result:
                summary_logger.info(f"  {config_name}:")
                summary_logger.info(f"    Configuration: {result['config']}")
                summary_logger.info(f"    MLP - FAA: {result['MLP_FAA']:.2f}, ASA: {result['MLP_ASA']:.2f}")
                summary_logger.info(f"    NME - FAA: {result['NME_FAA']:.2f}, ASA: {result['NME_ASA']:.2f}")
            else:
                summary_logger.info(f"  {config_name}: FAILED - {result['error']}")
    
    # Create comparison table
    summary_logger.info("\n" + "="*80)
    summary_logger.info("COMPARISON TABLE (MLP_FAA / NME_FAA)")
    summary_logger.info("="*80)
    
    # Header
    header = f"{'Dataset':<15}"
    for config_name in experiment_configs.keys():
        header += f"{config_name:<20}"
    summary_logger.info(header)
    summary_logger.info("-" * len(header))
    
    # Data rows
    for dataset_name, dataset_results in all_results.items():
        row = f"{dataset_name:<15}"
        for config_name in experiment_configs.keys():
            result = dataset_results[config_name]
            if "error" not in result:
                row += f"{result['MLP_FAA']:.1f}/{result['NME_FAA']:.1f}".ljust(20)
            else:
                row += "FAILED".ljust(20)
        summary_logger.info(row)
    
    print(f"\n{'='*80}")
    print("All experiments completed!")
    print(f"{'='*80}")
    print("Generated log files:")
    print(f"  📊 Main summary: logs/{timestamp}_FULL_SUMMARY.log")
    print("  📁 Individual logs:")
    for dataset_name in data_table.keys():
        for config_name in experiment_configs.keys():
            print(f"    - logs/{timestamp}_{dataset_name}_{config_name}.log")
    print(f"{'='*80}")
    
    return all_results


def create_unique_logger(logger_name, log_file):
    """Create a unique logger with no duplicate handlers"""
    import logging
    
    # Create or get logger with unique name
    logger = logging.getLogger(logger_name)
    
    # Clear any existing handlers to prevent duplicates
    logger.handlers.clear()
    
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(message)s')
    console_formatter = logging.Formatter("%(asctime)s [%(filename)s] => %(message)s")
    
    # Set formatters
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# Run the experiments
if __name__ == "__main__":
    start_time = time.time()
    results = run_experiments()
    total_time = time.time() - start_time
    print(f"\nTotal experiment time: {total_time:.2f} seconds")
