import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import random
import timm
from tqdm import tqdm
import os
from datetime import datetime
from peft import LoraConfig, get_peft_model
import copy
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
from petl import vision_transformer_ssf
from petl import vision_transformer_adapter
from petl.vpt import build_promptmodel
from easydict import EasyDict
from _exp import ContinualLearnerHead, setup_logger, get_backbone, RandomReplayBuffer, merge, compute_metrics
from inc_net import CosineLinear
from util import accuracy, set_random
import gc
import time
from deps import StreamingLDA


timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
logger = setup_logger(f"logs/_exp59.log")


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
        f = F.normalize(f, dim=1)
        return f

    def forward(self, x):
        f = self.get_features(x)
        y = self.fc(f)

        return y

    def __repr__(self):
        trainable_params = count_parameters(self, trainable=True)
        total_params = count_parameters(self)
        return f"Model(trainable_params={trainable_params}, total_params={total_params}, percentage={trainable_params * 100 / total_params:.2f})"


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
        os.makedirs("/home/lis/checkpoints", exist_ok=True)

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
        self.data_manager = data_manager

        # self.buffer = ReplayBuffer(config["train_buffer_size"], data_manager.get_total_classnum())
        self.buffer = RandomReplayBuffer(config["train_buffer_size"], 0.95)

        num_tasks = data_manager.nb_tasks
        self._total_classnum = data_manager.get_total_classnum()
        self._slda_classifier = StreamingLDA(
            input_shape=self.model.feature_dim, num_classes=self._total_classnum
        )

        self._centers = self.sample_cluster_centers(self._total_classnum * 5)
        print(f"Cluster centers sampled: {self._centers.shape}")

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

                # logits = self.model(x)["logits"]

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

        if (
            not os.path.exists(self.backbone_checkpoint(self._cur_task))
            or not os.path.exists(self.head_checkpoint(self._cur_task))
            or self._config["reset"]
        ):
            if self._config["train_method"] == "seq":
                self.model.backbone.load_state_dict(
                    torch.load(self.backbone_checkpoint(self._cur_task - 1)),
                    strict=False,
                )

            self.model.update_head(
                self._total_classes - self._known_classes, freeze_old=True
            )
            self.model.cuda()
            print(self.model)

            epochs = self._config["train_epochs"]

            parameters = [
                {
                    "params": [
                        p for p in self.model.backbone.parameters() if p.requires_grad
                    ],
                    "lr": 1e-2,
                    "weight_decay": 5e-4,
                },
                # {
                #     "params": [
                #         p
                #         for i, head in enumerate(self.model.head.heads)
                #         if i != self._cur_task
                #         for p in head.parameters()
                #         if p.requires_grad
                #     ],
                #     "lr": 1e-4,
                #     "weight_decay": 1e-4,
                # },
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

            # all_features, all_labels = [], []
            # with torch.no_grad():
            #     pretrained_backbone = timm.create_model(
            #         "vit_base_patch16_224", pretrained=True, num_classes=0
            #     ).eval().cuda()

            #     for _, (_, _, x, y) in tqdm(enumerate(train_loader)):
            #         x, y = x.cuda(), y.cuda()
            #         features = pretrained_backbone(x)
            #         # features = self.model.get_features(x)
            #         all_features.append(features.cpu())
            #         all_labels.append(y.cpu())

            # all_features = torch.cat(all_features, dim=0)
            # all_labels = torch.cat(all_labels, dim=0)

            # K = int(5 * (self._total_classes - self._known_classes))
            # kmeans = KMeans(n_clusters=K, random_state=0, n_init='auto').fit(all_features.numpy())
            # cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).cuda()
            
            base_reg_weight = 1.0

            for epoch in range(epochs):
                self.model.train()
                total_ce, total_rb, total_acc, total = 0, 0, 0, 0

                epoch_scale = max(0.1, 1.0 - (epoch / epochs))
                reg_weight = base_reg_weight * epoch_scale
                reg_weight = max(0.1, reg_weight)
                logger.info(
                    f"Task {self._cur_task + 1}, Epoch {epoch + 1}/{epochs}, "
                    f"Reg Weight: {reg_weight:.4f}"
                )

                for _, (_, x_aug, x, y) in enumerate(train_loader):
                    x_aug, x, y = x_aug.cuda(), x.cuda(), y.cuda()
                    y = torch.where(
                        y - self._known_classes >= 0, y - self._known_classes, -100
                    )

                    features = self.model.get_features(x)
                    logits = self.model.head.heads[self._cur_task](features)
                    ce_loss = F.cross_entropy(logits, y)

                    rb_loss = torch.tensor(0.0).cuda()


                    # calculate local robustness loss
                    with torch.no_grad():
                        # distances = torch.cdist(features, self._centers, p=2)
                        # nearest_idx = distances.argmin(dim=1) # [B]

                        sim_matrix = torch.matmul(features, self._centers.T)  # dot product
                        nearest_idx = sim_matrix.argmax(dim=1) # [B]
                    
                    num_clusters = self._centers.shape[0]
                    total_samples = x.size(0)

                    for i in range(num_clusters):
                        cluster_mask = nearest_idx == i
                        if cluster_mask.sum() == 0:
                            continue
                        cluster_features = features[cluster_mask]
                        center_features = self._centers[i].unsqueeze(0).expand_as(cluster_features)

                        # cluster_features = F.normalize(cluster_features, dim=1)
                        # center_features = F.normalize(center_features, dim=1)

                        # pos_sim = (cluster_features * center_features).sum(dim=1)

                        # reg_loss = (cluster_mask.sum().float() / total_samples ) * (1.0 - pos_sim.mean())
                        # rb_loss += reg_loss

                        cluster_logits = logits[cluster_mask]
                        cluster_probs = F.softmax(cluster_logits, dim=1)
                        cluster_entropies = -(cluster_probs * cluster_probs.log()).sum(dim=1)

                        center_features = F.normalize(center_features, dim=1)
                        center_logits = self.model.head.heads[self._cur_task](center_features)
                        center_probs = F.softmax(center_logits, dim=1)
                        center_entropies = -(center_probs * center_probs.log()).sum(dim=1)

                        rb_loss += (cluster_mask.sum().float() / total_samples ) * torch.abs(cluster_entropies - center_entropies).mean()


                    # probs = F.softmax(logits, dim=1)
                    # sample_entropies = -(probs * probs.log()).sum(dim=1)  # [B]

                    # aug_features = task_model.get_features(x_aug)
                    # temperature = 0.1
                    # features_norm = F.normalize(features, dim=1)
                    # aug_features_norm = F.normalize(aug_features, dim=1)

                    # sim_matrix = torch.matmul(features_norm, aug_features_norm.t())  # [B, B]
                    # logits_nce = sim_matrix / temperature

                    # labels_nce = torch.arange(x.size(0)).cuda()
                    # info_nce_loss = F.cross_entropy(logits_nce, labels_nce)

                    # rb_loss = sample_entropies.mean() + info_nce_loss

                    # # === Entropy LOSS ===
                    # with torch.no_grad():
                    #     distances = torch.cdist(pretrained_backbone(x), cluster_centers, p=2)
                    #     nearest_idx = distances.argmin(dim=1)
                    # probs = F.softmax(logits, dim=1)
                    # sample_entropies = -(probs * probs.log()).sum(dim=1)  # [B]
                    # cluster_feats = cluster_centers[nearest_idx]  # [B, D]
                    # cluster_logits = task_model.head(cluster_feats)['logits']
                    # cluster_probs = F.softmax(cluster_logits, dim=1)
                    # cluster_entropies = -(cluster_probs * cluster_probs.log()).sum(dim=1)
                    # rb_loss = torch.abs(sample_entropies - cluster_entropies).mean()
                    # rb_loss = sample_entropies.mean()

                    # # === L2 LOSS ===
                    # lambda_rb_loss = 0.05
                    # nearest_centers = cluster_centers[nearest_idx]  # [B, D]
                    # l2_loss = F.mse_loss(features, nearest_centers) + F.l1_loss(features, nearest_centers)
                    # rb_loss = l2_loss

                    # # === InfoNCE LOSS ===
                    # lambda_rb_loss = 0.5
                    # temperature = 0.01
                    # sim_matrix = torch.matmul(F.normalize(features, dim=1), F.normalize(cluster_centers, dim=1).t())  # [B, K]
                    # logits_nce = sim_matrix / temperature
                    # labels_nce = nearest_idx
                    # info_nce_loss = F.cross_entropy(logits_nce, labels_nce)
                    # rb_loss = info_nce_loss

                    # # === KL Divergence LOSS ===
                    # lambda_rb_loss = 0.5
                    # probs = F.softmax(logits, dim=1)
                    # sample_log_probs = probs.log()

                    # with torch.no_grad():
                    #     distances = torch.cdist(pretrained_backbone(x), cluster_centers, p=2)
                    #     nearest_idx = distances.argmin(dim=1)

                    # cluster_feats = cluster_centers[nearest_idx]
                    # cluster_logits = task_model.head(cluster_feats)['logits']
                    # cluster_probs = F.softmax(cluster_logits, dim=1)
                    # rb_loss = F.kl_div(sample_log_probs, cluster_probs, reduction='batchmean')

                    # loss = ce_loss + lambda_rb_loss * rb_loss

                    loss = ce_loss + reg_weight * rb_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_ce += ce_loss.item()
                    total_rb += rb_loss.item()
                    total_acc += (logits.argmax(dim=1) == y).sum().item()
                    total += len(y)

                scheduler.step()

                logger.info(
                    f"Task {self._cur_task + 1}, Epoch {epoch + 1}/{epochs}, "
                    f"Loss: {(total_ce + reg_weight * total_rb) / total:.4f}, "
                    f"CE Loss: {total_ce / total:.4f}, RB Loss: {total_rb / total:.4f}, "
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

        # # Add samples to replay buffer...
        # self.model.eval()
        # with torch.no_grad():
        #     for _, batch in enumerate(train_loader):
        #         _, _, x, y = batch
        #         x, y = x.cuda(), y.cuda()
        #         features = self.model.get_features(x)
        #         self.buffer.add(x, features, y)
        # logger.info(f"Buffer size: {self.buffer.size}")
        # logger.info(f"Buffer size by class: {self.buffer.size_by_class}")

        if self._config["model_merge"] != "none":
            print(f"Perform model merging with method {self._config['model_merge']}")
            self.merge()

        # proto_set = self.data_manager.get_dataset(
        #     np.arange(self._known_classes, self._total_classes),
        #     source="train",
        #     mode="test",
        # )
        # proto_loader = DataLoader(
        #     proto_set, batch_size=128, shuffle=False, num_workers=4
        # )

        # self.model.update_fc(self._total_classes)
        # embedding_list = []
        # label_list = []
        # with torch.no_grad():
        #     for i, batch in tqdm(enumerate(proto_loader)):
        #         (_, _, data, label) = batch
        #         data = data.cuda()
        #         label = label.cuda()
        #         embedding = self.model.get_features(data)
        #         embedding_list.append(embedding.cpu())
        #         label_list.append(label.cpu())

        # embedding_list = torch.cat(embedding_list, dim=0)
        # label_list = torch.cat(label_list, dim=0)

        # class_list = np.unique(proto_set.labels)
        # for class_index in class_list:
        #     data_index = (label_list == class_index).nonzero().squeeze(-1)
        #     embedding = embedding_list[data_index]
        #     proto = embedding.mean(0)
        #     self.model.fc.weight.data[class_index] = proto





        # with torch.no_grad():
        #     for i, batch in tqdm(enumerate(proto_loader)):
        #         (_, _, data,label)=batch
        #         data=data.cuda()
        #         label=label.cuda()
        #         embedding = self.model.get_features(data)
        #         for x, y in zip(embedding, label):
        #             self._slda_classifier.fit(x.cpu(), y.view(1, ))

        # logger.info("Aligning classifier with robust training")
        # self.model.train()
        # if self._cur_task > 0:
        #     print(self.model)
        #     epochs = 10

        #     # clustered_sets = self.buffer.split(int(1.2 * self._total_classes))

        #     optimizer = optim.SGD(self.model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
        #     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)

        #     for epoch in range(epochs):
        #         total_acc, total_samples = 0, 0
        #         total_losses, total_ce_losses, total_rb_losses = 0.0, 0.0, 0.0

        #         lambda_ce_loss, lambda_rb_loss = 1.0, 1.0

        #         # for i, cluster in enumerate(clustered_sets):
        #         #     if len(cluster) == 0:
        #         #         continue

        #         #     x, z, y = zip(*cluster)
        #         #     x = torch.stack(x).cuda()
        #         #     z = torch.stack(z).cuda()
        #         #     y = torch.tensor(y, dtype=torch.long).cuda()

        #         #     f = self.model.get_features(x)
        #         #     logits = self.model.head(f)['logits']
        #         #     ce_loss = F.cross_entropy(logits, y)

        #         #     cluster_center = z.mean(dim=0, keepdim=True).cuda()
        #         #     cluster_logits = self.model.head(cluster_center)['logits']
        #         #     cluster_probs = F.softmax(cluster_logits, dim=1)
        #         #     cluster_entropy = -(cluster_probs * cluster_probs.log()).sum(dim=1)

        #         #     probs = F.softmax(logits, dim=1)
        #         #     sample_entropies = -(probs * probs.log()).sum(dim=1)

        #         #     local_rb_loss = torch.abs(sample_entropies - cluster_entropy).mean()

        #         #     cluster_weight = len(cluster) / self.buffer.size
        #         #     weighted_rb_loss = cluster_weight * local_rb_loss

        #         #     loss = lambda_ce_loss * ce_loss + lambda_rb_loss * weighted_rb_loss

        #         #     optimizer.zero_grad()
        #         #     loss.backward()
        #         #     optimizer.step()

        #         #     total_losses += loss.item() * x.size(0)
        #         #     total_ce_losses += ce_loss.item() * x.size(0)
        #         #     total_rb_losses += weighted_rb_loss.item() * x.size(0)
        #         #     total_acc += (logits.argmax(dim=1) == y).sum().item()
        #         #     total_samples += x.size(0)

        #         for _ in range(int(self.buffer.size // 64)):
        #             x, z, y = self.buffer.sample(64)
        #             x, z, y = x.cuda(), z.cuda(), y.cuda()
        #             f = self.model.get_features(x)
        #             logits = self.model.head(f)['logits']

        #             ce_loss = F.cross_entropy(logits, y)
        #             probs = F.softmax(logits, dim=1)
        #             sample_entropies = -(probs * probs.log()).sum(dim=1)
        #             rb_loss = sample_entropies.mean()

        #             loss = lambda_ce_loss * ce_loss + lambda_rb_loss * rb_loss

        #             optimizer.zero_grad()
        #             loss.backward()
        #             optimizer.step()

        #             total_losses += loss.item() * x.size(0)
        #             total_ce_losses += ce_loss.item() * x.size(0)
        #             total_rb_losses += rb_loss.item() * x.size(0)
        #             total_acc += (logits.argmax(dim=1) == y).sum().item()
        #             total_samples += x.size(0)

        #         scheduler.step()
        #         log = {
        #             "total_loss": total_losses / total_samples,
        #             "ce_loss": total_ce_losses / total_samples,
        #             "rb_loss": total_rb_losses / total_samples,
        #             "accuracy": total_acc / total_samples,
        #         }
        #         logger.info(
        #             f"Task {self._cur_task + 1}, Epoch {epoch + 1}/{epochs}, "
        #             f"Total Loss: {log['total_loss']:.4f}, CE Loss: {log['ce_loss']:.4f}, "
        #             f"RB Loss: {log['rb_loss']:.4f}, Accuracy: {log['accuracy']:.4f}"
        #         )

        # self.buffer.update_weights()

    def prefix(self):
        return f"{self._config['seed']}_{self._config['dataset_name']}_{self._config['dataset_num_task']}_{self._config['model_backbone']}_{self._config['train_method']}_robust_training"

    def backbone_checkpoint(self, task=-1):
        return f"/home/lis/checkpoints/{self.prefix()}_backbone" + (
            f"_{task}.pt" if task >= 0 else "_base.pt"
        )

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
    "cifar224": [(10, 10, 10)],
    "imagenetr": [(10, 20, 20)],
    "imageneta": [(10, 20, 20)],
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
                    "train_epochs": 10,
                    "train_batch_size": 48,
                    "train_method": "seq",
                    "train_buffer_size": 1000,
                    "train_split_K": 250,
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

                for model_merge in ["none"]:
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
