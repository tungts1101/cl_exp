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


def setup_logger(log_file=f"logs/{timestamp}_exp56.log"):
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


class ReplayBuffer:
    def __init__(self, buffer_size: int, total_num_classes: int):
        self._buffer_size = buffer_size
        self._total_num_classes = total_num_classes
        self._buffer_size_per_class = buffer_size // total_num_classes
        self._buffer = {i: [] for i in range(total_num_classes)}

    def add(self, x: torch.Tensor, z: torch.Tensor, y: torch.Tensor):
        for i in range(y.size(0)):
            clz = y[i].item()
            self._buffer[clz].append((x[i].cpu(), z[i].cpu(), y[i].cpu()))

    def truncate(self):
        for target, records in self._buffer.items():
            if len(records) > self._buffer_size_per_class:
                self._buffer[target] = random.sample(
                    records, self._buffer_size_per_class
                )

    def __iter__(self, batch_size: int = 32):
        all_records = []
        for target, records in self._buffer.items():
            all_records.extend(records)
        random.shuffle(all_records)

        for i in range(0, len(all_records), batch_size):
            batch = all_records[i : i + batch_size]
            x_batch = torch.stack([x for x, _, _ in batch])
            z_batch = torch.stack([z for _, z, _ in batch])
            y_batch = torch.tensor([y for _, _, y in batch], dtype=torch.long)
            yield x_batch, z_batch, y_batch

    @property
    def size(self):
        return sum(len(samples) for samples in self._buffer.values())


class RandomReplayBuffer:
    def __init__(self, buffer_size: int, total_num_classes: int):
        self._buffer_size = buffer_size
        self._total_num_classes = total_num_classes
        self._num_seen_samples = 0
        self._buffer = []

    def add(self, x: torch.Tensor, z: torch.Tensor, y: torch.Tensor):
        for i in range(y.size(0)):
            self._num_seen_samples += 1
            if len(self._buffer) < self._buffer_size:
                self._buffer.append((x[i].cpu(), z[i].cpu(), y[i].cpu()))
            else:
                idx = random.randint(0, self._num_seen_samples - 1)
                if idx < len(self._buffer):
                    self._buffer[idx] = (x[i].cpu(), z[i].cpu(), y[i].cpu())

    def truncate(self):
        pass

    def __iter__(self, batch_size: int = 32):
        random.shuffle(self._buffer)
        for i in range(0, len(self._buffer), batch_size):
            batch = self._buffer[i : i + batch_size]
            x_batch = torch.stack([x for x, _, _ in batch])
            z_batch = torch.stack([z for _, z, _ in batch])
            y_batch = torch.tensor([y for _, _, y in batch], dtype=torch.long)
            yield x_batch, z_batch, y_batch

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
        torch.save(self.model.get_backbone_trainable_params(), self.backbone_base())

        self._faa, self._ffm = 0, 0

    def learn(self, data_manager):
        self.data_manager = data_manager

        # self.buffer = ReplayBuffer(config["train_buffer_size"], data_manager.get_total_classnum())
        self.buffer = RandomReplayBuffer(
            config["train_buffer_size"], data_manager.get_total_classnum()
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
            base_params = torch.load(self.backbone_base())
        elif self._config["model_merge_base"] == "first":
            base_params = torch.load(self.backbone_checkpoint(0))

        session_params = [
            torch.load(self.backbone_checkpoint(session))
            for session in range(self._cur_task + 1)
        ]
        backbone_params = merge(
            base_params,
            session_params,
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
            task_model = Model(self._config)
            if self._config["train_method"] == "seq":  # sequential training
                task_model.backbone.load_state_dict(
                    self.model.backbone.state_dict(), strict=True
                )
            elif self._config["train_method"] == "last":
                if self._cur_task > 0:
                    task_model.backbone.load_state_dict(
                        torch.load(self.backbone_checkpoint(self._cur_task - 1)),
                        strict=False,
                    )
            elif self._config["train_method"] == "first":
                if self._cur_task > 0:
                    task_model.backbone.load_state_dict(
                        torch.load(self.backbone_checkpoint(0)), strict=False
                    )
            
            for i in range(self._cur_task + 1):
                task_model.update_head(self._class_increments[i][1] - self._class_increments[i][0] + 1, freeze_old=False)
                if i < self._cur_task:
                    task_model.head.heads[i].load_state_dict(
                        self.model.head.heads[i].state_dict(), strict=True
                    )
            
            task_model.cuda()

            epochs = self._config["train_epochs"]

            parameters = [
                {
                    "params": [p for p in task_model.backbone.parameters() if p.requires_grad],
                    "lr": 1e-4,
                    "weight_decay": 5e-4,
                },
                # {
                #     "params": [
                #         p
                #         for i, head in enumerate(task_model.head.heads)
                #         if i != self._cur_task
                #         for p in head.parameters()
                #         if p.requires_grad
                #     ],
                #     "lr": 1e-6,
                #     "weight_decay": 0.0,
                # },
                {
                    "params": [
                        p
                        for p in task_model.head.heads[self._cur_task].parameters()
                        if p.requires_grad
                    ],
                    "lr": 1e-4,
                    "weight_decay": 5e-4,
                },
            ]

            
            total_params = 0
            for i, group in enumerate(parameters):
                group_params = sum(p.numel() for p in group["params"] if p.requires_grad)
                total_params += group_params
                print(f"Parameter group {i}: {group_params:,} trainable parameters")

            print(f"Total trainable parameters: {total_params:,}")
            
            # optimizer = optim.SGD(parameters, lr=1e-2, momentum=0.9, weight_decay=5e-4)
            optimizer = optim.AdamW(parameters, betas=(0.9, 0.999), eps=1e-8)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=1e-6
            )

            for epoch in range(epochs):
                task_model.train()
                total_loss, total_acc, total = 0, 0, 0

                for _, (_, _, x, y) in enumerate(train_loader):
                    x, y = x.cuda(), y.cuda()
                    # y = torch.where(y - self._known_classes >= 0, y - self._known_classes, -100)

                    features = task_model.get_features(x)
                    logits = task_model.head(features)["logits"]
                    loss = F.cross_entropy(logits, y)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(task_model.head.parameters(), 5.0)
                    optimizer.step()
                    
                    # total_norm = 0.0
                    # for name, param in task_model.head.named_parameters():
                    #     if param.grad is not None:
                    #         param_norm = param.grad.data.norm(2)
                    #         total_norm += param_norm.item() ** 2
                    #         print(f"{name:60s} | grad norm: {param_norm.item():.4e}")
                    #     else:
                    #         print(f"{name:60s} | grad: None")

                    # total_norm = total_norm ** 0.5
                    # print(f"\nTotal gradient norm: {total_norm:.4e}")

                    total_loss += loss.item()
                    total_acc += (logits.argmax(dim=1) == y).sum().item()
                    total += len(y)

                scheduler.step()

                if epoch % 2 == 1:
                    info = f"Task {self._cur_task + 1}, Epoch {epoch + 1}/{epochs}, Loss: {total_loss / total:.4f}, Acc: {total_acc / total:.4f}"
                    logger.info(info)

            torch.save(
                task_model.get_backbone_trainable_params(),
                self.backbone_checkpoint(self._cur_task),
            )
            # torch.save(
            #     task_model.head.state_dict(), self.head_checkpoint(self._cur_task)
            # )

            # del task_model
            # gc.collect()
            # torch.cuda.empty_cache()

        if self._config["model_merge"] == "none":
            print("Load session model...")
            self.model.backbone.load_state_dict(
                torch.load(self.backbone_checkpoint(self._cur_task)), strict=False
            )
        else:
            print("Perform model merging...")
            self.merge()
        
        self.model.update_head(self._class_increments[self._cur_task][1] - self._class_increments[self._cur_task][0] + 1)
        self.model.cuda()
        
        # print(self.model)
        
        for i in range(self._cur_task + 1):
            if i == self._cur_task:
                self.model.head.heads[i].load_state_dict(task_model.head.heads[i].state_dict(), strict=True)
            # elif i == self._cur_task - 1:
            #     for p_self, p_task in zip(self.model.head.heads[i].parameters(), task_model.head.heads[i].parameters()):
            #         if p_self.shape != p_task.shape:
            #             continue
            #         p_self.data.copy_(torch.lerp(p_self.data, p_task.data, 0.01))
        
        # self.model.head.load_state_dict(task_model.head.state_dict(), strict=True)
        
        del task_model
        gc.collect()
        torch.cuda.empty_cache()
                

        # task_model = Model(self._config)
        # task_model.backbone.load_state_dict(self.model.backbone.state_dict(), strict=False)
        # task_model.update_head(self._total_classes - self._known_classes)
        # task_model.cuda().eval()
        # with torch.no_grad():
        #     for _, (_, _, x, y) in enumerate(train_loader):
        #         x, y = x.cuda(), y.cuda()
        #         features = task_model.backbone(x)
        #         # z = task_model.head(features)['logits']
        #         # self.buffer.add(x, z, y)
        #         self.buffer.add(x, features, y)

        # self.buffer.truncate()

        # del task_model
        # gc.collect()
        # torch.cuda.empty_cache()

        # print(f"Buffer size: {self.buffer.size}")

        # cls_coefficients = torch.zeros(self._total_classes, dtype=torch.float32, device='cuda')
        # cls_counts = torch.zeros(self._total_classes, dtype=torch.float32, device='cuda')

        # for i, (x, z, y) in tqdm(enumerate(self.buffer, 32)):
        #     x, z, y = x.cuda(), z.cuda(), y.cuda()

        #     with torch.no_grad():
        #         f = self.model.get_features(x)

        #     z_norm = F.normalize(z, dim=1)
        #     f_norm = F.normalize(f, dim=1)

        #     similarity = (z_norm * f_norm).sum(dim=1)

        #     for j in range(y.size(0)):
        #         cls = y[j]
        #         cls_coefficients[cls] += similarity[j]
        #         cls_counts[cls] += 1

        # nonzero_mask = cls_counts > 0
        # cls_coefficients[nonzero_mask] /= cls_counts[nonzero_mask]
        # cls_coefficients[~nonzero_mask] = 1.0
        # # logger.info(f"Class coefficients: {cls_coefficients}")

        # self.model.update_fc(self._total_classes)
        # proto_set = self.data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="test")
        # proto_loader = DataLoader(proto_set, batch_size=128, shuffle=False, num_workers=4)
        # embedding_list = []
        # label_list = []
        # with torch.no_grad():
        #     for i, batch in tqdm(enumerate(proto_loader)):
        #         (_, _, data,label)=batch
        #         data=data.cuda()
        #         label=label.cuda()
        #         embedding = self.model.get_features(data)
        #         embedding_list.append(embedding.cpu())
        #         label_list.append(label.cpu())

        # embedding_list = torch.cat(embedding_list, dim=0)
        # label_list = torch.cat(label_list, dim=0)

        # cur_class_list=np.unique(proto_set.labels)
        # for class_index in range(self._total_classes):
        #     if class_index in cur_class_list:
        #         data_index=(label_list==class_index).nonzero().squeeze(-1)
        #         embedding=embedding_list[data_index]
        #         proto=embedding.mean(0)

        #         coef = cls_coefficients[class_index].item()  # scalar
        #         proto = coef * proto

        #         self.model.fc.weight.data[class_index]=proto
        #     else:
        #         self.model.fc.weight.data[class_index] *= cls_coefficients[class_index].item()

        # self.model.update_head(self._total_classes - self._known_classes, freeze_old=False)
        # self.model.head.heads[-1].load_state_dict(torch.load(self.head_checkpoint(self._cur_task)), strict=True)
        # self.model.train()
        # for p in self.model.get_backbone_trainable_params().values():
        #     p.requires_grad_(False)
        # head_trainable_params = sum(p.numel() for p in self.model.head.parameters() if p.requires_grad)
        # head_total_params = sum(p.numel() for p in self.model.head.parameters())
        # logger.info(f"Head trainable params: {head_trainable_params}, total params: {head_total_params}, percentage: {head_trainable_params * 100 / head_total_params:.2f}")

        # epochs = 5
        # B = 32
        # epsilon = 0.1  # noise scale
        # k = 8          # number of noisy samples
        # lambda_consist = 1.0  # consistency loss weight

        # optimizer = optim.SGD(self.model.head.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)

        # for epoch in range(epochs):
        #     loss_log = {
        #         "loss_cls": 0.0,
        #         "loss_dis": 0.0,
        #         "loss_consist": 0.0,
        #         "loss": 0.0,
        #         "acc": 0.0
        #     }
        #     num_samples = 0

        #     for i, (x, z, y) in enumerate(self.buffer, B):
        #         x, z, y = x.cuda(), z.cuda(), y.cuda()

        #         z_scatter = torch.zeros((x.size(0), self._total_classes), dtype=z.dtype, device=z.device)
        #         for yi in range(y.size(0)):
        #             clz = y[yi].item()
        #             task_idx = self._cls_to_task_idx[clz]
        #             range_start, range_end = self._class_increments[task_idx]
        #             z_scatter[yi, range_start:range_end+1] = z[yi]

        #         with torch.no_grad():
        #             features = self.model.get_features(x)

        #         logits = self.model.head(features)['logits']
        #         loss_cls = F.cross_entropy(logits, y)
        #         loss_dis = F.mse_loss(logits, z_scatter)

        #         # Gentle local robustness (consistency) loss
        #         x_repeat = x.unsqueeze(1).repeat(1, k, 1, 1, 1).view(-1, *x.shape[1:])
        #         noise = torch.randn_like(x_repeat) * epsilon
        #         x_noisy = (x_repeat + noise).clamp(0.0, 1.0)

        #         with torch.no_grad():
        #             clean_probs = logits.softmax(dim=1).detach().unsqueeze(1).repeat(1, k, 1).view(-1, logits.size(-1))
        #             noisy_features = self.model.get_features(x_noisy)
        #             noisy_logits = self.model.head(noisy_features)['logits']
        #             noisy_log_probs = noisy_logits.log_softmax(dim=1)

        #         loss_consist = F.kl_div(noisy_log_probs, clean_probs, reduction='batchmean')

        #         loss = loss_cls + 0.1 * loss_dis + lambda_consist * loss_consist
        #         # loss = loss_cls + lambda_consist * loss_consist

        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()

        #         loss_log["loss_cls"] += loss_cls.item()
        #         loss_log["loss_dis"] += loss_dis.item()
        #         loss_log["loss_consist"] += loss_consist.item()
        #         loss_log["loss"] += loss.item()
        #         loss_log["acc"] += (logits.argmax(dim=1) == y).sum().item()
        #         num_samples += x.size(0)

        #     logger.info(
        #         f"Task {self._cur_task + 1}, Epoch {epoch + 1}/{epochs}"
        #         f" Acc {loss_log['acc'] / num_samples:.4f},"
        #         f" Loss: {loss_log['loss'] / num_samples:.4f},"
        #         f" Loss_cls: {loss_log['loss_cls'] / num_samples:.4f},"
        #         f" Loss_dis: {loss_log['loss_dis'] / num_samples:.4f},"
        #         f" Loss_consist: {loss_log['loss_consist'] / num_samples:.4f}"
        #     )
        #     scheduler.step()

    def prefix(self):
        return f"{self._config['seed']}_{self._config['dataset_name']}_{self._config['dataset_num_task']}_\
            {self._config['model_backbone']}_{self._config['train_method']}_{self._config['model_merge']}_{self._config['model_merge_base']}"

    def backbone_base(self):
        return f"/media/ellen/HardDisk/cl/logs/checkpoints/{self.prefix()}_kd_backbone_base.pt"

    def backbone_checkpoint(self, task):
        return f"/media/ellen/HardDisk/cl/logs/checkpoints/{self.prefix()}_kd_backbone_{task}.pt"

    def head_checkpoint(self, task):
        return f"/media/ellen/HardDisk/cl/logs/checkpoints/{self.prefix()}_kd_head_{task}.pt"


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
    # "imagenetr": [(10, 20, 20)],
}

for model_backbone in ["vit_base_patch16_224_lora"]:
    for dataset_name in ["cifar224"]:
        start_time = time.time()
        faa, ffm = [], []
        for seed in [1993]:
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
                        "train_buffer_size": 5000,
                    }

                    config.update(train_config)

                    for item in config.items():
                        logger.info(f"{item[0]}: {item[1]}")

                    learner = Learner(config)
                    learner.learn(data_manager)

        logger.info(f"End experiment in {time.time() - start_time}\n")
