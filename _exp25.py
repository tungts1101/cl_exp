# sequential peft + adv program
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import timm
from tqdm import tqdm
import os
import logging
from datetime import datetime
from peft import LoraConfig, get_peft_model, TaskType

from utils.data_manager import DataManager
from utils.toolkit import count_parameters
from petl import vision_transformer_ssf
from _exp import ContinualLearnerHead
from util import compute_metrics, accuracy, set_random
import copy
import gc


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


def setup_logger(log_file=f"logs/{timestamp}_exp25.log"):
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG)

    logger.propagate = False

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_formatter = logging.Formatter("%(asctime)s - %(message)s")
    console_formatter = logging.Formatter("%(message)s")

    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = setup_logger()

config = {
    "peft": "lora",
    "r": 16,
    "lora_alpha": 16,
}
pretrained_backbone = "vit_base_patch16_224_ssf"


def configure_model(model):
    model.train()
    model.requires_grad_(False)
    for name, param in model.named_parameters():
        if "ssf_" in name:
            param.requires_grad = True
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " "check which require grad"
    assert not has_all_params, (
        "tent should not update all params: " "check which require grad"
    )


def collect_params(model):
    params = []
    names = []
    for name, param in model.named_parameters():
        if "ssf_scale_1" in name or "ssf_shift_1" in name:
            params.append(param)
            names.append(name)
    return params, names


class Tent(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """

    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = copy_model_and_optimizer(
            self.model, self.optimizer
        )

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = forward_and_adapt(x, self.model, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(
            self.model, self.optimizer, self.model_state, self.optimizer_state
        )


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    outputs = model(x)
    # adapt
    loss = softmax_entropy(outputs).mean(0)
    print(f"Adapt loss: {loss.item()}")
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = copy.deepcopy(model.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            pretrained_backbone, pretrained=True, num_classes=0
        )
        self.freeze_backbone()
        self.head = ContinualLearnerHead(self.backbone.num_features, 20).to(device)

    def freeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            if ("ssf_scale_" not in name) and ("ssf_shift_" not in name):
                param.requires_grad = False

    def get_features(self, x):
        return self.backbone(x)

    def forward(self, x):
        f = self.backbone(x)
        y = self.head(f)["logits"]
        return y

    def __repr__(self):
        trainable_params = count_parameters(self, trainable=True)
        total_params = count_parameters(self)
        return f"Model(trainable_params={trainable_params}, total_params={total_params}, percentage={trainable_params * 100 / total_params:.2f})"


def kl_divergence_loss(new_features, pretrained_features, temperature=1.0):
    new_logits = new_features / temperature
    pretrained_logits = pretrained_features / temperature
    return F.kl_div(
        F.log_softmax(new_logits, dim=-1),
        F.softmax(pretrained_logits, dim=-1),
        reduction="batchmean",
    )


def contrastive_loss(new_features, pretrained_features, temperature=0.5):
    new_features = F.normalize(new_features, dim=-1)
    pretrained_features = F.normalize(pretrained_features, dim=-1)
    similarity = torch.exp(
        torch.sum(new_features * pretrained_features, dim=-1) / temperature
    )
    return -torch.log(similarity).mean()


class Learner:
    def __init__(self):
        self._known_classes = 0
        self._classes_seen_so_far = 0
        self._class_increments = []
        self.model = Model().to(device)
        self.accuracy_matrix = []
        self.cur_task = -1

    def learn(self, data_manager):
        self.data_manager = data_manager
        num_tasks = data_manager.nb_tasks - 1

        for task in range(num_tasks):
            self.before_task(task, data_manager)

            # if task == 0:
            trainset = data_manager.get_dataset(
                np.arange(self._known_classes, self._classes_seen_so_far),
                source="train",
                mode="train",
            )
            train_loader = DataLoader(
                trainset,
                batch_size=config["fine_tune_train_batch_size"],
                shuffle=True,
                num_workers=4,
            )
            self.finetune(train_loader)

            test_set = self.data_manager.get_dataset(
                np.arange(0, self._classes_seen_so_far), source="test", mode="test"
            )
            test_loader = DataLoader(
                test_set, batch_size=64, shuffle=False, num_workers=4
            )
            self.eval(test_loader)
            self.after_task()

    def before_task(self, task, data_manager):
        print(self.model)
        self._classes_seen_so_far = self._known_classes + data_manager.get_task_size(
            task
        )
        self._class_increments.append(
            (self._known_classes, self._classes_seen_so_far - 1)
        )
        self.cur_task = task

    def after_task(self):
        self.model.head.update(
            self._classes_seen_so_far - self._known_classes, freeze_old=True
        )
        self.model.head.to(device)
        self._known_classes = self._classes_seen_so_far

    def eval(self, test_loader):
        y_pred, y_true = [], []

        configure_model(self.model)
        check_model(self.model)
        params, param_names = collect_params(self.model)
        # print(param_names)
        optimizer = optim.SGD(params, lr=1e-2)
        tented_model = Tent(self.model, optimizer, steps=1, episodic=True)

        # with torch.no_grad():
        for _, (_, x, y) in tqdm(enumerate(test_loader)):
            x, y = x.cuda(), y.cuda()

            # logits = self.model(x)
            logits = tented_model(x)
            predicts = torch.topk(logits, k=1, dim=1, largest=True, sorted=True)[1]

            y_pred.append(predicts.cpu().numpy())
            y_true.append(y.cpu().numpy())
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        acc_total, grouped = accuracy(y_pred.T[0], y_true, self._class_increments)
        logger.info(f"Total Acc: {acc_total:.2f}, Grouped Acc: {grouped}")

        self.accuracy_matrix.append(grouped)

        num_tasks = len(self.accuracy_matrix)
        accuracy_matrix = np.zeros((num_tasks, num_tasks))
        for i in range(num_tasks):
            for j in range(i + 1):
                accuracy_matrix[i, j] = self.accuracy_matrix[i][j]

        faa, ffm = compute_metrics(accuracy_matrix)
        logger.info(f"Final Average Accuracy (FAA): {faa:.2f}")
        logger.info(f"Final Forgetting Measure (FFM): {ffm:.2f}")

    def finetune(self, train_loader):
        # last_task_backbone = timm.create_model(pretrained_backbone, pretrained=True, num_classes=0).cuda()
        # for name, param in last_task_backbone.named_parameters():
        #     if "ssf_scale_" in name:
        #         nn.init.ones_(param)
        #     elif "ssf_shift_" in name:
        #         nn.init.zeros_(param)

        # # last_task_backbone.load_state_dict(copy.deepcopy(self.model.backbone.state_dict()))
        # last_task_backbone.eval()

        # if self.cur_task > 0:
        #     self.model.backbone.requires_grad_(False)

        base_lr = 1e-2
        backbone_lr = 1
        head_lr = 1
        weight_decay = 5e-4
        min_lr = 0.0
        params = [
            {"params": self.model.backbone.parameters(), "lr": base_lr * backbone_lr},
            {"params": self.model.head.parameters(), "lr": base_lr * head_lr},
        ]
        optimizer = optim.SGD(
            self.model.parameters(), lr=1e-2, momentum=0.9, weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["fine_tune_train_epochs"], eta_min=min_lr
        )

        pbar = tqdm(range(config["fine_tune_train_epochs"]))
        for _, epoch in enumerate(pbar):
            self.model.train()
            total_loss, total, correct = 0, 0, 0
            total_cls_loss, total_emb_loss = 0, 0

            for i, (_, x, y) in enumerate(train_loader):
                x, y = x.cuda(), y.cuda()
                y = torch.where(
                    y - self._known_classes >= 0, y - self._known_classes, -100
                )
                logits = self.model(x)[:, -20:]

                # cur_repr = self.model.backbone(x)
                # last_repr = last_task_backbone(x)

                cls_loss = F.cross_entropy(logits, y)

                # emb_loss = F.cosine_embedding_loss(F.normalize(cur_repr), F.normalize(last_repr), torch.ones(len(x)).cuda())
                # emb_loss = F.mse_loss(F.normalize(cur_repr), F.normalize(last_repr))
                # emb_loss = kl_divergence_loss(F.normalize(cur_repr), F.normalize(last_repr), 0.7)
                emb_loss = torch.tensor(0.0).cuda()

                loss = cls_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                correct += (logits.argmax(dim=1) == y).sum().item()
                total += len(y)
                total_cls_loss += cls_loss.item()
                total_emb_loss += emb_loss.item()

                info = f"Epoch {epoch}, Samples: {total * 100 / len(train_loader.dataset):.2f}, Loss: {total_loss / total:.4f}, Cls loss: {total_cls_loss / total:.4f}, Emb loss: {total_emb_loss / total:.4f}, Acc: {correct * 100 / total:.2f}"
                pbar.set_description(info)

            scheduler.step()
        logger.info(f"Train accuracy: {correct * 100 / total:.2f}")

        # del last_task_backbone
        # gc.collect()


dataset_name = "cub"
logger.info(f"pretrained_backbone: {pretrained_backbone}")
logger.info(f"dataset_name: {dataset_name}")

for seed in [1993]:
    logger.info(f"Seed: {seed}")
    set_random(seed)

    num_classes = 200
    num_init_cls = 20
    data_manager = DataManager(dataset_name, True, seed, 20, 20, False)

    for epoch in [5]:
        config.update(
            {
                "fine_tune_train_batch_size": 64,
                "fine_tune_train_epochs": epoch,
            }
        )
        logger.info(f"{' | '.join('%s: %s' % item for item in config.items())}")

        learner = Learner()
        learner.learn(data_manager)
