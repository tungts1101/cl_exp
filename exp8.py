import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm
from petl import vision_transformer_ssf  # register vision_transformer_ssf
import numpy as np
from tqdm import tqdm
import random
from utils.data_manager import DataManager
from utils.toolkit import accuracy


seed = 1995
num_classes = 100
num_init_cls = 10
dataset = "cifar224"
M = 10000

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

data_manager = DataManager(dataset, True, seed, 0, num_init_cls, False)


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.base = timm.create_model(
            "vit_base_patch16_224_ssf", pretrained=True, num_classes=0
        )
        # self.base.norm = nn.Identity()
        self.freeze_backbone()
        
        self.num_layers = 1

        # self.norm = nn.BatchNorm1d(768, affine=True).cuda()
        # self.norm.weight.data.fill_(1)
        # self.norm.bias.data.fill_(0)

        # self.norm = nn.LayerNorm(768 * self.num_layers, elementwise_affine=True).cuda()

        self.fc = nn.Linear(768, num_init_cls, bias=False).cuda()
        self.fc.requires_grad_(False)
        self.fc.weight.data = torch.normal(0, 1, (num_init_cls, 768)).cuda()

    def configure(self):
        self.base.requires_grad_(False)
        self.fc = None

        self.Win = torch.normal(0, 1, (768 * self.num_layers, M)).cuda()
        self.Wout = torch.zeros(M, num_classes).cuda()

        self.features = []

        def get_features(model, inut, output):
            self.features.append(output)

        for name, module in self.base.named_modules():
            if isinstance(
                module, timm.models.vision_transformer.VisionTransformer
            ) or isinstance(module, vision_transformer_ssf.VisionTransformer):
                for i, block in enumerate(module.blocks):
                    block.register_forward_hook(get_features)

        self.norm.requires_grad_(True)
        # self.norm.track_running_stats = False
        # self.norm.running_mean = None
        # self.norm.running_var = None
        self.norm_optimizer = optim.SGD(self.norm.parameters(), lr=1e-3)

    def get_features(self, x):
        self.features = []
        self.base(x)

        # f = torch.cat(self.features[-self.num_layers :], dim=-1)
        # f = f[:, 0, :]
        f = self.features[-1][:, 0, :]

        return f

    def forward(self, x):
        f = self.get_features(x)
        # f = self.norm(f)

        f = f @ self.Win
        f = F.relu(f)

        y = f @ self.Wout
        return y

    def tune_forward(self, x):
        f = self.base(x)
        # f = self.norm(f)

        y = self.fc(f)
        return y

    def freeze_backbone(self):
        for name, param in self.base.named_parameters():
            if (
                "head." not in name
                and "ssf_scale" not in name
                and "ssf_shift_" not in name
            ):
                param.requires_grad = False

    def show_num_params(self, verbose=False):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total number of parameters: {total_params:,}")
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total number of trainable parameters: {trainable_params:,}")


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1).mean(0)


class Learner:
    def __init__(self):
        self.model = Net()
        self.model = self.model.cuda()

        self._known_classes = 0
        self._classes_seen_so_far = 0
        self.class_increments = []

        self.C = torch.zeros(M, num_classes)
        self.Q = torch.zeros(M, M)
        self.I = torch.eye(M)

    def train(self, datamanager):
        num_task = datamanager.nb_tasks - 1

        for task in range(num_task):
            self._classes_seen_so_far = self._known_classes + datamanager.get_task_size(
                task + 1
            )
            self.class_increments.append(
                (self._known_classes, self._classes_seen_so_far - 1)
            )

            print(
                f"Learn classes: {self._known_classes} - {self._classes_seen_so_far - 1}"
            )
            self.model.show_num_params()

            if task == 0:
                trainset = datamanager.get_dataset(
                    np.arange(self._known_classes, self._classes_seen_so_far),
                    source="train",
                    mode="train",
                )
                trainloader = DataLoader(
                    trainset, batch_size=32, shuffle=True, num_workers=4
                )
                testset = datamanager.get_dataset(
                    np.arange(self._known_classes, self._classes_seen_so_far),
                    source="test",
                    mode="test",
                )
                testloader = DataLoader(
                    testset, batch_size=128, shuffle=False, num_workers=4
                )

                self.tune(trainloader, testloader, self._known_classes)
                self.model.configure()

            trainset_CPs = datamanager.get_dataset(
                np.arange(self._known_classes, self._classes_seen_so_far),
                source="train",
                mode="test",
            )
            trainloader_CPs = DataLoader(
                trainset_CPs, batch_size=128, shuffle=False, num_workers=4
            )
            testset_CPs = datamanager.get_dataset(
                np.arange(0, self._classes_seen_so_far), source="test", mode="test"
            )
            testloader_CPs = DataLoader(
                testset_CPs, batch_size=128, shuffle=False, num_workers=4
            )
            self.fit(trainloader_CPs, testloader_CPs)

            # after task
            self._known_classes = self._classes_seen_so_far

    def fit(self, trainloader, testloader):
        self.model.train()

        fs = []
        ys = []
        for i, (_, x, y, _) in tqdm(enumerate(trainloader)):
            x, y = x.cuda(), y.cuda()
            with torch.no_grad():
                f = self.model.get_features(x)
                f = self.model.norm(f)

            fs.append(f.cpu())
            ys.append(y.cpu())

        fs = torch.cat(fs, dim=0)
        ys = torch.cat(ys, dim=0)

        H = F.relu(fs @ self.model.Win.cpu())
        Y = F.one_hot(ys, num_classes=num_classes).float()

        self.Q = self.Q + H.T @ H
        self.C = self.C + H.T @ Y
        ridge = 1e6
        Wout = torch.linalg.solve(self.Q + ridge * self.I, self.C)
        self.model.Wout.data = Wout.cuda()

        y_pred, y_true = [], []
        for _, (_, x, y, _) in tqdm(enumerate(testloader)):
            x = x.cuda()

            outputs = self.model(x)

            loss = softmax_entropy(outputs)
            loss.backward()
            self.model.norm_optimizer.step()
            self.model.norm_optimizer.zero_grad()

            predicts = torch.topk(outputs, k=1, dim=1, largest=True, sorted=True)[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(y.cpu().numpy())
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        acc_total, grouped = accuracy(
            y_pred.T[0], y_true, self._known_classes, self.class_increments
        )
        print(f"Acc total: {acc_total}, Acc grouped: {grouped}")

    def tune(self, trainloader, testloader, starting_label):
        tune_epochs = 20
        body_lr = 1e-2
        weight_decay = 5e-4
        min_lr = 0.0

        optimizer = optim.SGD(
            self.model.parameters(), lr=body_lr, momentum=0.9, weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tune_epochs, eta_min=min_lr
        )

        pbar = tqdm(range(tune_epochs))
        for _, epoch in enumerate(pbar):
            self.model.train()
            losses = 0.0
            train_acc = 0
            for i, (_, x, y, a) in enumerate(trainloader):
                x, y, a = x.cuda(), y.cuda(), a.cuda()
                logits = self.model.tune_forward(x)
                loss = F.cross_entropy(F.softmax(logits, dim=1), y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()
                correct = (logits.argmax(dim=1) == y).sum().item()
                train_acc += correct
            scheduler.step()
            train_acc = np.around(
                train_acc * 100 / len(trainloader.dataset), decimals=2
            )

            self.model.eval()
            test_acc = 0
            for i, (_, x, y, _) in enumerate(testloader):
                x, y = x.cuda(), y.cuda()

                with torch.no_grad():
                    logits = self.model.tune_forward(x)
                correct = (logits.argmax(dim=1) == y).sum().item()
                test_acc += correct

            test_acc = np.around(test_acc * 100 / len(testloader.dataset), decimals=2)

            info = "Loss {:.3f}, Train_acc {:.2f}, Test_acc {:.2f}".format(
                losses / len(trainloader.dataset),
                train_acc,
                test_acc,
            )
            pbar.set_description(info)


learner = Learner()
learner.train(data_manager)
