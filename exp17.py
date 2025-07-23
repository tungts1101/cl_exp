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


seed = 1993

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dataset = "imagenetr"
num_init_cls = 20
# dataset = "cifar224"
# num_init_cls = 10
data_manager = DataManager(dataset, True, seed, 0, num_init_cls, False)
num_classes = data_manager.get_total_classnum()


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # config = ViTConfig.from_pretrained(
        #     "google/vit-base-patch16-224", output_attentions=True
        # )
        # self.base = ViTModel(config)
        # self.base.requires_grad_(False)
        
        self.base = timm.create_model(
            "vit_base_patch16_224_ssf", pretrained=True, num_classes=0
        )
        self.freeze_backbone()
        self.features = []

        def get_features(model, inut, output):
            self.features.append(output)

        for name, module in self.base.named_modules():
            if isinstance(
                module, timm.models.vision_transformer.VisionTransformer
            ) or isinstance(module, vision_transformer_ssf.VisionTransformer):
                for i, block in enumerate(module.blocks):
                    block.register_forward_hook(get_features)

        self.conv_head = nn.Sequential(
            nn.Conv2d(768, num_init_cls, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.conv_head.apply(init_weights)

        self.linear_head = nn.Sequential(
            nn.Linear(768, num_init_cls),
        )
        self.linear_head.apply(init_weights)
        
        self.Win = nn.Parameter(torch.randn(768, 10000))
        self.Wout = nn.Parameter(torch.randn(10000, num_classes))
    
    def get_features(self, x):
        self.features = []
        self.base(x)
        return self.features[-1]
    
    def tune_forward(self, x):
        f = self.get_features(x)
        f_conv, f_linear = f[:, 1:], f[:, 0]
        
        B, P, E = f_conv.shape
        f_conv = torch.reshape(f_conv, [B, int(P**0.5), int(P**0.5), E])
        f_conv = f_conv.permute([0, 3, 1, 2])
        f_conv = f_conv.contiguous()
        
        y_conv = self.conv_head(f_conv)
        y_linear = self.linear_head(f_linear)
        y = y_linear + y_conv
        return y

    def forward(self, x):
        f = self.get_features(x)
        # f = torch.mean(f, dim=1)
        f = f[:, 0]
        f = f @ self.Win
        f = F.relu(f)
        y = f @ self.Wout
        
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


class Learner:
    def __init__(self):
        self.model = Net()
        self.model = self.model.cuda()

        self._known_classes = 0
        self._classes_seen_so_far = 0
        self.class_increments = []
        
        self.C = torch.zeros(10000, num_classes)
        self.Q = torch.zeros(10000, 10000)
        self.I = torch.eye(10000)

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

                self.tune(trainloader)


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
        fs = []
        ys = []
        for i, (_, x, y) in tqdm(enumerate(trainloader)):
            x, y = x.cuda(), y.cuda()
            with torch.no_grad():
                f = self.model.get_features(x)
                # f = torch.mean(f, dim=1)
                f = f[:, 0]

            fs.append(f.cpu())
            ys.append(y.cpu())

        fs = torch.cat(fs, dim=0)
        ys = torch.cat(ys, dim=0)

        H = F.relu(fs @ self.model.Win.cpu())
        Y = F.one_hot(ys, num_classes=num_classes).float()

        self.Q = self.Q + H.T @ H
        self.C = self.C + H.T @ Y
        ridge = 1e4
        W = torch.linalg.solve(self.Q + ridge * self.I, self.C)
        self.model.Wout.data = W.cuda()

        y_pred, y_true = [], []
        for _, (_, x, y) in tqdm(enumerate(testloader)):
            x = x.cuda()
            with torch.no_grad():
                outputs = self.model(x)

            predicts = torch.topk(outputs, k=1, dim=1, largest=True, sorted=True)[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(y.cpu().numpy())
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        acc_total, grouped = accuracy(
            y_pred.T[0], y_true, self._known_classes, self.class_increments
        )
        print(f"Acc total: {acc_total}, Acc grouped: {grouped}")

    def tune(self, trainloader):
        self.model.train()

        tune_epochs = 20
        weight_decay = 5e-4
        min_lr = 0.0

        optimizer = optim.SGD(
            [
                {"params": self.model.base.parameters(), "lr": 1e-2},
                {"params": self.model.conv_head.parameters(), "lr": 1e-2},
                {"params": self.model.linear_head.parameters(), "lr": 1e-2},
            ],
            momentum=0.9,
            weight_decay=weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tune_epochs, eta_min=min_lr
        )

        pbar = tqdm(range(tune_epochs))
        for _, epoch in enumerate(pbar):
            self.model.train()
            losses = 0.0
            train_acc = 0
            for i, (_, x, y) in enumerate(trainloader):
                x, y = x.cuda(), y.cuda()

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

            info = "Loss {:.3f}, Train_acc {:.2f}".format(
                losses / len(trainloader.dataset), train_acc
            )
            pbar.set_description(info)


learner = Learner()
learner.train(data_manager)
