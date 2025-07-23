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
data_manager = DataManager(dataset, True, seed, 0, num_init_cls, False)
num_classes = data_manager.get_total_classnum()
projection_dim = int(1e6)

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class PadPrompt(nn.Module):
    def __init__(self, target_shape, pad_size):
        super().__init__()
        self.target_shape = target_shape
        self.pad_size = pad_size
        
        self.pad = nn.Parameter(torch.randn(1, *target_shape))
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        x_resized = F.interpolate(x, size=(H - 2 * self.pad_size, W - 2 * self.pad_size), mode='bilinear')
        x_padded = F.pad(x_resized, (self.pad_size, self.pad_size, self.pad_size, self.pad_size), value=0)
        mask = torch.ones_like(x_resized)
        mask = F.pad(mask, (self.pad_size, self.pad_size, self.pad_size, self.pad_size), value=0)
        
        # pad = self.pad.expand(B, C, -1, -1)
        x_prompted = x_padded * mask + self.pad * (1 - mask)
        
        return x_prompted

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.base = timm.create_model("vit_base_patch16_224", pretrained=True)
        self.freeze_backbone()
        
        self.Wout = nn.Parameter(torch.randn(num_classes, 1000))
        self.prompt = PadPrompt(target_shape=(224, 224), pad_size=30)

    def freeze_backbone(self):
        for name, param in self.base.named_parameters():
            param.requires_grad = False
    
    def forward(self, x):
        x_prompt = self.prompt(x)
        f = self.base(x_prompt)
        y = F.linear(f, self.Wout)
        return y

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

        self.C = torch.zeros(projection_dim, 10)
        self.Q = torch.zeros(projection_dim, projection_dim)
        self.I = torch.eye(projection_dim)

    def train(self, datamanager):
        num_task = datamanager.nb_tasks - 1

        for task in range(num_task):
            self._classes_seen_so_far = self._known_classes + datamanager.get_task_size(task + 1)
            self.class_increments.append(self._known_classes, (self._classes_seen_so_far - 1))

            print(f"Learn classes: {self._known_classes} - {self._classes_seen_so_far - 1}")
            self.model.show_num_params()

            trainset = datamanager.get_dataset(
                np.arange(self._known_classes, self._classes_seen_so_far),
                source="train",
                mode="train",
            )
            trainloader = DataLoader(
                trainset, batch_size=32, shuffle=True, num_workers=4
            )
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
            self.fit(trainloader_CPs, trainloader, testloader_CPs)

            self._known_classes = self._classes_seen_so_far

    def fit(self, trainloader_CPs, trainloader, testloader_CPs):
        fs, ys = [], []
        for i, (_, x, y, _) in tqdm(enumerate(trainloader_CPs)):
            x, y = x.cuda(), y.cuda()
            with torch.no_grad():
                f = self.model.base(x)

            fs.append(f.cpu())
            ys.append(y.cpu())

        fs = torch.cat(fs, dim=0)
        ys = torch.cat(ys, dim=0)

        H = fs
        Y = F.one_hot(ys, num_classes=num_classes).float()

        self.Q = self.Q + H.T @ H
        self.C = self.C + H.T @ Y
        ridge = 1e2
        
        W = torch.linalg.solve(self.Q + ridge * self.I, self.C)
        self.model.Wout.data = W.T.cuda()
        
        self.model.train()
        epochs = 10
        optimizer = optim.SGD(
            self.model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=0.0
        )

        pbar = tqdm(range(epochs))
        for _, epoch in enumerate(pbar):
            self.model.train()
            train_acc, total = 0, 0
            for i, (_, x, y, _) in enumerate(trainloader):
                x, y = x.cuda(), y.cuda()
                
                logits = self.model(x)
                loss = F.cross_entropy(F.softmax(logits, dim=1), y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                correct = (logits.argmax(dim=1) == y).sum().item()
                train_acc += correct
                total += y.size(0)

                info = "Loss {:.3f}, Acc {:.2f}, y {}".format(
                    loss.item(),
                    np.around(train_acc * 100 / total, decimals=2),
                    np.unique(y.cpu().numpy()),
                )
                pbar.set_description(info)

            scheduler.step()
        
        
        self.model.eval()
        y_pred, y_true = [], []
        for _, (_, x, y, _) in tqdm(enumerate(testloader_CPs)):
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

learner = Learner()
learner.train(data_manager)
