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
projection_dim = int(1e4)

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def info_nce_loss(query, key, temperature=0.07):
    """
    Compute InfoNCE loss for a batch of embeddings.

    Args:
        query (torch.Tensor): Tensor of shape (batch_size, embedding_dim), representing anchor embeddings.
        key (torch.Tensor): Tensor of shape (batch_size, embedding_dim), representing positive pair embeddings.
        temperature (float): Temperature scaling parameter.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Normalize embeddings to unit vectors
    query = F.normalize(query, dim=1)
    key = F.normalize(key, dim=1)

    # Compute similarity matrix (dot product of query and key)
    logits = torch.mm(query, key.t())  # Shape: (batch_size, batch_size)

    # Scale logits by temperature
    logits /= temperature

    # Create labels (diagonal are positives)
    batch_size = query.size(0)
    labels = torch.arange(batch_size).to(query.device)

    # Compute cross-entropy loss
    loss = F.cross_entropy(logits, labels)

    return loss

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.base = timm.create_model(
            "vit_base_patch16_224_ssf", pretrained=True, num_classes=0
        )
        self.freeze_backbone()
        # self.base = timm.create_model(
        #     "vit_base_patch16_224", pretrained=True, num_classes=0
        # )
        self.fc = nn.Linear(self.base.embed_dim, num_init_cls, bias=False)
        # self.fc = nn.Sequential(
        #     nn.Linear(self.base.embed_dim, 512),
        #     nn.ReLU(),
        #     nn.LayerNorm(512),
        #     nn.Linear(512, num_init_cls),
        # )
        self.fc.apply(init_weights)

    def freeze_backbone(self):
        for name, param in self.base.named_parameters():
            if (
                "head." not in name
                and "ssf_scale" not in name
                and "ssf_shift_" not in name
            ):
                param.requires_grad = False

    def update(self):
        self.base.requires_grad_(False)
        self.fc = None
        self.Wrand = nn.Parameter(torch.randn(self.base.embed_dim, projection_dim)).cuda()
        self.W = nn.Parameter(torch.zeros(projection_dim, num_classes)).cuda()
    
    def get_features(self, x):
        f = self.base(x)
        return f

    def tune_forward(self, x):
        f = self.get_features(x)
        y = self.fc(f)
        return y

    def forward(self, x):
        f = self.get_features(x)
        f = F.relu(f @ self.Wrand)
        y = f @ self.W
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

        self.C = torch.zeros(projection_dim, num_classes)
        self.Q = torch.zeros(projection_dim, projection_dim)
        self.I = torch.eye(projection_dim)

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
                self.model.update()
            
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
        self.model.eval()
        
        fs = []
        ys = []
        for i, (_, x, y, a) in tqdm(enumerate(trainloader)):
            x, y, a = x.cuda(), y.cuda(), a.cuda()
            with torch.no_grad():
                f = self.model.get_features(x)
                fa = self.model.get_features(a)

            fs.append(f.cpu())
            fs.append(fa.cpu())
            ys.append(y.cpu())
            ys.append(y.cpu())

        fs = torch.cat(fs, dim=0)
        ys = torch.cat(ys, dim=0)

        H = F.relu(fs @ self.model.Wrand.cpu())
        Y = F.one_hot(ys, num_classes=num_classes).float()

        self.Q = self.Q + H.T @ H
        self.C = self.C + H.T @ Y
        ridge = 1e4
        W = torch.linalg.solve(self.Q + ridge * self.I, self.C)
        self.model.W.data = W.cuda()

        y_pred, y_true = [], []
        for _, (_, x, y, _) in tqdm(enumerate(testloader)):
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
        # optimizer = optim.SGD(
        #     self.model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4
        # )
        optimizer = optim.Adam(self.model.parameters(), lr=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tune_epochs, eta_min=0.0
        )

        pbar = tqdm(range(tune_epochs))

        for _, epoch in enumerate(pbar):
            self.model.train()
            losses = 0.0
            train_acc = 0
            for i, (_, x, y, a) in enumerate(trainloader):
                x, y, a = x.cuda(), y.cuda(), a.cuda()
                
                x_embed = self.model.base(x)
                a_embed = self.model.base(a)
                logits = self.model.fc(x_embed)
                
                supervised_loss = F.cross_entropy(F.softmax(logits, dim=1), y)
                self_supervised_loss = info_nce_loss(x_embed, a_embed)

                # print(supervised_loss.item(), self_supervised_loss.item())
                loss = 0.5 * supervised_loss + 0.5 * self_supervised_loss

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
                losses / len(trainloader.dataset),
                train_acc,
            )
            pbar.set_description(info)


learner = Learner()
learner.train(data_manager)
