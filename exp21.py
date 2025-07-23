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


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.base = timm.create_model(
            "vit_base_patch16_224_ssf", pretrained=True, num_classes=0
        )
        self.freeze_base()
        self.fc = nn.Sequential(
            nn.Linear(self.base.embed_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, num_classes),
        )
        self.fc.apply(init_weights)
        # self.fc = None
    
    def freeze_base(self):
        for name, param in self.base.named_parameters():
            if (
                "head." not in name
                and "ssf_scale" not in name
                and "ssf_shift_" not in name
            ):
                param.requires_grad = False
    
    def update(self, num_classes):
        # fc = nn.Linear(self.base.embed_dim, num_classes, bias=False).cuda()
        # if self.fc is not None:
        #     fc.weight.data[: self.fc.weight.data.shape[0]] = self.fc.weight.data
        # self.fc = fc  
        pass   

    def forward(self, x):
        f = self.base(x)
        y = self.fc(f)
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

        self.clazz = torch.zeros(num_classes, 768).cuda()

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

            trainset = datamanager.get_dataset(
                np.arange(self._known_classes, self._classes_seen_so_far),
                source="train",
                mode="train",
            )
            trainloader = DataLoader(
                trainset, batch_size=32, shuffle=True, num_workers=4
            )

            self.tune(trainloader)
                
            testset_CPs = datamanager.get_dataset(
                np.arange(0, self._classes_seen_so_far), source="test", mode="test"
            )
            testloader_CPs = DataLoader(
                testset_CPs, batch_size=128, shuffle=False, num_workers=4
            )
            self.fit(testloader_CPs)

            # after task
            self._known_classes = self._classes_seen_so_far

    def fit(self, testloader):
        self.model.eval()

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
        self.model.update(self._classes_seen_so_far)
        self.model.train()

        tune_epochs = 10
        optimizer = optim.SGD(
            self.model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4
        )
        # optimizer = optim.Adam(self.model.parameters(), lr=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tune_epochs, eta_min=0.0
        )

        pbar = tqdm(range(tune_epochs))

        for _, epoch in enumerate(pbar):
            self.model.train()
            losses = 0.0
            train_acc, total = 0, 0
            for i, (_, x, y, a) in enumerate(trainloader):
                x, y, a = x.cuda(), y.cuda(), a.cuda()
                
                x_embed = self.model.base(x)
                a_embed = self.model.base(a)
                logits = self.model.fc(x_embed)
                
                supervised_loss = F.cross_entropy(F.softmax(logits, dim=1), y)
                self_supervised_loss = info_nce_loss(x_embed, a_embed, self.clazz)

                loss = 0.5 * supervised_loss + 0.5 * self_supervised_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()
                correct = (logits.argmax(dim=1) == y).sum().item()
                train_acc += correct
                total += len(y)

                info = "Loss {:.3f}, SL {:.3f}, SSL {:.3f}, Train acc {:.2f}".format(
                    loss.item(), supervised_loss.item(), self_supervised_loss.item(),
                    np.around(train_acc * 100 / total, decimals=2),
                )
                pbar.set_description(info)

            scheduler.step()
            
        fs = []
        ys = []
        for _, (_, x, y, _) in tqdm(enumerate(trainloader)):
            x = x.cuda()
            with torch.no_grad():
                f = self.model.base(x)
            fs.append(f.cpu())
            ys.append(y.cpu())
        
        fs = torch.cat(fs, dim=0)
        ys = torch.cat(ys, dim=0)
        print(np.unique(ys))
        for class_index in np.unique(ys):
            data_index = (ys == class_index).nonzero().squeeze(-1)
            class_prototype = fs[data_index].mean(0)
            self.clazz[class_index] = class_prototype.cuda()


def info_nce_loss(queries, positives, negatives, temperature=0.7):
    queries = F.normalize(queries, dim=-1)
    positives = F.normalize(positives, dim=-1)
    negatives = F.normalize(negatives, dim=-1)
    negatives = negatives.unsqueeze(0).expand(queries.shape[0], -1, -1)
    
    positive_scores = torch.einsum("ij,ij->i", queries, positives) / temperature
    negative_scores = torch.einsum("ij,ikj->ik", queries, negatives) / temperature
    
    logits = torch.cat([positive_scores.unsqueeze(1), negative_scores], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
    
    return F.cross_entropy(logits, labels)


learner = Learner()
learner.train(data_manager)
