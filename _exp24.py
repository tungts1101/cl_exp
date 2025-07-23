import copy
import logging
import numpy as np
import os
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from petl import vision_transformer_ssf

from utils.toolkit import target2onehot, tensor2numpy, accuracy
import timm
import math

num_workers = 4

class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, nb_proxy=1, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features * nb_proxy
        self.nb_proxy = nb_proxy
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()
        self.W_rand = None
        self.use_RP=False

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input):
        if self.W_rand is not None:
            inn = torch.nn.functional.relu(input @ self.W_rand)
        else:
            inn=input
        
        out = F.linear(inn,self.weight)
            
        if self.sigma is not None:
            out = self.sigma * out

        return {'logits': out}


class SimpleVitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.convnet = timm.create_model("vit_base_patch16_224_ssf", pretrained=True, num_classes=0)
        self.convnet.out_dim=768
        self.convnet.eval()
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim
    
    def update_fc(self, nb_classes):
        fc = CosineLinear(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x)
        return out
    
    
class BaseLearner(object):
    def __init__(self):
        self._cur_task = -1
        self._known_classes = 0
        self._classes_seen_so_far = 0
        self.class_increments = []
        self._network = None
        self._device = 'cuda:0'

    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        acc_total, grouped = self._evaluate(y_pred, y_true)
        return acc_total, grouped, y_pred[:, 0], y_true

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.topk(outputs, k=1, dim=1, largest=True, sorted=True)[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)

    def _evaluate(self, y_pred, y_true):
        ret = {}
        acc_total, grouped = accuracy(
            y_pred.T[0], y_true, self._known_classes, self.class_increments
        )
        return acc_total, grouped

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)


class Learner(BaseLearner):
    def __init__(self):
        super().__init__()
        self._network = SimpleVitNet()
        self._batch_size = 48

        self.weight_decay = 5e-4
        self.min_lr = 0.0

    def after_task(self):
        self._known_classes = self._classes_seen_so_far

    def replace_fc(self, trainloader):
        self._network = self._network.eval()

        self._network.fc.use_RP = True
        self._network.fc.W_rand = self.W_rand

        Features_f = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_, data, label) = batch
                data = data.cuda()
                label = label.cuda()
                embedding = self._network.convnet(data)
                Features_f.append(embedding.cpu())
                label_list.append(label.cpu())
        Features_f = torch.cat(Features_f, dim=0)
        label_list = torch.cat(label_list, dim=0)

        Y = target2onehot(label_list, self.total_classnum)
        Features_h = torch.nn.functional.relu(
            Features_f @ self._network.fc.W_rand.cpu()
        )
        self.Q = self.Q + Features_h.T @ Y
        self.G = self.G + Features_h.T @ Features_h
        ridge = self.optimise_ridge_parameter(Features_h, Y)
        Wo = torch.linalg.solve(
            self.G + ridge * torch.eye(self.G.size(dim=0)), self.Q
        ).T  # better nmerical stability than .inv
        
        self._network.fc.weight.data = Wo[
            0 : self._network.fc.weight.shape[0], :
        ].to(device="cuda")

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
        logging.info("Optimal lambda: " + str(ridge))
        return ridge

    def incremental_train(self, data_manager):
        self.total_classnum = data_manager.get_total_classnum()
        self._cur_task += 1
        self._classes_seen_so_far = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        
        del self._network.fc
        self._network.fc = None
        self._network.update_fc(
            self._classes_seen_so_far
        )  # creates a new head with a new number of classes (if CIL)
        
        logging.info("Starting CIL Task {}".format(self._cur_task + 1))
        logging.info(
            "Learning on classes {}-{}".format(
                self._known_classes, self._classes_seen_so_far - 1
            )
        )
        self.class_increments.append(
            [self._known_classes, self._classes_seen_so_far - 1]
        )
        self.train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._classes_seen_so_far),
            source="train",
            mode="train",
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        train_dataset_for_CPs = data_manager.get_dataset(
            np.arange(self._known_classes, self._classes_seen_so_far),
            source="train",
            mode="test",
        )
        self.train_loader_for_CPs = DataLoader(
            train_dataset_for_CPs,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._classes_seen_so_far), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        self._train(self.train_loader, self.test_loader, self.train_loader_for_CPs)

    def freeze_backbone(self, is_first_session=False):
        if isinstance(self._network.convnet, nn.Module):
            for name, param in self._network.convnet.named_parameters():
                if is_first_session:
                    if "ssf_scale" not in name and "ssf_shift_" not in name:
                        param.requires_grad = False
                else:
                    param.requires_grad = False

    def show_num_params(self, verbose=False):
        # show total parameters and trainable parameters
        total_params = sum(p.numel() for p in self._network.parameters())
        logging.info(f"{total_params:,} total parameters.")
        total_trainable_params = sum(
            p.numel() for p in self._network.parameters() if p.requires_grad
        )
        logging.info(f"{total_trainable_params:,} training parameters.")
        if total_params != total_trainable_params and verbose:
            for name, param in self._network.named_parameters():
                if param.requires_grad:
                    print(name, param.numel())

    def _train(self, train_loader, test_loader, train_loader_for_CPs):
        self._network.to(self._device)
            # this branch is either CP updates only, or SGD on a PETL method first task only
        if self._cur_task == 0:
            self.freeze_backbone(is_first_session=True)
            
            # this will be a PETL method. Here, 'body_lr' means all parameters
            self.show_num_params()
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=0.01,
                weight_decay=self.weight_decay,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=20, eta_min=self.min_lr
            )
            
            self._init_train(train_loader, test_loader, optimizer, scheduler)
            self.freeze_backbone()
            
            self.setup_RP()
            
        self.replace_fc(train_loader_for_CPs)
        self.show_num_params()

    def setup_RP(self):
        self._network.fc.use_RP = True
        
        M = 10000
        self._network.fc.weight = nn.Parameter(
            torch.Tensor(self._network.fc.out_features, M).to(device="cuda")
        )  # num classes in task x M
        self._network.fc.reset_parameters()
        self._network.fc.W_rand = torch.randn(self._network.fc.in_features, M).to(
            device="cuda"
        )
        self.W_rand = copy.deepcopy(
            self._network.fc.W_rand
        )  # make a copy that gets passed each time the head is replaced
            
        self.Q = torch.zeros(M, self.total_classnum)
        self.G = torch.zeros(M, M)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(1))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                20,
                losses / len(train_loader),
                train_acc,
                test_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)


import gc
import sys
from util import set_random
from utils.data_manager import DataManager
from utils.toolkit import count_parameters

def train():
    gc.collect()
    torch.cuda.empty_cache()

    logs_name = "logs/tmp"
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)
    logfilename = "logs/tmp/tmp"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info("Starting new run")
    
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = Learner()
    data_manager = DataManager("cub", True, 1993, 20, 20, False)
    num_tasks = data_manager.nb_tasks

    acc_curve = {"top1_total": [], "ave_acc": []}
    logging.info(
        "Pre-trained network parameters: {}".format(count_parameters(model._network))
    )
    for task in range(num_tasks):
        model.incremental_train(data_manager)
        acc_total, acc_grouped, predicted_classes, true_classes = model.eval_task()
        model.after_task()

        acc_curve["top1_total"].append(acc_total)
        acc_curve["ave_acc"].append(np.round(np.mean(list(acc_grouped.values())), 2))

        logging.info("Group Accuracies after this task: {}".format(acc_grouped))
        logging.info("Ave Acc curve: {}".format(acc_curve["ave_acc"]))
        logging.info("Top1 curve: {}".format(acc_curve["top1_total"]))

    logging.info("Finishing run")
    logging.info("")
    return acc_curve["ave_acc"]

train()