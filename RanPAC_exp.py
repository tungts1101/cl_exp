import copy
import logging
import numpy as np
import os
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import gc

from inc_net import ResNetCosineIncrementalNet, SimpleVitNet
from utils.toolkit import target2onehot, tensor2numpy, accuracy
from merging.task_vectors import TaskVector, merge_max_abs
from merging.ties import merge_methods, state_dict_to_vector, vector_to_state_dict

num_workers = 4

class BaseLearner(object):
    def __init__(self, args):
        self._cur_task = -1
        self._known_classes = 0
        self._classes_seen_so_far = 0
        self.class_increments = []
        self._network = None

        self._device = args["device"][0]
        self._multiple_gpus = args["device"]

    def eval_task(self):
        assert self._network.fc.use_RP
        
        y_pred, y_true = self._eval_cnn(self.test_loader)
        acc_total, grouped = self._evaluate(y_pred, y_true)
        return acc_total, grouped, y_pred[:, 0], y_true

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, _, inputs, targets) in enumerate(loader):
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
        for i, (_, _, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

os.makedirs("/media/ellen/HardDisk/cl/logs/checkpoints", exist_ok=True) 

def backbone_base():
    return "/media/ellen/HardDisk/cl/logs/checkpoints/imagenetr_backbone_base.pt"
   
def backbone_checkpoint(task):
    return f"/media/ellen/HardDisk/cl/logs/checkpoints/imagenetr_backbone_{task}.pt"

def backbone_first_session():
    return "/media/ellen/HardDisk/cl/logs/checkpoints/backbone_first_session.pt"

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        if args["model_name"] != "ncm":
            if (
                args["model_name"] == "adapter"
                and "_adapter" not in args["convnet_type"]
            ):
                raise NotImplementedError("Adapter requires Adapter backbone")
            if args["model_name"] == "ssf" and "_ssf" not in args["convnet_type"]:
                raise NotImplementedError("SSF requires SSF backbone")
            if args["model_name"] == "vpt" and "_vpt" not in args["convnet_type"]:
                raise NotImplementedError("VPT requires VPT backbone")

            if "resnet" in args["convnet_type"]:
                self._network = ResNetCosineIncrementalNet(args, True)
                self._batch_size = 128
            else:
                self._network = SimpleVitNet(args, True)
                self._batch_size = args["batch_size"]

            self.weight_decay = (
                args["weight_decay"] if args["weight_decay"] is not None else 0.0005
            )
            self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        else:
            self._network = SimpleVitNet(args, True)
            self._batch_size = args["batch_size"]
        self.args = args
        
        torch.save(self._network.get_backbone_trainable_params(), backbone_base())

    def after_task(self):
        self._known_classes = self._classes_seen_so_far

    def replace_fc(self, trainloader):
        self._network = self._network.eval()

        if self.args["use_RP"]:
            # these lines are needed because the CosineLinear head gets deleted between streams and replaced by one with more classes (for CIL)
            self._network.fc.use_RP = True
            if self.args["M"] > 0:
                self._network.fc.W_rand = self.W_rand
            else:
                self._network.fc.W_rand = None

        Features_f = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_, _, data, label) = batch
                data = data.cuda()
                label = label.cuda()
                embedding = self._network.convnet(data)
                Features_f.append(embedding.cpu())
                label_list.append(label.cpu())
        Features_f = torch.cat(Features_f, dim=0)
        label_list = torch.cat(label_list, dim=0)

        Y = target2onehot(label_list, self.total_classnum)
        if self.args["use_RP"]:
            # print('Number of pre-trained feature dimensions = ',Features_f.shape[-1])
            if self.args["M"] > 0:
                Features_h = torch.nn.functional.relu(
                    Features_f @ self._network.fc.W_rand.cpu()
                )
            else:
                Features_h = Features_f
            self.Q = self.Q + Features_h.T @ Y
            self.G = self.G + Features_h.T @ Features_h
            ridge = self.optimise_ridge_parameter(Features_h, Y)
            Wo = torch.linalg.solve(
                self.G + ridge * torch.eye(self.G.size(dim=0)), self.Q
            ).T  # better nmerical stability than .inv
            self._network.fc.weight.data = Wo[
                0 : self._network.fc.weight.shape[0], :
            ].to(device="cuda")
        else:
            for class_index in np.unique(self.train_dataset.labels):
                data_index = (label_list == class_index).nonzero().squeeze(-1)
                if self.is_dil:
                    class_prototype = Features_f[data_index].sum(0)
                    self._network.fc.weight.data[class_index] += class_prototype.to(
                        device="cuda"
                    )  # for dil, we update all classes in all tasks
                else:
                    # original cosine similarity approach of Zhou et al (2023)
                    class_prototype = Features_f[data_index].mean(0)
                    self._network.fc.weight.data[class_index] = (
                        class_prototype  # for cil, only new classes get updated
                    )

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
        if self.args["use_RP"]:
            # temporarily remove RP weights
            del self._network.fc
            self._network.fc = None
        self._network.update_fc(
            self._classes_seen_so_far
        )  # creates a new head with a new number of classes (if CIL)
        
        if self._cur_task == 0:
            self.setup_RP()
        
        if self.is_dil == False:
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
        if len(self._multiple_gpus) > 1:
            print("Multiple GPUs")
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.train_loader_for_CPs)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def setup_RP(self):
        self.initiated_G = False
        self._network.fc.use_RP = True
        if self.args["M"] > 0:
            # RP with M > 0
            M = self.args["M"]
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
        else:
            # no RP, only decorrelation
            M = self._network.fc.in_features  # this M is L in the paper
        self.Q = torch.zeros(M, self.total_classnum)
        self.G = torch.zeros(M, M)

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
    
    def _train(self, train_loader, train_loader_for_CPs):
        self._network.to(self._device)
        
        # if self._cur_task >= 0:
        #     self._init_train(train_loader)
        #     self.merge()
        
        self._network.convnet.load_state_dict(torch.load(backbone_first_session()), strict=True)
        self.replace_fc(train_loader_for_CPs)
    
    # def _init_train(self, train_loader):
    #     optimizer = optim.SGD(
    #         self._network.parameters(),
    #         momentum=0.9,
    #         lr=self.args["body_lr"],
    #         weight_decay=self.weight_decay,
    #     )
    #     scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #         optimizer, T_max=self.args["tuned_epoch"], eta_min=self.min_lr
    #     )
    
    #     prog_bar = tqdm(range(self.args["tuned_epoch"]))
    #     for _, epoch in enumerate(prog_bar):
    #         self._network.train()
    #         losses = 0.0
    #         correct, total = 0, 0
    #         for i, (_, _, inputs, targets) in enumerate(train_loader):
    #             inputs, targets = inputs.to(self._device), targets.to(self._device)
    #             logits = self._network(inputs)["logits"]
    #             loss = F.cross_entropy(logits, targets)
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #             losses += loss.item()
    #             _, preds = torch.max(logits, dim=1)
    #             correct += preds.eq(targets.expand_as(preds)).cpu().sum()
    #             total += len(targets)
    #         scheduler.step()
    #         train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
    #         info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
    #             self._cur_task,
    #             epoch + 1,
    #             self.args["tuned_epoch"],
    #             losses / len(train_loader),
    #             train_acc
    #         )
    #         prog_bar.set_description(info)

    #     logging.info(info)
    
    def _init_train(self, train_loader):
        task_network = SimpleVitNet(self.args, True).to(self._device)
        # if self._cur_task > 0:
        #     # task_network.convnet.load_state_dict(torch.load(backbone_checkpoint(self._cur_task-1)), strict=False)
        #     task_network.convnet.load_state_dict(self._network.get_backbone_trainable_params(), strict=False)
        task_network.add_cls_head(self._classes_seen_so_far - self._known_classes)
        print(task_network.show_num_params())
        
        optimizer = optim.SGD(
            task_network.parameters(),
            momentum=0.9,
            lr=self.args["body_lr"],
            weight_decay=self.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.args["tuned_epoch"], eta_min=self.min_lr
        )
        
        prog_bar = tqdm(range(self.args["tuned_epoch"]))
        for _, epoch in enumerate(prog_bar):
            task_network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, _, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                targets = torch.where(targets >= self._known_classes, targets - self._known_classes, -100)
                logits = task_network.cls_forward(inputs)['logits']
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
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task + 1,
                epoch + 1,
                self.args["tuned_epoch"],
                losses / len(train_loader),
                train_acc
            )
            prog_bar.set_description(info)

        logging.info(info)
        
        task_network.cpu()
        torch.save(task_network.get_backbone_trainable_params(), backbone_checkpoint(self._cur_task))
        
        del task_network
        gc.collect()
        torch.cuda.empty_cache()
    
    def merge(self):
        task_vectors = [
            TaskVector(backbone_base(), backbone_checkpoint(task)) for task in range(self._cur_task + 1)]
        
        # reset_type = 'topk'
        # reset_thresh = 100
        # resolve = 'none'
        # merge = 'dis-mean'
        # tv_flat_checks = torch.vstack([state_dict_to_vector(tv.vector) for tv in task_vectors])
        # # print(f"\nMerging with TIES merging: pruning {reset_type}-{reset_thresh}, resolve sign by {resolve}, merge by {merge}")
        
        # merged_flat_tv = merge_methods(
        #     reset_type,
        #     tv_flat_checks,
        #     reset_thresh=reset_thresh,
        #     resolve_method=resolve,
        #     merge_func=merge,
        # )
        # merged_tv = vector_to_state_dict(
        #     merged_flat_tv, task_vectors[0].vector, remove_keys=[]
        # )
        # merged_tv = TaskVector(vector=merged_tv)
        
        merged_tv = merge_max_abs(task_vectors)
        
        backbone_params = merged_tv.apply_to(backbone_base(), scaling_coef=1.0)
        
        if self._cur_task > 0:
            self._network.cpu()
            prev_backbone = torch.cat([v.view(-1) for v in self._network.get_backbone_trainable_params().values()])
            curr_backbone = torch.cat([v.view(-1) for v in backbone_params.values()])
            
            # scale = torch.norm(curr_backbone) / torch.norm(prev_backbone)
            scale = (curr_backbone @ prev_backbone) / (prev_backbone @ prev_backbone + 1e-8)
            
            self.G *= scale.cpu()
        
        self._network.convnet.load_state_dict(backbone_params, strict=False)
        self._network.to(self._device)