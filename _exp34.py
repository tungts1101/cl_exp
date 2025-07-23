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
import copy
from utils.data_manager import DataManager
from utils.toolkit import count_parameters, target2onehot, tensor2numpy
from petl import vision_transformer_ssf
from _exp import ContinualLearnerHead, RandomProjectionHead, plot_heatmap, Buffer
from util import compute_metrics, accuracy, set_random
import math
from merging.task_vectors import TaskVector, merge_max_abs
from merging.ties import merge_methods, state_dict_to_vector, vector_to_state_dict
from test_adaptation import configure_model, check_model, collect_params, Tent
import gc


os.makedirs("logs/checkpoints", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

def setup_logger(log_file=f'logs/{timestamp}_exp34.log'):
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)
    
    logger.propagate = False

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_formatter = logging.Formatter('%(asctime)s - %(message)s')
    console_formatter = logging.Formatter('%(message)s')
    
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logger()

config = {
}
logger.info(f"{' | '.join('%s: %s' % item for item in config.items())}")

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # self.backbone = timm.create_model("vit_base_patch16_224_ssf", pretrained=True, num_classes=0)
        # self.freeze_backbone()
        # self.init_weights()
        
        self.backbone = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        self.freeze_backbone()
        self.apply_lora()
        
        self.head = ContinualLearnerHead(768, 20, with_norm=False)
    
    def apply_lora(self):
        """Wraps backbone with LoRA adapters."""
        lora_config = LoraConfig(
            r=16,  # LoRA rank
            lora_alpha=32,  # Scaling factor
            target_modules=["qkv", "fc2"],  # LoRA applied to attention and MLP layers
            lora_dropout=0.1,
            bias="none"
        )
        self.backbone = get_peft_model(self.backbone, lora_config)
        self.backbone.print_trainable_parameters()
    
    def init_weights(self):
        for name, param in self.backbone.named_parameters():
            if "ssf_scale_" in name:
                nn.init.ones_(param)
            elif "ssf_shift_" in name:
                nn.init.zeros_(param)
    
    def freeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            if "ssf_" not in name:
                param.requires_grad_(False)
    
    def get_backbone_trainable_params(self):
        return {name: param for name, param in self.backbone.named_parameters() if param.requires_grad}
    
    def forward(self, x):
        f = self.backbone(x)
        y = self.head(f)['logits']
        return y
    
    def __repr__(self):
        trainable_params = count_parameters(self, trainable=True)
        total_params = count_parameters(self)
        return f"Model(trainable_params={trainable_params}, total_params={total_params}, percentage={trainable_params * 100 / total_params:.2f})"


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
    

os.makedirs("/media/ellen/HardDisk/cl/logs/checkpoints", exist_ok=True) 
   
def ranpac_checkpoint():
    return "/media/ellen/HardDisk/cl/logs/checkpoints/imagenetr_ranpac.pt"

def backbone_checkpoint(task):
    return f"/media/ellen/HardDisk/cl/logs/checkpoints/1_exp_imagenetr_backbone_{task}.pt"

def head_checkpoint(task):
    return f"/media/ellen/HardDisk/cl/logs/checkpoints/1_exp_imagenetr_head_{task}.pt"

def backbone_base():
    return "/media/ellen/HardDisk/cl/logs/checkpoints/1_exp_imagenetr_backbone_base.pt"

def head_base():
    return "/media/ellen/HardDisk/cl/logs/checkpoints/1_exp_imagenetr_head_base.pt"

def head_merge(task):
    return f"/media/ellen/HardDisk/cl/logs/checkpoints/1_exp_imagenetr_head_merge_{task}.pt"

def trim(tensor, top_k=50):
    flattened = tensor.view(-1)
    magnitudes = torch.abs(flattened)
    num_keep = max(1, int(len(flattened) * top_k / 100))
    threshold = torch.topk(magnitudes, num_keep, largest=True, sorted=True).values[-1]
    mask = magnitudes >= threshold
    trimmed = torch.where(mask, flattened, torch.tensor(0.0, dtype=tensor.dtype))
    
    gamma = torch.sign(trimmed)
    mu = torch.abs(trimmed)
    
    return (trimmed.view_as(tensor), gamma.view_as(tensor), mu.view_as(tensor))

def merge(trimmed_task_vectors):
    gamma_tvs = torch.stack([tv[1] for tv in trimmed_task_vectors], dim=0)
    gamma = torch.sign(gamma_tvs.sum(dim=0))
    mask = (gamma_tvs == gamma)
    tau_tvs = torch.stack([tv[0] for tv in trimmed_task_vectors], dim=0)
    mean_tvs = torch.where(mask, tau_tvs, torch.tensor(0.0, dtype=tau_tvs.dtype)).sum(dim=0) / mask.sum(dim=0).clamp(min=1)

    return mean_tvs

def ties_merge(base_params, tasks_params, lamb=1.0, trim_top_k=100):
    params = {}
    for name in base_params:
        base_tv = copy.deepcopy(base_params[name])
        task_vectors = [copy.deepcopy(task_params[name]) for task_params in tasks_params]
        tvs = [tv - base_tv for tv in task_vectors]
        tvs = [trim(tv, trim_top_k) for tv in tvs]
        merged_tv = merge(tvs)
        params[name] = base_tv + lamb * merged_tv
        
    return params

class Learner:
    def __init__(self):
        self._known_classes = 0
        self._classes_seen_so_far = 0
        self._class_increments = []
        
        self.model = Model()
        torch.save(self.model.get_backbone_trainable_params(), backbone_base())
        self.model.cuda()
        
        self.accuracy_matrix = []
        self.cur_task = -1
        
        self._network = SimpleVitNet()
        
    def replace_fc(self, trainloader):
        self._network = self._network.eval()

        self._network.fc.use_RP = True
        self._network.fc.W_rand = self.W_rand
        
        Features_f = []
        label_list = []
        with torch.no_grad():
            for i, batch in tqdm(enumerate(trainloader)):
                (_, _, data, label) = batch
                data = data.cuda()
                label = label.cuda()
                embedding = self._network.convnet(data)
                Features_f.append(embedding.cpu())
                label_list.append(label.cpu())
        Features_f = torch.cat(Features_f, dim=0)
        label_list = torch.cat(label_list, dim=0)

        Y = F.one_hot(label_list, 200).float()
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
        print("Optimal lambda: " + str(ridge))
        return ridge
    
    def freeze_backbone(self, is_first_session=False):
        if isinstance(self._network.convnet, nn.Module):
            for name, param in self._network.convnet.named_parameters():
                if is_first_session:
                    if "ssf_" not in name:
                        param.requires_grad = False
                else:
                    param.requires_grad = False
    
    def setup_RP(self):
        self._network.fc.use_RP = True
        
        M = 10000
        self._network.fc.weight = nn.Parameter(
            torch.Tensor(self._network.fc.out_features, M).to(device="cuda")
        )  # num classes in task x M
        self._network.fc.reset_parameters()
        self.Q = torch.zeros(M, 200)
        self.G = torch.zeros(M, M)
           
        self._network.fc.W_rand = torch.randn(self._network.fc.in_features, M).cuda()
        self.W_rand = copy.deepcopy(self._network.fc.W_rand)
    
    def _init_train(self, train_loader, optimizer, scheduler):
        prog_bar = tqdm(range(20))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, _, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
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
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self.cur_task,
                epoch + 1,
                20,
                losses / len(train_loader),
                train_acc
            )
            prog_bar.set_description(info)
    
    def _train(self, train_loader, train_loader_for_CPs):
        self._network.cuda()
        if self.cur_task == 0:
            # if os.path.exists(ranpac_checkpoint()):
            #     self._network.load_state_dict(torch.load(ranpac_checkpoint()), strict=True)
            # else:
            self.freeze_backbone(is_first_session=True)
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=0.01,
                weight_decay=5e-4,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=20, eta_min=0.0
            )
            self._init_train(train_loader, optimizer, scheduler)
                
            self.freeze_backbone()
            self.setup_RP()
        self.replace_fc(train_loader_for_CPs)
    
    def learn(self, data_manager):
        self.data_manager = data_manager
        num_tasks = data_manager.nb_tasks

        for task in range(num_tasks):
            self.before_task(task, data_manager)
            
            trainset = data_manager.get_dataset(
                np.arange(self._known_classes, self._classes_seen_so_far), source="train", mode="train")
            train_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
            self.finetune(train_loader)
            
            # self.merge()
            
            # train_dataset_for_CPs = data_manager.get_dataset(
            #     np.arange(self._known_classes, self._classes_seen_so_far), source="train", mode="test")
            # train_loader_for_CPs = DataLoader(train_dataset_for_CPs, 64, shuffle=True, num_workers=4)
            # self._train(train_loader, train_loader_for_CPs)

            test_set = self.data_manager.get_dataset(
                np.arange(0, self._classes_seen_so_far), source="test", mode="test")
            test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4)
            self.eval(test_loader)
            
            self.after_task()

    def merge(self):
        # task_vectors = [
        #     TaskVector(backbone_base(), backbone_checkpoint(task)) for task in range(self.cur_task + 1)]
        
        # # reset_type = 'topk'
        # # reset_thresh = 20
        # # resolve = 'mass'
        # # merge = 'dis-mean'
        # # tv_flat_checks = torch.vstack([state_dict_to_vector(tv.vector) for tv in task_vectors])
        
        # # print(f"\nMerging with TIES merging: pruning {reset_type}-{reset_thresh}, resolve sign by {resolve}, merge by {merge}")
        
        # # merged_flat_tv = merge_methods(
        # #     reset_type,
        # #     tv_flat_checks,
        # #     reset_thresh=reset_thresh,
        # #     resolve_method=resolve,
        # #     merge_func=merge,
        # # )
        # # merged_tv = vector_to_state_dict(
        # #     merged_flat_tv, task_vectors[0].vector, remove_keys=[]
        # # )
        # # merged_tv = TaskVector(vector=merged_tv)
        
        # merged_tv = merge_max_abs(task_vectors)
        
        # merged_backbone = merged_tv.apply_to(backbone_base(), scaling_coef=0.5)
        
        # self.merged_model = Model()
        # self.merged_model.backbone.load_state_dict(merged_backbone, strict=True)
        
        # for task in range(self.cur_task + 1):
        #     # print(f"Loading head at logs/checkpoints/head_{task}.pt")
        #     self.merged_model.head.heads[-1].load_state_dict(torch.load(head_checkpoint(task)), strict=True)
        #     if task < self.cur_task:
        #         self.merged_model.head.update(20)
        # # self.merged_model.requires_grad_(False)
        # self.merged_model.cuda()
        # self.merged_model.eval()
        
        # print(f"Merged model: {self.merged_model}")
        
        # backbone = configure_model(self.merged_model.backbone)
        # check_model(backbone)
        # params, param_names = collect_params(backbone)
        
        base_params = torch.load(backbone_base())
        tasks_params = [torch.load(backbone_checkpoint(task)) for task in range(self.cur_task + 1)]
        backbone_params = ties_merge(base_params, tasks_params, lamb=1.0)
        self.model.backbone.load_state_dict(backbone_params, strict=False)

    def before_task(self, task, data_manager):
        print(self.model)
        self._classes_seen_so_far = self._known_classes + data_manager.get_task_size(task)
        self._class_increments.append((self._known_classes, self._classes_seen_so_far - 1))
        self.cur_task = task
        
        del self._network.fc
        self._network.fc = None
        self._network.update_fc(self._classes_seen_so_far)

    def after_task(self):
        # self.model.head.update(self._classes_seen_so_far - self._known_classes)
        # self.model.head.cuda()
        self._known_classes = self._classes_seen_so_far
    
    def eval(self, test_loader):
        self.model.eval()
        
        y_true, y_pred = [], []
        total, num_fallback = 0, 0
        
        with torch.no_grad():
            for _, (_, _, x, y) in tqdm(enumerate(test_loader)):
                x, y = x.cuda(), y.cuda()
                
                if self.cur_task > 0:
                    logits = []
                    for i in range(self.cur_task + 1):
                        self.model.backbone.load_state_dict(torch.load(backbone_checkpoint(i)), strict=False)
                        self.model.head.heads[0].load_state_dict(torch.load(head_checkpoint(i)), strict=True)
                        
                        logits.append(self.model(x))
                        
                    logits = torch.stack(logits, dim=1)
                    max_logit_values, max_logit_indices = torch.max(logits, dim=-1)
                    
                    temperature = 1
                    energy = -torch.logsumexp(logits/ temperature, dim=-1)
                    
                    task_indices_offset = torch.arange(self.cur_task+1, dtype=torch.long, device=device) * 20
                    max_logit_indices += task_indices_offset

                    # top2_values, top2_indices = torch.topk(max_logit_values, 2, dim=-1)
                    # # print(top2_logit_values[0, :])
                    # mask = (top2_logit_values[:, 0] - top2_logit_values[:, 1]) > 5
                    
                    top2_values, top2_indices = torch.topk(energy, 2, dim=-1, largest=False)
                    # mask = (-top2_values[:, 0] + top2_values[:, 1]) > 0.5
                    
                    # selected_indices = max_logit_indices[torch.arange(max_logit_indices.shape[0]), top2_indices[:, 0]]
                    
                    predicts = max_logit_indices[torch.arange(max_logit_indices.shape[0]), top2_indices[:, 0]]

                    # predicts = torch.ones(y.size(0), dtype=torch.long, device=device) * -100
                    # predicts[mask] = selected_indices[mask]
                    
                    # # merged_logits = F.softmax(self.merged_model(x), dim=-1)
                    
                    # merged_logits = self._network(x)['logits']
                    # combined_logits = logits.view(-1, merged_logits.size(1))
                    
                    # merged_logits = F.log_softmax(merged_logits / torch.norm(merged_logits, dim=-1, keepdim=True), dim=-1)
                    # combined_logits = F.log_softmax(combined_logits / torch.norm(combined_logits, dim=-1, keepdim=True), dim=-1)
                    
                    # merged_logits = self.standardize(merged_logits)
                    # combined_logits = self.standardize(combined_logits)
                    
                    # # print(f"Merge: {merged_logits[0, :5]}, Combine: {combined_logits[0, :5]}")
                    # merged_logits = torch.maximum(merged_logits, combined_logits)
                    
                    # # merged_logits = self.tented_model(x, ~mask)
                    # # merged_logits = F.softmax(self.merged_model(x), dim=-1)
                    
                    # mask = torch.zeros(y.size(0), dtype=torch.bool, device=device)
                    # mask = torch.max(combined_logits, dim=-1).values > torch.max(merged_logits, dim=-1).values
                    
                    # predicts[mask] = combined_logits.argmax(dim=1)[mask]
                    # predicts[~mask] = merged_logits.argmax(dim=1)[~mask]
                    
                    y_pred.append(predicts.cpu().numpy())
                    
                    # num_fallback += (~mask).sum().item()
                    
                    # print(f"Correct merged: {correct_merged}, Num fallback: {num_fallback}")
                    # incorrect_mask = (predicts != y)
                    # incorrect_indices = merged_logits.argmax(dim=1)[incorrect_mask]
                    # incorrect_logits = merged_logits[torch.arange(len(y), device=device)[incorrect_mask], incorrect_indices]
                    # print("Incorrect Predictions:")
                    # for i, (index, value) in enumerate(zip(incorrect_indices.cpu().numpy(), incorrect_logits.cpu().numpy())):
                    #     print(f"Sample {i}: Predicted class {index}, Logit {value:.4f}, True label {y[incorrect_mask][i].item()}, Logit true label {merged_logits[incorrect_mask][i, y[incorrect_mask][i]].item()}")
                else:
                    self.model.backbone.load_state_dict(torch.load(backbone_checkpoint(0)), strict=False)
                    self.model.head.heads[0].load_state_dict(torch.load(head_checkpoint(0)), strict=True)
            
                    logits = self.model(x)
                    y_pred.append(logits.argmax(dim=1).cpu().numpy())
                
                # merged_logits = self.model(x)
                # ranpac_logits = self._network(x)['logits']
                # merged_log_probs = F.log_softmax(merged_logits, dim=-1)
                # ranpac_log_probs = F.log_softmax(ranpac_logits, dim=-1)
                
                # merged_values, merged_indices = merged_log_probs.max(dim=-1)
                # ranpac_values, ranpac_indices = ranpac_log_probs.max(dim=-1)
                
                # mask = merged_values > ranpac_values
                
                # temperature = 2.0  # Adjust temperature to reduce overconfidence
                # merged_logits = self.model(x)
                # ranpac_logits = self._network(x)['logits']

                # # merged_log_probs = F.log_softmax(merged_logits, dim=-1)
                # # ranpac_log_probs = F.log_softmax(ranpac_logits, dim=-1)
                
                # normalized_merged_logits = F.normalize(merged_logits, p=1, dim=-1)
                # normalized_ranpac_logits = F.normalize(ranpac_logits, p=1, dim=-1)

                # # merged_entropy = -(merged_log_probs.exp() * merged_log_probs).sum(dim=-1)
                # # ranpac_entropy = -(ranpac_log_probs.exp() * ranpac_log_probs).sum(dim=-1)

                # # # mask = merged_entropy > ranpac_entropy
                
                # # merged_entropy = merged_entropy / merged_entropy.max()
                # # ranpac_entropy = ranpac_entropy / ranpac_entropy.max()

                # # # Compute confidence scores
                # # merged_confidence, merged_indices = merged_log_probs.exp().max(dim=-1)
                # # ranpac_confidence, ranpac_indices = ranpac_log_probs.exp().max(dim=-1)

                # # # Mask when self.model is uncertain (entropy higher and confidence lower)
                # # mask = (merged_entropy > ranpac_entropy) & (merged_confidence < 0.3)
                
                # mask = normalized_merged_logits > normalized_ranpac_logits
                
                # predicts = torch.ones(y.size(0), dtype=torch.long, device=device) * -100
                # predicts[mask] = merged_logits.argmax(dim=-1)[mask]
                # predicts[~mask] = ranpac_logits.argmax(dim=-1)[~mask]
                # # predicts[mask] = ranpac_indices[mask]
                # # predicts[~mask] = merged_indices[~mask]
                
                # temperature = 10.0  # Reduce overconfidence
                # # Apply temperature scaling
                # merged_logits = self.model(x) / temperature
                # ranpac_logits = self._network(x)['logits'] / temperature

                # # Normalize logits with L2 norm for better comparison
                # normalized_merged_logits = F.normalize(merged_logits, p=2, dim=-1)
                # normalized_ranpac_logits = F.normalize(ranpac_logits, p=2, dim=-1)

                # # Compute max confidence scores
                # merged_confidence, merged_indices = normalized_merged_logits.max(dim=-1)
                # ranpac_confidence, ranpac_indices = normalized_ranpac_logits.max(dim=-1)

                # mask = (merged_confidence < ranpac_confidence) & (ranpac_confidence > 0.5)

                # # Assign final predictions
                # predicts = torch.ones(y.size(0), dtype=torch.long, device=device) * -100
                # predicts[mask] = ranpac_indices[mask]  # Use ranpac when merged is unreliable
                # predicts[~mask] = merged_indices[~mask]  # Otherwise, use merged
                
                # logits = self._network(x)['logits']
                logits = self.model(x)
                predicts = logits.argmax(dim=1)
                
                y_pred.append(predicts.cpu().numpy())
                y_true.append(y.cpu().numpy())
                
                total += len(y)
                # num_fallback += (mask).sum().item()
        
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        acc_total, grouped = accuracy(y_pred.T, y_true, self._class_increments)
        logger.info(f"Total Acc: {acc_total:.2f}, Grouped Acc: {grouped}, Fallback: {num_fallback * 100 / total:.2f}")

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
        task_model = Model().cuda()
        
        epochs = 5
        weight_decay = 5e-4
        min_lr = 0.0
        optimizer = optim.SGD(task_model.parameters(), lr=1e-2, momentum=0.9, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)

        pbar = tqdm(range(epochs))
        for _, epoch in enumerate(pbar):
            task_model.train()
            total_loss, total, correct = 0, 0, 0

            for i, (_, _, x, y) in enumerate(train_loader):
                x, y = x.cuda(), y.cuda()
                y = torch.where(y - self._known_classes >= 0, y - self._known_classes, -100)
                
                logits = task_model(x)
                loss = F.cross_entropy(logits, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                correct += (logits.argmax(dim=1) == y).sum().item()
                total += len(y)

                info = f"Task {self.cur_task}, Epoch {epoch}, %: {total * 100 / len(train_loader.dataset):.2f}, Loss: {total_loss / total:.4f}, Acc: {correct * 100 / total:.2f}"
                pbar.set_description(info)

            scheduler.step()
        
        # temperature = config["temperature"]
        # model_optimizer = optim.SGD(self.model.head.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)
        # # model_optimizer = optim.AdamW(self.model.head.parameters(), lr=3e-4, weight_decay=5e-4)
        # self.model.train()
        # for _ in range(5):
        #     for _, (_, _, x, y) in enumerate(train_loader):
        #         x, y = x.cuda(), y.cuda()
                
        #         logits = self.model(x)
        #         probs = F.softmax(logits / temperature, dim=1)
        #         neg_entropy = (probs * torch.log(probs)).sum(dim=1).mean()
                
        #         model_optimizer.zero_grad()
        #         neg_entropy.backward()
        #         model_optimizer.step()
        
        # self.model.head.heads[-1].load_state_dict(task_model.head.heads[0].state_dict(), strict=True)
        
        task_model.cpu()
        torch.save(task_model.get_backbone_trainable_params(), backbone_checkpoint(self.cur_task))
        torch.save(task_model.head.heads[0].state_dict(), head_checkpoint(self.cur_task))

        del task_model
        gc.collect()
        torch.cuda.empty_cache()

def set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

for seed in [1993]:
    logger.info(f"Seed: {seed}")
    set_random(1)

    data_manager = DataManager("imageneta", True, seed, 20, 20, False)
    # print(f"Class order: {data_manager._class_order}")

    for temperature in [10.0]:
        config.update(
            {
                "fine_tune_train_batch_size": 64,
                "temperature": temperature
            }
        )
        logger.info(f"{' | '.join('%s: %s' % item for item in config.items())}")

        learner = Learner()
        learner.learn(data_manager)