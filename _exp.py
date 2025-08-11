import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers.weight_init import trunc_normal_
import math
import copy
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import random


def weight_init(m):
    """Enhanced weight initialization using timm's best practices"""
    if isinstance(m, nn.Linear):
        # Use truncated normal for linear layers (standard for transformers)
        trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        # For convolutional layers, use Kaiming normal (He initialization)
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        # For normalization layers
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Embedding):
        # For embedding layers
        trunc_normal_(m.weight, std=.02)


def plot_heatmap(true_labels, pred_labels):
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    
    plt.figure(figsize=(10, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', linewidths=0.5)
    plt.xlabel('Predicted Task Ids')
    plt.ylabel('True Task Ids')
    plt.savefig('confusion_matrix.png')


class Buffer(Dataset):
    def __init__(self, buffer_size: int):
        # store samples & labels
        self.buffer_size = buffer_size # maximum number of samples per class
        self.samples = defaultdict(list)
        self.xs = []
        self.ys = []
    
    def add(self, x, y):
        for i, iy in enumerate(y):
            self.samples[iy].append(x[i])
    
    def update(self):
        for clz in self.samples:
            if len(self.samples[clz]) > self.buffer_size:
                random.shuffle(self.samples[clz])
                self.samples[clz] = self.samples[clz][:self.buffer_size]

        self.xs = []
        self.ys = []
        for key, value in self.samples.items():
            self.xs.extend(value)
            self.ys.extend([key] * len(value))
    
    def __len__(self):
        return len(self.xs)
    
    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]


class TaskVector():
    def __init__(self, params=None):
        self.params = params # map of {name: tensor}
    
    def zero_like(self):
        with torch.no_grad():
            params = {}
            for name in self.params:
                params[name] = torch.zeros_like(self.params[name])
        return TaskVector(params)
    
    def ones_like(self):
        with torch.no_grad():
            params = {}
            for name in self.params:
                params[name] = torch.ones_like(self.params[name])
        return TaskVector(params)
    
    def __add__(self, other):
        assert isinstance(other, TaskVector) and self.params.keys() == other.params.keys()
        with torch.no_grad():
            params = {}
            for name in self.params:
                params[name] = self.params[name] + other.params[name]
        return TaskVector(params)

    def __add___(self, other):
        return self.__add__(other)
    
    def __mul__(self, other):
        assert isinstance(other, (int, float))
        with torch.no_grad():
            params = {}
            for name in self.params:
                params[name] = other * self.params[name]
        return TaskVector(params)
    
    def __mul___(self, other):
        return self.__mul__(other)
        
    def __repr__(self):
        return f"TaskVector({self.params})"


def merge(task_vectors, strategy):
    assert len(task_vectors) > 0, "task_vectors cannot be empty"

    first = task_vectors[0]

    merge_params = strategy.split('-')
    merge_type = merge_params[0]
    
    if merge_type == 'avg':
        ret = first.zero_like()
        for tv in task_vectors:
            ret += tv
        return ret * (1.0 / len(task_vectors))
    elif merge_type in {'max_mag', 'min_mag'}:
        ret = first.zero_like()
        for name in first.params:
            values = torch.stack([torch.abs(tv.params[name]) for tv in task_vectors])
            if merge_type == 'max_mag':
                indices = values.argmax(dim=0)
            else:
                indices = values.argmin(dim=0)
            ret.params[name] = torch.stack([tv.params[name] for tv in task_vectors])[indices]

        return ret
    elif merge_type in {'max', 'min'}:
        ret = first.zero_like()
        for name in first.params:
            values = torch.stack([tv.params[name] for tv in task_vectors])
            if merge_type == 'max':
                ret.params[name], _ = torch.max(values, dim=0)
            else:
                ret.params[name], _ = torch.min(values, dim=0)
        return ret
    elif merge_type == 'ties':
        ret = first.zero_like()
        
        merge_reset_type = merge_params[1]          # topk | bottomk
        merge_reset_thresh = int(merge_params[2])   # 20 | 50 | 100
        merge_resolve = merge_params[3]             # mass | normfrac | normmass
        merge_func = merge_params[4]                # dis | ...
        
        for name in first.params:
            values = torch.stack([tv.params[name] for tv in task_vectors])
            ret.params[name] = merge_methods(
                merge_reset_type, values, merge_reset_thresh, merge_resolve, merge_func)
            
        return ret
    else:
        raise NotImplementedError(f"Strategy '{strategy}' not implemented.")


def merge_methods(
    reset_type,
    flat_task_checks,
    reset_thresh=None,
    resolve_method=None,
    merge_func="",
):
    all_checks = flat_task_checks.clone()

    if "nf" in reset_type and reset_thresh != "none":
        updated_checks, *_ = topk_mask_preserve_normfrac(
            all_checks, reset_thresh, return_mask=False
        )
    elif "topk" in reset_type and reset_thresh != "none":
        updated_checks, *_ = topk_values_mask(
            all_checks, K=reset_thresh, return_mask=False
        )
    elif "std" in reset_type and reset_thresh != "none":
        updated_checks, *_ = greater_than_std_mask(
            all_checks, reset_thresh, return_mask=False
        )
    else:
        updated_checks = all_checks

    if resolve_method != "none":
        final_signs = resolve_sign(updated_checks, resolve_method)
        assert final_signs is not None
    else:
        final_signs = None

    if "dis" in merge_func:
        merged_tv = disjoint_merge(updated_checks, merge_func, final_signs)
    else:
        merged_tv = aggregate(updated_checks, merge_func, final_signs)

    return merged_tv


### PRUNING ###

def topk_mask_preserve_normfrac(T, normfrac=0.9, return_mask=False):
    row_norms = torch.norm(T, p=2, dim=1, keepdim=True)

    # Calculate the proportion of each element's contribution to its row's norm
    proportion = T.abs() ** 2 / row_norms ** 2

    # Sort the proportions and their indices in descending order
    sorted_proportions, sorted_indices = torch.sort(proportion, dim=1, descending=True)

    # Calculate the cumulative sum of proportions
    cumsum_proportions = torch.cumsum(sorted_proportions, dim=1)

    # Find the indices where cumulative sum >= normfrac
    normfrac_mask = cumsum_proportions >= normfrac
    normfrac_indices = torch.argmax(normfrac_mask.float(), dim=1)

    # Create a range tensor to compare with normfrac_indices
    range_tensor = torch.arange(T.size(1)).unsqueeze(0).expand(T.size(0), -1)

    # Create a mask based on the normfrac_indices
    mask = range_tensor <= normfrac_indices.unsqueeze(1)

    # Initialize final_indices with a value that is out of bounds
    final_indices = torch.full_like(sorted_indices, T.size(1) - 1)

    # Use the mask to get the final indices
    final_indices[mask] = sorted_indices[mask]

    # Initialize the mask with zeros
    M = torch.zeros_like(T, dtype=torch.bool)

    # Use the final indices to update the final mask M
    M.scatter_(1, final_indices, True)

    if return_mask:
        return (T * M), M.float().mean(dim=1), M
    else:
        return (T * M), M.float().mean(dim=1)
    

def topk_values_mask(M, K=0.7, return_mask=False):
    if K > 1:
        K /= 100

    original_shape = M.shape
    if M.dim() == 1:
        M = M.unsqueeze(0)

    n, d = M.shape
    k = int(d * K)
    k = d - k  # Keep top k elements instead of bottom k elements

    # Find the k-th smallest element by magnitude for each row
    kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
    # Create a mask tensor with True for the top k elements in each row
    mask = M.abs() >= kth_values
    final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask

    if return_mask:
        return M * final_mask, final_mask.float().mean(dim=1), final_mask
    return M * final_mask, final_mask.float().mean(dim=1)


def bottomk_values_mask(M, K=0.7, return_mask=False):
    if K > 1:
        K /= 100

    original_shape = M.shape
    if M.dim() == 1:
        M = M.unsqueeze(0)

    n, d = M.shape
    k = int(d * K)
    # Find the k-th smallest element by magnitude for each row
    kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)

    # Create a mask tensor with True for the bottom k elements in each row
    mask = M.abs() <= kth_values
    final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask

    if return_mask:
        return M * final_mask, final_mask.float().mean(dim=1), final_mask
    return M * final_mask, final_mask.float().mean(dim=1)


def greater_than_std_mask(tensor, factor, return_mask=False):
    mask = (tensor - tensor.mean(dim=1).unsqueeze(1)).abs() > factor * tensor.std(
        dim=1
    ).unsqueeze(1)
    if return_mask:
        return tensor * mask, mask.float().mean(dim=1), mask
    return tensor * mask, mask.float().mean(dim=1)


def less_than_std_mask(tensor, factor, return_mask=False):
    mask = (tensor - tensor.mean(dim=1).unsqueeze(1)).abs() < factor * tensor.std(
        dim=1
    ).unsqueeze(1)
    if return_mask:
        return tensor * mask, mask.float().mean(dim=1), mask
    return tensor * mask, mask.float().mean(dim=1)


### RESOLVING SIGN ###

def resolve_sign(Tensor, resolve_method):
    if resolve_method == "mass":
        sign_to_mult = torch.sign(Tensor.sum(dim=0))
    elif resolve_method == "normfrac":
        sign_to_mult = normfrac_based_sign(Tensor)
    elif resolve_method == "normmass":
        sign_to_mult = normmass_based_sign(Tensor)
    else:
        raise ValueError(f"Sign resolve method {resolve_method} is not defined.")
    sign_to_mult = resolve_zero_signs(sign_to_mult, "majority")
    return sign_to_mult


def normfrac_based_sign(Tensor):
    row_norms = torch.norm(Tensor, dim=1, keepdim=True)
    norm_fracs = (Tensor ** 2) / row_norms ** 2
    return torch.sign(Tensor[norm_fracs.argmax(dim=0), torch.arange(Tensor.shape[1])])


def normmass_based_sign(Tensor):
    row_norms = torch.norm(Tensor, dim=1, keepdim=True)
    norm_fracs = (Tensor ** 2) / row_norms ** 2
    return (Tensor.sign() * norm_fracs.abs()).sum(dim=0).sign()


def resolve_zero_signs(sign_to_mult, method="majority"):
    majority_sign = torch.sign(sign_to_mult.sum())

    if method == "majority":
        sign_to_mult[sign_to_mult == 0] = majority_sign
    elif method == "minority":
        sign_to_mult[sign_to_mult == 0] = -1 * majority_sign
    return sign_to_mult


### AGGREGATION ###

def disjoint_merge(Tensor, merge_func, sign_to_mult):

    merge_func = merge_func.split("_")[-1]

    # If sign is provided then we select the corresponding entries and aggregate.
    if sign_to_mult is not None:
        rows_to_keep = torch.where(
            sign_to_mult.unsqueeze(0) > 0, Tensor > 0, Tensor < 0
        )
        selected_entries = Tensor * rows_to_keep
    # Else we select all non-zero entries and aggregate.
    else:
        rows_to_keep = Tensor != 0
        selected_entries = Tensor * rows_to_keep

    if merge_func == "mean":
        non_zero_counts = (selected_entries != 0).sum(dim=0).float()
        disjoint_aggs = torch.sum(selected_entries, dim=0) / torch.clamp(
            non_zero_counts, min=1
        )
    elif merge_func == "sum":
        disjoint_aggs = torch.sum(selected_entries, dim=0)
    elif merge_func == "max":
        disjoint_aggs = selected_entries.abs().max(dim=0)[0]
        disjoint_aggs *= sign_to_mult
    else:
        raise ValueError(f"Merge method {merge_func} is not defined.")

    return disjoint_aggs


def aggregate(T, agg_type, final_signs, dim=0):
    if agg_type == "mean":
        result = torch.mean(T, dim=dim)
    elif agg_type == "sum":
        result = torch.sum(T, dim=dim)
    elif agg_type == "median":
        result = torch.median(T, dim=dim)[0]
    elif agg_type == "magnitude":
        max_indices = T.abs().argmax(dim=0)
        result = T[max_indices, torch.arange(T.shape[1])]
    else:
        raise ValueError("Invalid agg_type: %s" % agg_type)

    if final_signs is not None:
        # print(final_signs)
        result = result.abs() * final_signs

    return result


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
    

M = 10000

class RandomProjectionHead(nn.Module):
    def __init__(self, embed_dim, projection_dim, nb_classes, sigma=True, init_matrix='normal', alpha=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.nb_classes = nb_classes
        self.projection_dim = projection_dim
        self.weight = nn.Parameter(torch.Tensor(nb_classes, projection_dim))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()
        
        self.alpha = alpha
        self.init_matrix = init_matrix
        if init_matrix == 'none':
            self.W_rand = None
        elif init_matrix == 'normal':
            self.W_rand = torch.randn(embed_dim, projection_dim)
        elif init_matrix == 'sparse_3':
            # Achlioptas' ternary distribution
            probs = torch.rand(embed_dim, projection_dim)
            self.W_rand = torch.zeros(embed_dim, projection_dim)

            sqrt_3 = torch.sqrt(torch.tensor(3.0))
            self.W_rand[probs < 1/6] = sqrt_3
            self.W_rand[(probs >= 1/6) & (probs < 1/3)] = -sqrt_3
            
            # probs = torch.rand(embed_dim, projection_dim)
            # self.W_rand = torch.zeros(embed_dim, projection_dim)
            # self.W_rand[probs < 1/2] = 1
            # self.W_rand[probs >= 1/2] = -1
        elif init_matrix == 'combine':
            self.W_rand = torch.randn(embed_dim, projection_dim)
            
            probs = torch.rand(projection_dim, M)
            self.W2_rand = torch.zeros(projection_dim, M)
            sqrt_3 = torch.sqrt(torch.tensor(3.0))
            self.W2_rand[probs < 1/6] = sqrt_3
            self.W2_rand[(probs >= 1/6) & (probs < 1/3)] = -sqrt_3
            
            # probs = torch.rand(projection_dim, M)
            # self.W2_rand = torch.zeros(projection_dim, M)
            # self.W2_rand[probs < 1/2] = 1
            # self.W2_rand[probs >= 1/2] = -1
        else:
            raise NotImplementedError(f"Type {init_matrix} not implemented.")
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)
    
    def update(self, nb_classes):
        weight = copy.deepcopy(self.weight.data)
        weight = torch.cat([weight, torch.zeros(nb_classes, self.projection_dim).cuda()])
        self.weight = nn.Parameter(weight)
        self.nb_classes = self.nb_classes + nb_classes
        
    def project(self, x, device='cuda:0'):
        if self.init_matrix == 'none':
            return x
        elif self.init_matrix == 'normal':
            return F.relu(x @ self.W_rand.to(device))
        elif self.init_matrix == 'sparse_3':
            return self.alpha * (x @ self.W_rand.to(device))
        elif self.init_matrix == 'combine':
            return self.alpha * F.relu(F.relu(x @ self.W_rand.to(device)) @ self.W2_rand.to(device))
        else:
            raise NotImplementedError(f"Type {self.init_matrix} not implemented.")
    
    def forward(self, x):
        if self.W_rand is None:
            out = F.linear(F.normalize(x, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        else:
            out = F.linear(self.project(x), self.weight)
        if self.sigma is not None:
            out = self.sigma * out
        
        return {'logits': out}
    
    
class ContinualLearnerHead(nn.Module):
    def __init__(self, embed_dim, nb_classes, with_norm=False, with_bias=False):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.with_norm = with_norm
        self.with_bias = with_bias
        
        self.heads = nn.ModuleList([])
        self.update(nb_classes)
    
    def create_head(self, nb_classes):
        single_head = []
        if self.with_norm:
            single_head.append(nn.LayerNorm(self.embed_dim))
        fc = nn.Linear(self.embed_dim, nb_classes, bias=self.with_bias)
        trunc_normal_(fc.weight, std=.02)
        if self.with_bias:
            nn.init.constant_(fc.bias, 0)
        single_head.append(fc)
        head = nn.Sequential(*single_head)
        return head

    def update(self, nb_classes, freeze_old=True):
        if freeze_old:
            for p in self.heads.parameters():
                p.requires_grad=False
        single_head = self.create_head(nb_classes)
        self.heads.append(single_head)
    
    def reset(self, head_indexes):
        if head_indexes is None:
            for i, head in enumerate(self.heads):
                trunc_normal_(head[-1].weight, std=.02)
        else:
            for i in head_indexes:
                trunc_normal_(self.heads[i][-1].weight, std=.02)
                if self.with_bias:
                    nn.init.constant_(self.heads[i][-1].bias, 0)
    
    def forward(self, x, grad_scales=None):
        """
        grad_scales: list of floats, one per head, specifying gradient scale
        """
        out = []
        for i, head in enumerate(self.heads):
            logits_i = head(x)  # shape: [B, num_classes_i]

            # Register hook to scale gradients only if required
            if grad_scales is not None and grad_scales[i] != 1.0:
                logits_i.register_hook(lambda g, scale=grad_scales[i]: g * scale)

            out.append(logits_i)

        return {'logits': torch.cat(out, dim=1)}


class SimpleContinualLinear(nn.Module):
    def __init__(self, embed_dim, nb_classes, feat_expand=False, with_norm=False):
        super().__init__()

        self.embed_dim = embed_dim
        self.feat_expand = feat_expand
        self.with_norm = with_norm
        heads = []
        single_head = []
        if with_norm:
            single_head.append(nn.LayerNorm(embed_dim))

        single_head.append(nn.Linear(embed_dim, nb_classes, bias=True))
        head = nn.Sequential(*single_head)

        heads.append(head)
        self.heads = nn.ModuleList(heads)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02) 
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def update(self, nb_classes, freeze_old=True):
        single_head = []
        if self.with_norm:
            single_head.append(nn.LayerNorm(self.embed_dim))

        _fc = nn.Linear(self.embed_dim, nb_classes, bias=True)
        trunc_normal_(_fc.weight, std=.02)
        nn.init.constant_(_fc.bias, 0)
        single_head.append(_fc)
        new_head = nn.Sequential(*single_head)


        if freeze_old:
            for p in self.heads.parameters():
                p.requires_grad=False

        self.heads.append(new_head)
    
    def forward(self, x):
        out = []
        for ti in range(len(self.heads)):
            fc_inp = x[ti] if self.feat_expand else x
            out.append(self.heads[ti](fc_inp))
        out = {'logits': torch.cat(out, dim=1)}
        return out


import logging
import timm
from peft import LoraConfig, get_peft_model
from easydict import EasyDict
from petl import vision_transformer_ssf
from petl import vision_transformer_adapter
import numpy as np

def setup_logger(log_file=f"logs/default.log"):
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG)

    logger.propagate = False

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_formatter = logging.Formatter('%(asctime)s - %(message)s')
    console_formatter = logging.Formatter("%(asctime)s [%(filename)s] => %(message)s")

    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


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
            r=config["model_lora_r"],
            lora_alpha=config["model_lora_alpha"],
            target_modules=["qkv"],
            lora_dropout=config["model_lora_dropout"],
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

from sklearn.cluster import KMeans
from collections import OrderedDict
from typing import Optional


class RandomReplayBuffer:
    def __init__(self, capacity: int, decay=1.0, seed: Optional[int] = None):
        self._capacity = capacity
        self._buffer = []
        self._weights = []
        self._total_seen = 0
        self._decay = decay
        self._seed = seed

    def add(self, x: torch.Tensor, z: torch.Tensor, y: torch.Tensor):
        for i in range(y.size(0)):
            self._total_seen += 1
            entry = (x[i].cpu(), z[i].cpu(), y[i].cpu())

            if len(self._buffer) < self._capacity:
                self._buffer.append(entry)
                self._weights.append(1.0)
            else:
                if random.random() >= (self._capacity / self._total_seen):
                    continue

                probs = torch.tensor(self._weights, dtype=torch.float32)
                inv_probs = 1.0 / (probs + 1e-6)
                inv_probs = inv_probs / inv_probs.sum()

                if self._seed is not None:
                    torch.manual_seed(self._seed + self._total_seen)  # vary by sample
                idx = torch.multinomial(inv_probs.cpu(), 1).item()

                if 1.0 >= self._weights[idx]:
                    self._buffer[idx] = entry
                    self._weights[idx] = 1.0

    def update_weights(self):
        if not self._buffer:
            return
        self._weights = [w * self._decay for w in self._weights]

    def sample(self, batch_size: int = 32, seed: Optional[int] = None):
        if not self._buffer:
            return None

        weights_np = np.array(self._weights, dtype=np.float32)
        probs = weights_np / weights_np.sum()

        if seed is not None:
            rng = np.random.RandomState(seed)
            indices = rng.choice(len(self._buffer), size=min(batch_size, len(self._buffer)), p=probs)
        elif self._seed is not None:
            rng = np.random.RandomState(self._seed + self._total_seen)
            indices = rng.choice(len(self._buffer), size=min(batch_size, len(self._buffer)), p=probs)
        else:
            indices = np.random.choice(len(self._buffer), size=min(batch_size, len(self._buffer)), p=probs)

        xs, zs, ys = zip(*[self._buffer[i] for i in indices])
        x_batch = torch.stack(xs)
        z_batch = torch.stack(zs)
        y_batch = torch.tensor(ys, dtype=torch.long)
        return x_batch, z_batch, y_batch

    def __iter__(self, batch_size: int = 32, seed: Optional[int] = None):
        buffer = self._buffer.copy()
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(buffer)
        elif self._seed is not None:
            rng = random.Random(self._seed + self._total_seen)
            rng.shuffle(buffer)
        else:
            random.shuffle(buffer)

        for i in range(0, len(buffer), batch_size):
            batch = buffer[i: i + batch_size]
            x_batch = torch.stack([x for x, _, _ in batch])
            z_batch = torch.stack([z for _, z, _ in batch])
            y_batch = torch.tensor([y for _, _, y in batch], dtype=torch.long)
            yield x_batch, z_batch, y_batch

    @property
    def size(self):
        return len(self._buffer)

    @property
    def size_by_class(self):
        class_counts = {}
        for _, _, y in self._buffer:
            y_value = y.item()
            class_counts[y_value] = class_counts.get(y_value, 0) + 1
        return OrderedDict(sorted(class_counts.items()))



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
    
    session_averages = []
    for i in range(accuracy_matrix.shape[0]):
        session_avg = np.mean(accuracy_matrix[i, :i+1])
        session_averages.append(session_avg)
    asa = np.mean(session_averages)
    
    if accuracy_matrix.shape[0] == 1:
        return faa, 0.0, 0.0, asa
    
    final_acc_per_task = accuracy_matrix[-1]
    max_acc_per_task = np.max(accuracy_matrix, axis=0)
    ffm = np.mean(max_acc_per_task[:-1] - final_acc_per_task[:-1])
    ffd = np.max(max_acc_per_task[:-1] - final_acc_per_task[:-1]) - np.min(
        max_acc_per_task[:-1] - final_acc_per_task[:-1]
    )

    return faa, ffm, ffd, asa


if __name__ == '__main__':
    # head = SimpleContinualLinear(758, 10)
    # print(head)
    # head.update(10)
    # print(head)
    # x = torch.randn(64, 758)
    # out = head(x)
    # print(out['logits'].shape)
    # head = nn.Linear(768, 20)
    # print(head)
    
    tv1 = TaskVector({'a': torch.randn(2), 'b': torch.randn(2)})
    tv2 = TaskVector({'a': torch.randn(2), 'b': torch.randn(2)})
    print(f"tv1: {tv1}, tv2: {tv2}")
    
    avg_tv = merge([tv1, tv2], 'avg')
    max_mag_tv = merge([tv1, tv2], 'max_mag')
    min_mag_tv = merge([tv1, tv2], 'min_mag')
    max_tv = merge([tv1, tv2], 'max')
    min_tv = merge([tv1, tv2], 'min')
    ties_tv = merge([tv1, tv2], 'ties-topk-20-mass-dis_mean')
    
    # print(f"avg_tv: {avg_tv}, max_mag_tv: {max_mag_tv}, min_mag_tv: {min_mag_tv}, max_tv: {max_tv}, min_tv: {min_tv}")
    
    print(f"ties_tv: {ties_tv}")
    
    print(f"tv1: {tv1}, tv2: {tv2}")
