import copy
import logging
import math
import torch
from torch import nn
import timm
from torch.nn import functional as F
from timm.models.vision_transformer import VisionTransformer
from petl import vision_transformer_ssf
import numpy as np
from peft import LoraConfig, get_peft_model
from _exp import ContinualLearnerHead
from utils.toolkit import count_parameters

class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, nb_proxy=1, to_reduce=False, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features * nb_proxy
        self.nb_proxy = nb_proxy
        self.to_reduce = to_reduce
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
        if not self.use_RP:
            out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        else:
            if self.W_rand is not None:
                inn = torch.nn.functional.relu(input @ self.W_rand)
            else:
                inn=input
                #inn=torch.bmm(input[:,0:100].unsqueeze(-1), input[:,0:100].unsqueeze(-2)).flatten(start_dim=1) #interaction terms instead of RP
            out = F.linear(inn,self.weight)

        if self.to_reduce:
            # Reduce_proxy
            out = reduce_proxies(out, self.nb_proxy)

        if self.sigma is not None:
            out = self.sigma * out

        return {'logits': out}


def get_convnet(args, pretrained=False):

    name = args["convnet_type"].lower()
    #Resnet
    if name=="pretrained_resnet50":
        from resnet import resnet50
        model=resnet50(pretrained=True,args=args)
        return model.eval()
    elif name=="pretrained_resnet152":
        from resnet import resnet152
        model=resnet152(pretrained=True,args=args)
        return model.eval()
    elif name=="vit_base_patch32_224_clip_laion2b":
        #note: even though this is "B/32" it has nearly the same num params as the standard ViT-B/16
        model=timm.create_model("vit_base_patch32_224_clip_laion2b", pretrained=True, num_classes=0)
        model.out_dim=768
        return model.eval()
    
    #NCM or NCM w/ Finetune
    elif name=="pretrained_vit_b16_224" or name=="vit_base_patch16_224":
        model=timm.create_model("vit_base_patch16_224",pretrained=True, num_classes=0)
        model.out_dim=768
        return model.eval()
    elif name=="pretrained_vit_b16_224_in21k" or name=="vit_base_patch16_224_in21k":
        model=timm.create_model("vit_base_patch16_224_in21k",pretrained=True, num_classes=0)
        model.out_dim=768
        return model.eval()
    
    # SSF 
    elif '_ssf' in name:
        if args["model_name"]=="ssf":
            from petl import vision_transformer_ssf #registers vit_base_patch16_224_ssf
            if name=="pretrained_vit_b16_224_ssf":
                model = timm.create_model("vit_base_patch16_224_ssf", pretrained=True, num_classes=0)
                model.out_dim=768
            elif name=="pretrained_vit_b16_224_in21k_ssf":
                model=timm.create_model("vit_base_patch16_224_in21k_ssf",pretrained=True, num_classes=0)
                model.out_dim=768
            return model.eval()
        else:
            raise NotImplementedError("Inconsistent model name and model type")
    
    # VPT
    elif '_vpt' in name:
        if args["model_name"]=="vpt":
            from petl.vpt import build_promptmodel
            if name=="pretrained_vit_b16_224_vpt":
                basicmodelname="vit_base_patch16_224" 
            elif name=="pretrained_vit_b16_224_in21k_vpt":
                basicmodelname="vit_base_patch16_224_in21k"
            
            #print("modelname,",name,"basicmodelname",basicmodelname)
            VPT_type="Deep"
            #if args["vpt_type"]=='shallow':
            #    VPT_type="Shallow"
            Prompt_Token_num=5#args["prompt_token_num"]

            model = build_promptmodel(modelname=basicmodelname,  Prompt_Token_num=Prompt_Token_num, VPT_type=VPT_type)
            prompt_state_dict = model.obtain_prompt()
            model.load_prompt(prompt_state_dict)
            model.out_dim=768
            return model.eval()
        else:
            raise NotImplementedError("Inconsistent model name and model type")

    elif '_adapter' in name:
        ffn_num=64#args["ffn_num"]
        if args["model_name"]=="adapter" :
            from petl import vision_transformer_adapter
            from easydict import EasyDict
            tuning_config = EasyDict(
                # AdaptFormer
                ffn_adapt=True,
                ffn_option="parallel",
                ffn_adapter_layernorm_option="none",
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar="0.1",
                ffn_num=ffn_num,
                d_model=768,
                # VPT related
                vpt_on=False,
                vpt_num=0,
            )
            if name=="pretrained_vit_b16_224_adapter":
                model = vision_transformer_adapter.vit_base_patch16_224_adapter(num_classes=0,
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim=768
            elif name=="pretrained_vit_b16_224_in21k_adapter":
                model = vision_transformer_adapter.vit_base_patch16_224_in21k_adapter(num_classes=0,
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim=768
            else:
                raise NotImplementedError("Unknown type {}".format(name))
            return model.eval()
        else:
            raise NotImplementedError("Inconsistent model name and model type")
    elif '_lora' in name:
        if name=="pretrained_vit_b16_224_lora":
            model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
            model.out_dim=768
        elif name == "pretrained_vit_b16_224_in21k_lora":
            model = timm.create_model("vit_base_patch16_224_in21k", pretrained=True, num_classes=0)
            model.out_dim=768
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["qkv"],
            lora_dropout=0.1,
            bias="none"
        )
        model = get_peft_model(model, lora_config)
        return model.eval()
    else:
        raise NotImplementedError("Unknown type {}".format(name))

class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()
        self.convnet = get_convnet(args, pretrained)
        
        # tuning_config = EasyDict(
        #     # AdaptFormer
        #     ffn_adapt=True,
        #     ffn_option="parallel",
        #     ffn_adapter_layernorm_option="none",
        #     ffn_adapter_init_option="lora",
        #     ffn_adapter_scalar="0.1",
        #     ffn_num=64,
        #     d_model=768,
        #     # VPT related
        #     vpt_on=False,
        #     vpt_num=0,
        # )
        # self.convnet = vision_transformer_adapter.vit_base_patch16_224_in21k_adapter(num_classes=0,
        #     global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
        # self.convnet.out_dim=768
        # self.convnet = self.convnet.eval()
        
        # model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        # model.out_dim=768
        # lora_config = LoraConfig(
        #     r=16,
        #     lora_alpha=16,
        #     target_modules=["qkv"],
        #     lora_dropout=0.1,
        #     bias="none"
        # )
        # model = get_peft_model(model, lora_config)
        # self.convnet = model.eval()
        
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        """
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        """
        out.update(x)

        return out

    def update_fc(self, nb_classes):
        pass

class ResNetCosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

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

class SimpleVitNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

    def update_fc(self, nb_classes):
        fc = CosineLinear(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.fc.weight.shape[1]).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x)
        return out
    
    def add_cls_head(self, nb_classes):
        self.cls_head = CosineLinear(self.feature_dim, nb_classes).cuda()
    
    def cls_forward(self, x):
        x = self.convnet(x)
        out = self.cls_head(x)
        return out
    
    def get_backbone_trainable_params(self):
        params = {}
        for name, param in self.convnet.named_parameters():
            if param.requires_grad:
                params[name] = param
        return params
    
    def show_num_params(self):
        trainable_params = count_parameters(self, trainable=True)
        total_params = count_parameters(self)
        return f"Model(trainable_params={trainable_params}, total_params={total_params}, percentage={trainable_params * 100 / total_params:.2f})"


def lanczos(W_sparse, k=6, num_iterations=100, tol=1e-6):
    """
    Lanczos algorithm for approximating the largest k eigenvalues of a symmetric sparse matrix.

    Parameters:
    - W_sparse: (torch.sparse) Sparse symmetric matrix.
    - k: (int) Number of largest eigenvalues to approximate.
    - num_iterations: (int) Number of Lanczos iterations to perform.
    - tol: (float) Tolerance for early stopping based on the change in eigenvalues.

    Returns:
    - eigenvalues: (torch.Tensor) Approximated largest k eigenvalues.
    - eigenvectors: (torch.Tensor) Approximated eigenvectors corresponding to the largest k eigenvalues.
    """
    # Ensure W_sparse is in COO format and device consistency
    W_sparse = W_sparse.coalesce()
    n = W_sparse.size(0)

    # Initial vector for the Lanczos iteration
    v = torch.randn(n, device=W_sparse.device)
    v /= torch.norm(v)

    # Storage for Lanczos vectors and tridiagonal components
    V = torch.zeros(n, num_iterations, device=W_sparse.device)
    alpha = torch.zeros(num_iterations, device=W_sparse.device)
    beta = torch.zeros(num_iterations - 1, device=W_sparse.device)

    # Initial Lanczos iteration
    w = torch.sparse.mm(W_sparse, v.unsqueeze(1)).squeeze(1)
    alpha[0] = torch.dot(v, w)
    w -= alpha[0] * v
    V[:, 0] = v

    for j in range(1, num_iterations):
        beta[j - 1] = torch.norm(w)
        if beta[j - 1] < tol:
            break

        v = w / beta[j - 1]
        V[:, j] = v

        w = torch.sparse.mm(W_sparse, v.unsqueeze(1)).squeeze(1)
        w -= beta[j - 1] * V[:, j - 1]
        alpha[j] = torch.dot(v, w)
        w -= alpha[j] * v

    # Construct the tridiagonal matrix Teigenvalues
    T = torch.diag(alpha) + torch.diag(beta, diagonal=1) + torch.diag(beta, diagonal=-1)

    # Compute the eigenvalues and eigenvectors of T
    eigenvalues, eigenvectors_T = torch.linalg.eigh(T[:j, :j])

    # Select the k largest eigenvalues and their corresponding eigenvectors
    largest_eigenvalues = eigenvalues[-k:]
    largest_eigenvectors = V[:, :j] @ eigenvectors_T[:, -k:]

    return largest_eigenvalues, largest_eigenvectors

class ESNNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        
        self.features = []
        def get_features(model, inut, output):
            self.features.append(output)

        for name, module in self.convnet.named_modules():
            if isinstance(module, timm.models.vision_transformer.VisionTransformer) or \
                isinstance(module, vision_transformer_ssf.VisionTransformer):
                for i, block in enumerate(module.blocks):
                    block.register_forward_hook(get_features)
        
        self.Win = None
        self.Wres = None
    
    def setup_esn(self, M):
        self.M = M
        
        reservoir_size = M
        spectral_radius = 0.9  
        sparsity = 0.9
        diag_included = True
        device = 'cuda'
        
        self.sparsity = sparsity
        self.spectral_radius = spectral_radius
        
        self.alpha = 0
        self.C1 = 1e-3
        self.C2 = 0
        self.C3 = 0
        
        self.Win = torch.randn(self.feature_dim, M).cuda()
        # self.Win = torch.normal(0, self.C1 * np.sqrt(M), (self.feature_dim, M)).cuda()
        # self.Win = torch.normal(0, 1, (self.feature_dim, M)).cuda()
        
        self.Wres = torch.bernoulli(torch.full((M, M), 1 - sparsity)).cuda()
        
        # W = torch.zeros((reservoir_size, reservoir_size), device=device)  # Assume CUDA usage
        # num_nonzero = int((1 - sparsity) * reservoir_size * (reservoir_size + 1) // 2)

        # row_indices = torch.randint(0, reservoir_size, (num_nonzero,), device=device)
        # col_indices = torch.randint(0, reservoir_size, (num_nonzero,), device=device)
        # mask = row_indices >= col_indices if diag_included else row_indices > col_indices

        # filtered_row_indices = row_indices[mask]
        # filtered_col_indices = col_indices[mask]
        # random_values = torch.randn(len(filtered_row_indices), device=device)

        # W[filtered_row_indices, filtered_col_indices] = random_values
        
        # self.Wres = W + W.T
        # if not diag_included:
        #     self.Wres.fill_diagonal_(0)
        
        self.Wres = self.Wres.to_sparse()
        eigenvalues, _ = lanczos(self.Wres, k=1)
        max_eigenvalue = torch.max(torch.abs(eigenvalues))
        print(f"Max eigenvalue of reservoir: {max_eigenvalue}")
        self.Wres *= (self.C2 * spectral_radius / max_eigenvalue)
        
        print(f"Max: {torch.max(self.Wres.coalesce().values())}, Min: {torch.min(self.Wres.coalesce().values())}")
        print(f"sparsity: {self.sparsity}, spectral_radius: {self.spectral_radius}, C1: {self.C1}, C2: {self.C2}, C3: {self.C3}")
    
    def get_conv_features(self, x):
        self.features = []
        self.convnet(x)
        return self.features
    
    
    def get_esn_features(self, fs):
        h = torch.zeros(fs[0].size(0), self.M).cuda()
        
        # for f in fs[:]:
        #     # h = (1 - alpha) * h + alpha * F.relu(f[:,0,:] @ self.Win + torch.sparse.mm(self.Wres, h.T).T)
        #     # h = torch.sparse.mm(self.Wres, ((1 - alpha) * h + alpha * (f[:,0,:] @ self.Win)).T).T
            
        #     a = torch.Tensor(2 * self.C3 * (1 - self.sparsity) * np.sqrt(self.M) * torch.sum(torch.abs(h))).cuda()
        #     # norm_f = f[:, 0, :] / torch.norm(f[:, 0, :], dim=1, keepdim=True)
            
        #     feat = f[:, 0, :] @ self.Win
        #     h = feat + self.alpha * torch.sparse.mm(self.Wres, h.T).T - a
        #     h = torch.tanh(F.layer_norm(h, h.size()))
            
        #     # print("f = {} {}".format(torch.max(f[:, 0, :]), torch.min(f[:, 0, :])))
        #     # print("fin = {} {}".format(torch.max(feat), torch.min(feat)))
        #     # print("h = {} {}".format(torch.max(h), torch.min(h)))
        
        h = fs[-1][:, 0, :] @ self.Win
            
        return h
    
    def forward(self, x):
        self.get_conv_features(x)
        
        if self.Win is not None:
            f = self.get_esn_features(self.features)
        else:
            f = self.features[-1][:, 0, :]
        
        y = self.fc(f)
        return y
    
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