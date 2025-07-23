import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm
from petl import vision_transformer_ssf # register vision_transformer_ssf
import numpy as np
from tqdm import tqdm
from utils.data_manager import DataManager
from utils.toolkit import accuracy


seed = 42
data_manager = DataManager('imagenetr', True, seed, 0, 20, False)  

def lanczos(W_sparse, k=6, num_iterations=100, tol=1e-6):
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

class SelfAttentionLayer(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, layer_features):
        combined_features = torch.stack(layer_features, dim=1)  # Shape: (B, num_layers, S, E)
        query = self.query(combined_features)  # Shape: (B, num_layers, S, E)
        key = self.key(combined_features).transpose(-2, -1)  # Shape: (B, num_layers, E, S)
        value = self.value(combined_features)  # Shape: (B, num_layers, S, E)

        attention_scores = torch.matmul(query, key) / (combined_features.size(-1) ** 0.5)  # (B, num_layers, S, S)
        attention_weights = self.softmax(attention_scores)  # (B, num_layers, S, S)
        combined_output = torch.matmul(attention_weights, value)  # (B, num_layers, S, E)

        return combined_output.mean(dim=1)  # Combine across layers (optional reduction)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.base_network = timm.create_model(
            "vit_base_patch16_224", pretrained=True, num_classes=0)
        self.attention = SelfAttentionLayer(self.base_network.embed_dim)
        
        self.features = []
        def get_features(model, inut, output):
            self.features.append(output)

        for name, module in self.base_network.named_modules():
            if isinstance(module, timm.models.vision_transformer.VisionTransformer) or \
                isinstance(module, vision_transformer_ssf.VisionTransformer):
                for i, block in enumerate(module.blocks):
                    block.register_forward_hook(get_features)
        
        self.fc = nn.Linear(self.base_network.embed_dim, 20, bias=False)
        nn.init.kaiming_normal_(self.fc.weight)
        
        embed_dim = self.base_network.embed_dim
        M = 10000
        num_classes = 200
        
        # self.Win = torch.randn(768, 10000).cuda()
        
        p = 0.5 # sparsity
        
        self.Win = torch.normal(0, 1, (embed_dim, M)).cuda()
        self.Wres = torch.bernoulli(torch.full((M, M), 1 - p)).cuda()
        self.Wout = torch.zeros(M, num_classes).cuda()
        
        self.Wres = self.Wres.to_sparse()
        eigenvalues, _ = lanczos(self.Wres, k=1)
        max_eigenvalue = torch.max(torch.abs(eigenvalues))
        self.Wres *= 1e2 / max_eigenvalue
        
        self.M = M
        self.p = p
        
        print(f"Max: {torch.max(self.Wres.coalesce().values())}, Min: {torch.min(self.Wres.coalesce().values())}")
    
    def get_features(self, x):
        self.features = []
        return self.base_network(x)
    
    def get_esn_features(self, x):
        self.get_features(x)
        fs = self.features
        
        B = fs[0].size(0)
        h = torch.zeros(B, self.M).cuda()
        C3 = 1e-6
        
        for f in fs[:]:
            f_in = f[:, 0, :] @ self.Win
            a = 2 * C3 * (1 - self.p) * np.sqrt(self.M) * torch.sum(torch.abs(h))
            h_prev = torch.sparse.mm(self.Wres, h.T).T
            
            h = f_in + h_prev - a
            h = F.leaky_relu(F.layer_norm(h, h.size()))
            
            # print(f"f_in max: {torch.max(f_in)}, f_in min: {torch.min(f_in)}")
            # print(f"a: {a}")
            # print(f"h max: {torch.max(h_prev)}, h min: {torch.min(h_prev)}")
        
        
        # for f in fs[:]:
        #     f_in = f[:, 0, :] @ self.Win
            
        #     h = f_in + torch.sparse.mm(self.Wres, h.T).T
        #     h = F.leaky_relu(F.layer_norm(h, h.size()))
            # h = F.leaky_relu(h)
            
            # h_prev = h.reshape(h.size(0) * h.size(1), h.size(2))
            # h_prev = torch.sparse.mm(self.Wres, h_prev.T).T
            # h_prev = h_prev.reshape(fs[0].size(0), 197, self.M)
            # h = f_in + h_prev
            # h = f[:, 0, :] @ self.Win + torch.sparse.mm(self.Wres, h.T).T
            # h = F.relu(F.layer_norm(h, h.size()))

            # print(f"h max: {torch.max(h)}, h min: {torch.min(h)}")
        
        # h = torch.cat([f[:, 0, :] for f in fs], dim=-1) @ self.Win
        # h = F.relu(h)
        # print(f"h max: {torch.max(h)}, h min: {torch.min(h)}")
        
        # h = fs[10][:, 0, :] @ self.Win
        # h = F.relu(h)
        # f = self.attention(fs)
        # h = f[:, 0, :] @ self.Win
        
        return h
    
    def forward(self, x):
        h = self.get_esn_features(x)
        y = h @ self.Wout
        return y

    def tune_forward(self, x):
        self.get_features(x)
        f = self.attention(self.features)
        y = self.fc(f[:, 0, :])
        return y
    
    def freeze_backbone(self):
        for param in self.base_network.parameters():
            param.requires_grad = False
    
    def show_num_params(self,verbose=False):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Toal number of parameters: {total_params:,}")
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total number of trainable parameters: {trainable_params:,}")
    
class Learner:
    def __init__(self):
        self.model = Net()
        self.model = self.model.cuda()
        self.model.freeze_backbone()
        
        self._known_classes = 0
        self._classes_seen_so_far = 0
        self.class_increments = []
        
        self.C = torch.zeros(10000, 200)
        self.Q = torch.zeros(10000, 10000)
        self.I = torch.eye(10000)
    
    def train(self, datamanager):
        num_task = datamanager.nb_tasks - 1
        
        for task in range(num_task):
            self.model.show_num_params()
            
            self._classes_seen_so_far = self._known_classes + datamanager.get_task_size(task+1)
            self.class_increments.append((self._known_classes, self._classes_seen_so_far - 1))
            
            print(f"Learn classes: {self._known_classes} - {self._classes_seen_so_far - 1}")
            trainset = datamanager.get_dataset(
                np.arange(self._known_classes, self._classes_seen_so_far),
                source="train", mode="train")
            trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
            testset = datamanager.get_dataset(
                np.arange(self._known_classes, self._classes_seen_so_far),
                source="test", mode="test")
            testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)
            
            trainset_CPs = datamanager.get_dataset(
                np.arange(self._known_classes, self._classes_seen_so_far),
                source="train", mode="test")
            trainloader_CPs = DataLoader(trainset_CPs, batch_size=128, shuffle=False, num_workers=4)
            testset_CPs = datamanager.get_dataset(
                np.arange(0, self._classes_seen_so_far),
                source="test", mode="test")
            testloader_CPs = DataLoader(testset_CPs, batch_size=128, shuffle=False, num_workers=4)
            
            # if task == 0:
                # self.tune(trainloader, testloader, self._known_classes)
            self.fit(trainloader_CPs, testloader_CPs)
            
            # after task
            self._known_classes = self._classes_seen_so_far
    
    def fit(self, trainloader, testloader):
        ys = []
        hs = []
        for i, (_, x, y) in tqdm(enumerate(trainloader)):
            x, y = x.cuda(), y.cuda()
            with torch.no_grad():
                h = self.model.get_esn_features(x)
                hs.append(h.cpu())
            ys.append(y.cpu())
        
        ys = torch.cat(ys, dim=0)
        H = torch.cat(hs, dim=0)
        Y = F.one_hot(ys, num_classes=200).float()
        
        self.Q = self.Q + H.T @ H
        self.C = self.C + H.T @ Y
        ridge = 1e6
        Wout = torch.linalg.solve(self.Q + ridge * self.I, self.C)
        self.model.Wout = Wout.cuda()

        self.model.eval()
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
        acc_total, grouped = accuracy(y_pred.T[0], y_true, self._known_classes, self.class_increments)
        print(f"Acc total: {acc_total}, Acc grouped: {grouped}")
    
    def tune(self, trainloader, testloader, starting_label):
        # self.model.update_tune_fc()
        self.model.show_num_params()
        self.model.train()
        
        tune_epochs = 10
        
        body_lr = 0.01
        head_lr = 0.01
        weight_decay = 5e-4
        min_lr = 0.0
        
        optimizer = optim.SGD(
            self.model.parameters(), lr=body_lr, 
            momentum=0.9, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tune_epochs, eta_min=min_lr)
        
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
            train_acc = np.around(train_acc * 100 / len(trainloader.dataset), decimals=2)
            
            self.model.eval()
            test_acc = 0
            for i, (_, x, y) in enumerate(testloader):
                x, y = x.cuda(), y.cuda()
                # y -= starting_label
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