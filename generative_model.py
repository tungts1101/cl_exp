import torch
from torch import nn
from torch import distributions
import torch.nn.functional as F
from typing import Optional
from abc import ABC, abstractmethod


def mlp_block(in_dim: int, out_dim: int, norm: bool=True, activation: Optional[nn.Module] = nn.ReLU(inplace=True)):
    mods = [nn.Linear(in_dim, out_dim, bias=False)]
    if norm:
        mods.append(nn.LayerNorm(out_dim))
    if activation is not None:
        mods.append(activation)
    return nn.Sequential(*mods)


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_dim = cfg["generative_feature_dim"] + cfg["generative_num_classes"]
        hid    = cfg["generative_hidden_dim"]
        z_dim  = cfg["generative_latent_dim"]
        self.net        = nn.Sequential(
            mlp_block(in_dim, hid * 2), 
            mlp_block(hid * 2, hid * 2), 
            mlp_block(hid * 2, hid))
        self.fc_mu      = nn.Linear(hid, z_dim)
        self.fc_logvar  = nn.Linear(hid, z_dim)

    def forward(self, x, y1h):
        h  = self.net(torch.cat([x, y1h], dim=1))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp_(-10, 10)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_dim = cfg["generative_latent_dim"] + cfg["generative_num_classes"]
        hid    = cfg["generative_hidden_dim"]
        out_dim= cfg["generative_feature_dim"]
        self.net = nn.Sequential(
            mlp_block(in_dim, hid * 2),
            mlp_block(hid * 2, hid * 2),
            mlp_block(hid * 2, out_dim, activation=None)      # linear output
        )

    def forward(self, z, y1h):
        return self.net(torch.cat([z, y1h], dim=1))


class ConditionalVAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.enc   = Encoder(cfg)
        self.dec   = Decoder(cfg)
        self.z_dim = cfg["generative_latent_dim"]

    def reparameterise(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def forward(self, x, y1h):
        mu, logvar = self.enc(x, y1h)
        z          = self.reparameterise(mu, logvar)
        recon      = self.dec(z, y1h)
        return recon, mu, logvar


class BaseFeatureGenerator(ABC):
    """Base class for feature generators used in classifier alignment"""
    
    @abstractmethod
    def update(self, features: torch.Tensor, labels: torch.Tensor):
        """Update the generator with new features and labels"""
        pass
    
    @abstractmethod
    def sample(self, n_samples: int, class_idx: int) -> torch.Tensor:
        """Sample n_samples features for the given class"""
        pass
    
    @abstractmethod
    def can_sample(self, class_idx: int) -> bool:
        """Check if we can sample from the given class"""
        pass


class MultivariateNormalGenerator(BaseFeatureGenerator):
    """Multivariate Normal feature generator for classifier alignment"""
    
    def __init__(self, feature_dim: int, num_classes: int, device="cuda"):
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.device = device
        
        # Store class statistics
        self._class_means = {}
        self._class_covs = {}
        self._class_counts = {}
    
    def update(self, features: torch.Tensor, labels: torch.Tensor):
        """Update class statistics with new features"""
        features = features.to(self.device)
        labels = labels.to(self.device)
        
        for class_idx in labels.unique():
            class_idx = class_idx.item()
            class_mask = (labels == class_idx)
            class_features = features[class_mask]
            
            if class_features.shape[0] > 1:  # Need at least 2 samples for covariance
                class_mean = class_features.mean(dim=0)
                class_cov = torch.cov(class_features.T)
                
                # Ensure positive definiteness
                min_eigenval = torch.linalg.eigvals(class_cov).real.min()
                if min_eigenval <= 0:
                    reg_term = abs(min_eigenval) + 1e-3
                else:
                    reg_term = 1e-4
                class_cov = class_cov + torch.eye(class_features.shape[1], device=self.device) * reg_term
                
                # Verify positive definiteness
                try:
                    torch.linalg.cholesky(class_cov)
                except RuntimeError:
                    # Fallback: use diagonal covariance
                    class_var = class_features.var(dim=0, unbiased=True)
                    class_cov = torch.diag(class_var + 1e-3)
                
                self._class_means[class_idx] = class_mean
                self._class_covs[class_idx] = class_cov
                self._class_counts[class_idx] = class_features.shape[0]
            
            elif class_features.shape[0] == 1:
                # Single sample: use identity covariance
                self._class_means[class_idx] = class_features.squeeze(0)
                self._class_covs[class_idx] = torch.eye(self.feature_dim, device=self.device) * 1e-2
                self._class_counts[class_idx] = 1
    
    def can_sample(self, class_idx: int) -> bool:
        """Check if we can sample from the given class"""
        return class_idx in self._class_means
    
    def sample(self, n_samples: int, class_idx: int) -> torch.Tensor:
        """Sample features from the multivariate normal distribution for a given class"""
        if not self.can_sample(class_idx):
            raise ValueError(f"Class {class_idx} has not been updated yet or has insufficient data.")
        
        cls_mean = self._class_means[class_idx]
        cls_cov = self._class_covs[class_idx]
        
        try:
            # Sample from multivariate normal distribution
            dist = distributions.MultivariateNormal(cls_mean.float(), cls_cov.float())
            samples = dist.sample((n_samples,))
            return samples
        except Exception as e:
            # Fallback: sample from simple normal around mean
            noise_scale = 0.1
            noise = torch.randn(n_samples, cls_mean.shape[0], device=self.device) * noise_scale
            samples = cls_mean.unsqueeze(0) + noise
            return samples
    
    def get_class_stats(self, class_idx: int):
        """Get statistics for a specific class"""
        if not self.can_sample(class_idx):
            return None
        return {
            'mean': self._class_means[class_idx],
            'cov': self._class_covs[class_idx],
            'count': self._class_counts[class_idx]
        }


class VAEFeatureGenerator(BaseFeatureGenerator):
    """VAE-based feature generator for classifier alignment"""
    
    def __init__(self, vae_model: ConditionalVAE, feature_dim: int, num_classes: int, device="cuda"):
        self.vae = vae_model
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.device = device
        self._updated_classes = set()
    
    def update(self, features: torch.Tensor, labels: torch.Tensor):
        """Update the VAE with new features (this is a placeholder - actual VAE training happens elsewhere)"""
        # For VAE, the training happens in the main training loop
        # This method just tracks which classes have been seen
        for class_idx in labels.unique():
            self._updated_classes.add(class_idx.item())
    
    def can_sample(self, class_idx: int) -> bool:
        """Check if we can sample from the given class"""
        return class_idx in self._updated_classes
    
    def sample(self, n_samples: int, class_idx: int) -> torch.Tensor:
        """Sample features using the VAE for a given class"""
        if not self.can_sample(class_idx):
            raise ValueError(f"Class {class_idx} has not been seen during training.")
        
        self.vae.eval()
        with torch.no_grad():
            # Sample from latent space
            z = torch.randn(n_samples, self.vae.z_dim, device=self.device)
            
            # Create one-hot encoding for the class
            class_labels = torch.full((n_samples,), class_idx, device=self.device)
            y_onehot = F.one_hot(class_labels, num_classes=self.num_classes).float()
            
            # Generate features
            generated_features = self.vae.dec(z, y_onehot)
            return generated_features


