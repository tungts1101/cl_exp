import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm
import argparse


class Program(nn.Module):
    def __init__(self, img_shape=(3, 224, 224), mask_size=50):
        super().__init__()
        if isinstance(mask_size, int):
            mask_size = (mask_size, mask_size)
        
        self.mask_size = mask_size
        self.img_shape = img_shape

        self.W = nn.Parameter(torch.randn(1, *img_shape), requires_grad=True)
        self.M = nn.Parameter(torch.ones(1, *img_shape), requires_grad=False)
        self.M[:, :, 
               (img_shape[1] - mask_size[0])//2:(img_shape[1] + mask_size[0])//2, 
               (img_shape[2] - mask_size[1])//2:(img_shape[2] + mask_size[1])//2] = 0
    
    def forward(self, x):
        B, C, H, W = x.size()
        if C != self.img_shape[0]:
            x = x.repeat(1, self.img_shape[0] // C, 1, 1)
        x_scaled = F.interpolate(x, size=self.mask_size, mode='bicubic', align_corners=False)
        background = torch.zeros(B, *self.img_shape).to(x.device)
        background[:, :, 
                   (self.img_shape[1] - self.mask_size[0])//2:(self.img_shape[1] + self.mask_size[0])//2,
                   (self.img_shape[2] - self.mask_size[1])//2:(self.img_shape[2] + self.mask_size[1])//2] = x_scaled

        x_perturbed = background + torch.tanh(self.W * self.M)
        return x_perturbed

class Learner(object):
    def __init__(self, cfg):
        self.cfg = cfg
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.program = Program().to(self.device)
        
        if cfg["dataset_from"] == "imagenet" and cfg["model"] == "resnet18":
            self.model =  models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(self.device)
        elif cfg["dataset_from"] == "imagenet" and cfg["model"] == "resnet50":
            self.model =  models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(self.device)
        elif cfg["dataset_from"] == "imagenet" and cfg["model"] == "resnet101":
            self.model =  models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1).to(self.device)
        elif cfg["dataset_from"] == "imagenet" and cfg["model"] == "resnet152":
            self.model =  models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1).to(self.device)
        elif cfg["dataset_from"] == "imagenet" and cfg["model"] == "vit_base_patch16_224":
            self.model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=1000).to(self.device)
        else:
            raise NotImplementedError(f"Model {cfg['model']} from {cfg['dataset_from']} is not supported.")
        self.model.requires_grad_(False)
        
        transform = transforms.ToTensor()
        if cfg["dataset_to"] == "mnist":
            train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
            test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
        elif cfg["dataset_to"] == "cifar10":
            train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
            test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
        elif cfg["dataset_to"] == "cifar100":
            train_dataset = datasets.CIFAR100(root="./data", train=True, transform=transform, download=True)
            test_dataset = datasets.CIFAR100(root="./data", train=False, transform=transform, download=True)
        else:
            raise NotImplementedError(f"Dataset {cfg['dataset']} is not supported.")
        self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    def label_mapping(self, y):
        if self.cfg["dataset_from"] == "imagenet" and self.cfg["dataset_to"] == "mnist":
            # return y[:, :10]
            B, C = y.size()
            y = y.view(B, 10, C // 10)
            y = y.sum(dim=2)
            return y
        elif self.cfg["dataset_from"] == "imagenet" and self.cfg["dataset_to"] == "cifar10":
            B, C = y.size()
            y = y.view(B, 10, C // 10)
            y = y.sum(dim=2)
            return y
        elif self.cfg["dataset_from"] == "imagenet" and self.cfg["dataset_to"] == "cifar100":
            return y[:, :100]
        else:
            raise NotImplementedError(f"Mapping from {self.cfg['dataset_from']} to {self.cfg['dataset_to']} is not supported.")
    
    def get_transform(self):
        if self.cfg["dataset_to"] == "mnist":
            return transforms.Compose([
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor()
            ])
        else:
            return transforms.Compose([
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
    
    def normalize(self, x_adv):
        if self.cfg["dataset_from"] == "imagenet":
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x_adv.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x_adv.device)
            x_adv = (x_adv - mean) / std
        else:
            raise NotImplementedError(f"Normalization for {self.cfg['dataset_from']} is not supported.")
        return x_adv
    
    def compute_loss(self, y_adv, y):
        nll_loss = F.cross_entropy(y_adv, y)
        reg_loss = self.cfg["lambda"] * torch.linalg.vector_norm(self.program.W.view(-1), ord=2) ** 2
        loss = nll_loss + reg_loss
        return {"loss": loss, "nll_loss": nll_loss, "reg_loss": reg_loss}
    
    def train(self):
        optimizer = optim.Adam(self.program.parameters(), lr=self.cfg["lr"], weight_decay=5e-4)
        # optimizer = optim.SGD(self.program.parameters(), self.cfg["lr"], momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg["epochs"], eta_min=0.0)
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1)
        
        # optimizer = optim.Adam(self.program.parameters(), lr=self.cfg["lr"], weight_decay=5e-4)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
        
        pbar = tqdm(range(self.cfg["epochs"]))
        for epoch in pbar:
            self.program.train()
            total_loss, correct, total = 0, 0, 0
            total_nll_loss, total_reg_loss = 0, 0

            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)

                x_adv = self.program(x)
                x_adv = self.normalize(x_adv)
                y_adv = self.model(x_adv)
                y_adv = self.label_mapping(y_adv)
                
                loss = self.compute_loss(y_adv, y)

                optimizer.zero_grad()
                loss["loss"].backward()
                optimizer.step()

                total_loss += loss["loss"].item()
                correct += (y_adv.argmax(dim=1) == y).sum().item()
                total_nll_loss += loss["nll_loss"].item()
                total_reg_loss += loss["reg_loss"].item()
                total += y.size(0)

                info = f"Epoch {epoch+1}: Loss {total_loss/total:.4f}, Nll Loss {total_nll_loss/total:.4f}, Reg Loss {total_reg_loss/total:.4f}, Accuracy {100 * correct/total:.2f}%"
                pbar.set_description(info)
                
            scheduler.step()
    
    def test(self):
        self.program.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)

                x_adv = self.program(x)
                x_adv = self.normalize(x_adv)
                y_adv = self.model(x_adv)
                y_adv = self.label_mapping(y_adv)
                
                correct += (y_adv.argmax(dim=1) == y).sum().item()
                total += y.size(0)

        print(f"Test Accuracy: {100 * correct/total:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", "-e", type=int, default=10)
    parser.add_argument("--to", "-t", type=str, default="mnist", choices=["mnist", "cifar10", "cifar100"])
    parser.add_argument("--lambda", "-l", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--model", "-m", type=str, default="resnet18", choices=[
        "resnet18", 
        "resnet50", 
        "resnet101", 
        "resnet152",
        "vit_base_patch16_224",])
    args = parser.parse_args()
    cfg = vars(args)
    
    cfg["dataset_to"] = cfg["to"]
    
    if "dataset_from" not in cfg:
        cfg["dataset_from"] = "imagenet"
    if "dataset_to" not in cfg:
        cfg["dataset_to"] = "mnist"
    if "model" not in cfg:
        cfg["model"] = "resnet18"
    
    leaner = Learner(cfg)
    leaner.train()
    leaner.test()
    