# import torch
# import torch.nn as nn
# from transformers import ViTModel

# class ViTRNN(nn.Module):
#     def __init__(self, rnn_hidden_size, num_classes):
#         super(ViTRNN, self).__init__()
#         self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')  # Pretrained ViT
#         self.rnn = nn.LSTM(input_size=self.vit.config.hidden_size,
#                            hidden_size=rnn_hidden_size,
#                            batch_first=True)
#         self.fc = nn.Linear(rnn_hidden_size, num_classes)

#     def forward(self, images):
#         vit_outputs = self.vit(images)  # (batch_size, num_patches, hidden_size)
#         patch_embeddings = vit_outputs.last_hidden_state  # Sequential input for RNN
#         rnn_output, _ = self.rnn(patch_embeddings)  # (batch_size, seq_len, rnn_hidden_size)
#         aggregated_features = rnn_output[:, -1, :]  # Use the last output for classification
#         print(rnn_output.shape)
#         print(aggregated_features.shape)
#         logits = self.fc(aggregated_features)
#         return logits

# model = ViTRNN(rnn_hidden_size=256, num_classes=10)
# images = torch.randn(32, 3, 224, 224)  # (batch_size, channels, height, width)

# logits = model(images)
# print(logits.shape)  # (batch_size, num_classes)

# from transformers import CLIPModel, CLIPProcessor
# import torch

# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
# image_encoder = model.vision_model
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
# image = torch.randn(1, 3, 224, 224)

# with torch.no_grad():
#     image_features = image_encoder(image)

# print(image_features.last_hidden_state.shape)

# import torch

# queries = torch.randn(32, 768)
# positives = torch.randn(32, 768)
# negatives = torch.randn(200, 768)

# # scores = torch.einsum("ij,kj->ij", queries, negatives)
# scores = torch.einsum("ij,kj -> ij", queries, positives)
# print(scores.shape)

import torch
from torch import nn
import torch.nn.functional as F

queries = torch.ones(2, 4)
positives = torch.ones(2, 4)
negatives = torch.zeros(3, 4)

# queries = F.normalize(queries, dim=-1)
# positives = F.normalize(positives, dim=-1)
# negatives = F.normalize(negatives, dim=-1)
# negatives = negatives.unsqueeze(0).expand(2, -1, -1)

# "ij,ij->i"
# "ij,ikj->ik"
# positive_scores = torch.einsum("ij,ij->i", queries, positives)
# negative_scores = torch.einsum("ij,ikj->ik", queries, negatives)
# print(positive_scores)
# print(negative_scores)

# def info_nce_loss(queries, positives, negatives, temperature=0.7):
#     queries = F.normalize(queries, dim=-1)
#     positives = F.normalize(positives, dim=-1)
#     negatives = F.normalize(negatives, dim=-1)
#     negatives = negatives.unsqueeze(0).expand(queries.shape[0], -1, -1)
    
#     positive_scores = torch.einsum("ij,ij->i", queries, positives) / temperature
#     negative_scores = torch.einsum("ij,ikj->ik", queries, negatives) / temperature
    
#     logits = torch.cat([positive_scores.unsqueeze(1), negative_scores], dim=1)
#     labels = torch.zeros(logits.shape[0], dtype=torch.long)

#     print(logits)
#     print(labels)
    
#     return F.cross_entropy(logits, labels)

# loss = info_nce_loss(queries, positives, negatives, temperature=1.0)
# print(loss)


class PadPrompt(nn.Module):
    def __init__(self, target_shape, pad_size):
        super().__init__()
        self.target_shape = target_shape
        self.pad_size = pad_size
        
        self.pad = nn.Parameter(torch.randn(1, *target_shape))
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        x_resized = F.interpolate(x, size=(H - 2 * self.pad_size, W - 2 * self.pad_size), mode='bilinear')
        x_padded = F.pad(x_resized, (self.pad_size, self.pad_size, self.pad_size, self.pad_size), value=0)
        mask = torch.ones_like(x_resized)
        mask = F.pad(mask, (self.pad_size, self.pad_size, self.pad_size, self.pad_size), value=0)
        
        # pad = self.pad.expand(B, C, -1, -1)
        x_prompted = x_padded * mask + self.pad * (1 - mask)
        
        return x_prompted

class BlockPrompt(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size
        self.block = nn.Parameter(torch.randn(1, 1, block_size, block_size))
    
    def forward(self, x):
        B, C, H, W = x.shape
        block = self.block.expand(B, C, self.block_size, self.block_size)
        
        x_prompted = x.clone()
        start_h = (H - self.block_size) // 2
        start_w = (W - self.block_size) // 2
        end_h = start_h + self.block_size
        end_w = start_w + self.block_size
        x_prompted[:, :, start_h:end_h, start_w:end_w] = block
        
        return x_prompted

class FilterPrompt(nn.Module):
    def __init__(self, filter_size):
        super().__init__()
        self.filter_size = filter_size
        self.filter = nn.Parameter(torch.randn(1, *filter_size))
    
    def forward(self, x):
        B, C, H, W = x.shape
        filter = self.filter.expand(B, -1, -1, -1)
        
        x_prompted = x * filter
        return x_prompted

input_shape = (5, 5)

# prompter = PadPrompt(target_shape=input_shape, pad_size=1)
# prompter = BlockPrompt(block_size=3)
prompter = FilterPrompt(filter_size=input_shape)
x = torch.ones(1, 3, input_shape[0], input_shape[1])
output = prompter(x)
print(output.shape)  # (32, 3, 224, 224)
print(output)