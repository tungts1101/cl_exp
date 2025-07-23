import os
import torch
from tqdm import tqdm

import open_clip

from datasets.templates import get_templates
from datasets.registry import get_dataset

model, train_preprocess, val_preprocess = open_clip.create_model_and_transforms("ViT-B-16", pretrained='openai', cache_dir='cache')

template = get_templates("CIFAR100")

classnames = get_dataset("CIFAR100", None, location="/media/ellen/datasets").classnames
model.eval()
model.cuda()

print('Building classification head.')
with torch.no_grad():
    zeroshot_weights = []
    for classname in tqdm(classnames):
        texts = []
        for t in template:
            texts.append(t(classname))
        texts = open_clip.tokenize(texts).cuda()
        embeddings = model.encode_text(texts) # embed with text encoder
        print(embeddings.shape)
        embeddings /= embeddings.norm(dim=-1, keepdim=True)

        embeddings = embeddings.mean(dim=0, keepdim=True)
        embeddings /= embeddings.norm()

        zeroshot_weights.append(embeddings)
    zeroshot_weights = torch.stack(zeroshot_weights, dim=0).cuda()
    print(zeroshot_weights.shape)
    zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

    zeroshot_weights *= model.logit_scale.exp()
    
    zeroshot_weights = zeroshot_weights.squeeze().float()
    zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)
    print(zeroshot_weights.shape)

#     classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights)

#     return classification_head