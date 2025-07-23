import torch

logits = torch.rand(4, 10, 6)
max_logit_values, max_logit_indices = torch.max(logits, dim=-1)
task_indices_offset = torch.arange(10, dtype=torch.long) * 20
max_logit_indices += task_indices_offset

print(max_logit_values.shape, max_logit_indices.shape)
print(max_logit_values)
print(max_logit_indices)

top2_logit_values, top2_logit_indices = torch.topk(max_logit_values, 2, dim=-1)
print(top2_logit_values.shape)
print(top2_logit_indices)
mask = (top2_logit_values[:, 0] - top2_logit_values[:, 1]) > 0.005
print(mask)

selected_indices = max_logit_indices[torch.arange(max_logit_indices.shape[0]), top2_logit_indices[:, 0]]

predicts = torch.ones(4, dtype=torch.long) * -100
predicts[mask] = selected_indices[mask]
print(predicts)