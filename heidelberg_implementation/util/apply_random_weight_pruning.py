import torch


def apply_random_weight_pruning(model, sparsity):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name and param.requires_grad:
                mask = (torch.rand_like(param) > sparsity).float()
                param.mul_(mask)