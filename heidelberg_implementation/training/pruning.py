import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


def apply_random_weight_pruning_mask(net, sparsity):
    for i, layer in enumerate(net.linears):
            if isinstance(layer, nn.Linear):
                prune.random_unstructured(layer, name="weight", amount=sparsity)

    return net

def make_pruning_permanent(net):
    for i, layer in enumerate(net.linears):
        if isinstance(layer, nn.Linear):
            prune.remove(layer, "weight")
    
    return net

def print_model_sparsity(model):
    total_params = 0
    zero_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            numel = param.numel()
            zeros = torch.sum(param == 0).item()
            total_params += numel
            zero_params += zeros
            print(f"{name}: {zeros}/{numel} ({100.0 * zeros / numel:.2f}%) zeros")
    print(f"Total sparsity: {100.0 * zero_params / total_params:.2f}% ({zero_params}/{total_params})")
