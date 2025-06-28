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