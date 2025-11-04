import gc

import torch


def remove_training_from_memory(model, optimizer):
    del model
    del optimizer
    torch.cuda.empty_cache()
    gc.collect()
