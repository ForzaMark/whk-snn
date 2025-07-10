import numpy as np
import torch
from constants import DEVICE, HEIDELBERG_DATASET_NUMBER_CLASSES
from snntorch import functional as SF


def compute_accuracy(data, targets, net, population_coding):
    with torch.no_grad():
        net.eval()

        data = data.to_dense().to(torch.float32).squeeze().permute(1, 0, 2).to(DEVICE)
        targets = targets.to(DEVICE)

        spk_recs, _ = net(data)

        output_spk_rec = spk_recs[-1]

        if population_coding:
            acc = SF.accuracy_rate(output_spk_rec, targets, population_code=True, num_classes=HEIDELBERG_DATASET_NUMBER_CLASSES)
        else:
            acc = SF.accuracy_rate(output_spk_rec, targets)

        return acc


def compute_test_set_accuracy(test_data_generator, net, population_coding):
    with torch.no_grad():
        net.eval()

        per_batch_test_acc = []

        for data, targets in test_data_generator:
            per_batch_test_acc.append(compute_accuracy(data, targets, net, population_coding))

    return np.mean(per_batch_test_acc)
