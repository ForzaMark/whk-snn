import torch
import torch.nn as nn
from constants import DEVICE, DTYPE, HEIDELBERG_DATASET_NUMBER_CLASSES
from snntorch import functional as SF

CROSS_ENTROPY_LOSS = nn.CrossEntropyLoss()
RATE_CODE_LOSS = SF.ce_rate_loss()
POPULATION_CODING_LOSS = SF.mse_count_loss(
    correct_rate=1.0,
    incorrect_rate=0.0,
    population_code=True,
    num_classes=HEIDELBERG_DATASET_NUMBER_CLASSES,
)


def sum_loss_over_timesteps(time_steps, mem_rec, targets, loss_function):
    loss_per_time_step = []
    loss_val = torch.zeros((1), dtype=DTYPE, device=DEVICE)

    for step in range(time_steps):
        loss = loss_function(mem_rec[step], targets)
        loss_val += loss
        loss_per_time_step.append(loss.item())

    return loss_val, loss_per_time_step


def calculate_rate_code_loss(spk_rec, targets):
    loss_val = RATE_CODE_LOSS(spk_rec, targets)

    return loss_val


def calculate_population_coding_loss(spk_rec, targets):
    loss_val = POPULATION_CODING_LOSS(spk_rec, targets)
    return loss_val


def calculate_loss(
    loss_configuration,
    targets,
    time_steps,
    output_mem_rec,
    output_spk_rec,
):
    if loss_configuration == "membrane_potential_cross_entropy":
        return sum_loss_over_timesteps(
            time_steps, output_mem_rec, targets, CROSS_ENTROPY_LOSS
        )
    elif loss_configuration == "rate_code_cross_entropy":
        return calculate_rate_code_loss(output_spk_rec, targets)
    else:
        return calculate_population_coding_loss(output_spk_rec, targets)
