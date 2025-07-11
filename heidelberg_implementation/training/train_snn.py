import copy
from datetime import datetime
from typing import Union

import torch
from constants import DEVICE, TIME_STEPS
from neural_nets.configurable_spiking_neural_net import ConfigurableSpikingNeuralNet
from training.pruning import (
    apply_random_weight_pruning_mask,
    make_pruning_permanent,
    print_model_sparsity,
)
from util.calculate_loss import Loss_Configuration, calculate_loss
from util.compute_accuracy import compute_accuracy, compute_test_set_accuracy
from util.compute_epoch_loss_per_time_step import compute_epoch_loss_per_time_step
from util.create_data_loader import create_data_loader
from util.early_stopping import EarlyStopping


def calculate_gradient(optimizer, loss_val):
    optimizer.zero_grad()
    loss_val.backward()


def update_weights(optimizer):
    optimizer.step()


def train(
    train_data,
    train_targets,
    net,
    optimizer,
    loss_configuration: Loss_Configuration,
    time_steps,
):
    data = train_data.to_dense().to(torch.float32).squeeze().permute(1, 0, 2).to(DEVICE)
    targets = train_targets.to(DEVICE)

    spk_recs, mem_recs = net(data)

    output_spk_rec = spk_recs[-1]
    output_mem_rec = mem_recs[-1]

    loss_result = calculate_loss(
        loss_configuration, targets, time_steps, output_mem_rec, output_spk_rec
    )

    if isinstance(loss_result, tuple):
        loss, loss_per_time_step = loss_result
    else:
        loss = loss_result
        loss_per_time_step = False

    calculate_gradient(optimizer=optimizer, loss_val=loss)
    update_weights(optimizer=optimizer)

    acc = compute_accuracy(
        train_data,
        train_targets,
        net,
        population_coding=loss_configuration == "population_coding",
    )

    return loss.item(), acc, loss_per_time_step


def create_current_net_copy(net, sparsity):
    model_to_be_saved = copy.deepcopy(net)
    if sparsity != 0:
        model_to_be_saved = make_pruning_permanent(model_to_be_saved)
        print_model_sparsity(model_to_be_saved)

    return model_to_be_saved


def evaluate_loop_condition(num_epochs, current_epoch, early_stopper):
    if num_epochs == "early_stopping":
        return not early_stopper.early_stop
    else:
        return current_epoch <= num_epochs


def train_snn(
    net: ConfigurableSpikingNeuralNet,
    num_epochs,
    sparsity: float = 0,
    time_steps=TIME_STEPS,
    loss_configuration: Loss_Configuration = "membrane_potential_cross_entropy",
    use_train_data_subset: Union[bool, int] = False,
):

    if sparsity != 0:
        net = apply_random_weight_pruning_mask(net, sparsity)

    early_stopper = EarlyStopping(patience=3, min_delta=0.01)

    train_data_loader, test_data_loader = create_data_loader(
        time_steps=time_steps, use_train_subset=use_train_data_subset
    )

    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

    loss_history = []
    training_acc_history = []
    test_acc_history = []
    models_per_epoch = []
    current_epoch = 0

    epoch_loss_per_time_step = []

    start = datetime.now()

    while evaluate_loop_condition(num_epochs, current_epoch, early_stopper):
        print(f"Epoch: {current_epoch}")

        loss_per_time_step_hist = []
        total_samples = 0
        epoch_loss = 0.0
        epoch_acc = 0.0

        for data, targets in train_data_loader:
            loss_val, acc, loss_per_time_step = train(
                data, targets, net, optimizer, loss_configuration, time_steps
            )
            batch_size = data.size(0)

            epoch_loss += loss_val * batch_size
            epoch_acc += acc * batch_size

            total_samples += batch_size

            if loss_per_time_step and loss_per_time_step_hist:
                loss_per_time_step_hist.append(loss_per_time_step)
                epoch_loss_per_time_step.append(
                    compute_epoch_loss_per_time_step(
                        loss_per_time_step_hist,
                        epoch=current_epoch,
                        time_steps=time_steps,
                    )
                )
            else:
                loss_per_time_step_hist = False

        test_accuracy = compute_test_set_accuracy(
            test_data_loader,
            net,
            population_coding=loss_configuration == "population_coding",
        )
        avg_epoch_loss = epoch_loss / total_samples
        avg_epoch_train_acc = epoch_acc / total_samples

        loss_history.append(avg_epoch_loss)
        training_acc_history.append(avg_epoch_train_acc)
        test_acc_history.append(test_accuracy)
        models_per_epoch.append(create_current_net_copy(net, sparsity))

        print(f"loss {avg_epoch_loss}")
        print(f"train accuracy {avg_epoch_train_acc}")
        print(f"test accuracy {test_accuracy}")

        if num_epochs == "early_stopping":
            early_stopper(test_accuracy)

        current_epoch += 1

    end = datetime.now()
    total_training_time = end - start

    return (
        training_acc_history,
        test_acc_history,
        loss_history,
        total_training_time,
        epoch_loss_per_time_step,
        models_per_epoch,
    )
