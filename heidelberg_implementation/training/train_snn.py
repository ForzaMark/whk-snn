import copy
import json
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
from util.save_plots import save_history_plot, save_loss_per_time_step_plot


def calculate_gradient(optimizer, loss_val):
    optimizer.zero_grad()
    loss_val.backward()

def update_weights(optimizer):
    optimizer.step()

def train(train_data, train_targets, net, optimizer, loss_configuration: Loss_Configuration, time_steps):
    data = train_data.to_dense().to(torch.float32).squeeze().permute(1, 0, 2).to(DEVICE)
    targets = train_targets.to(DEVICE)

    spk_recs, mem_recs = net(data)

    output_spk_rec = spk_recs[-1]
    output_mem_rec = mem_recs[-1]

    loss_result = calculate_loss(loss_configuration, 
                                 targets, 
                                 time_steps, 
                                 output_mem_rec, 
                                 output_spk_rec)

    if isinstance(loss_result, tuple):
        loss, loss_per_time_step = loss_result
    else:
        loss = loss_result
        loss_per_time_step = False

    calculate_gradient(optimizer=optimizer, loss_val=loss)
    update_weights(optimizer=optimizer)

    acc = compute_accuracy(train_data, train_targets, net, population_coding=loss_configuration=='population_coding')

    return loss.item(), acc, loss_per_time_step

def train_snn(net: ConfigurableSpikingNeuralNet, 
            num_epochs, 
            sparsity: float=0,
            time_steps = TIME_STEPS,
            save_model: Union[bool, str]=False, 
            save_model_per_epoch: Union[bool, str]=False,
            save_plots: Union[bool, str]=False, 
            additional_output_information={}, 
            output_file_path='output/generic_output_results.json',
            loss_configuration: Loss_Configuration ='membrane_potential_cross_entropy'):
    
    if sparsity != 0:
        net = apply_random_weight_pruning_mask(net, sparsity)

    early_stopper = EarlyStopping(patience=3, min_delta=0.01)
    
    train_data_loader, test_data_loader = create_data_loader(time_steps=time_steps)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

    loss_history = []
    acc_history = []
    best_model: ConfigurableSpikingNeuralNet = net
    best_test_accuracy = 0
    early_stopped_number_epoch = 0
    epoch_loss_per_time_step = []

    start = datetime.now()

    for epoch in range(100 if num_epochs == 'early_stopping' else num_epochs):
        print(f"Epoch: {epoch}")

        loss_per_time_step_hist = []
        total_samples = 0
        epoch_loss = 0.0
        epoch_acc = 0.0

        for data, targets in train_data_loader:
            loss_val, acc, loss_per_time_step = train(data, targets, net, optimizer, loss_configuration, time_steps)
            batch_size = data.size(0)

            epoch_loss += loss_val * batch_size
            epoch_acc += acc * batch_size

            total_samples += batch_size

            if loss_per_time_step and loss_per_time_step_hist:
                loss_per_time_step_hist.append(loss_per_time_step)
            else:
                loss_per_time_step_hist = False

        if loss_per_time_step_hist:
            epoch_loss_per_time_step.append(compute_epoch_loss_per_time_step(loss_per_time_step_hist, epoch=epoch, time_steps=time_steps))
        
        test_accuracy = compute_test_set_accuracy(test_data_loader, net, population_coding=loss_configuration=='population_coding')
        avg_epoch_loss = epoch_loss / total_samples
        avg_epoch_train_acc = epoch_acc / total_samples

        loss_history.append(avg_epoch_loss)
        acc_history.append(avg_epoch_train_acc)
        
        print(f"loss {avg_epoch_loss}")
        print(f"train accuracy {avg_epoch_train_acc}")
        print(f"test accuracy {test_accuracy}")

        if num_epochs == 'early_stopping':
            if test_accuracy > best_test_accuracy:
                best_model = copy.deepcopy(net)
                best_test_accuracy = test_accuracy

            early_stopper(test_accuracy)

            if early_stopper.early_stop:
                print("Early stopping triggered.")
                early_stopped_number_epoch = epoch
                break
        else:
            best_model = net

        if isinstance(save_model_per_epoch, str):
            best_model_to_be_saved = copy.deepcopy(best_model)

            if sparsity != 0:
                best_model_to_be_saved = make_pruning_permanent(best_model_to_be_saved)

            torch.save(best_model_to_be_saved.state_dict(), f'{save_model_per_epoch}_epoch_{epoch}.pth')
            
    if sparsity != 0:
        best_model = make_pruning_permanent(best_model)
        print_model_sparsity(best_model)

    end = datetime.now()
    time_diff = end - start

    if isinstance(save_plots, str):
        save_history_plot(loss_history, path=f'{save_plots}_simplified_loss.png')
        save_history_plot(acc_history, path=f'{save_plots}_simplified_accuracy.png')
        if epoch_loss_per_time_step:
            save_loss_per_time_step_plot(epoch_loss_per_time_step, path=f'{save_plots}_loss_per_time_steps.png')

    data = {
        'epochs': early_stopped_number_epoch if num_epochs == 'early_stopping' else num_epochs,
        'training_accuracy': acc_history[-1],
        'test_accuracy': best_test_accuracy,
        'time':  time_diff.total_seconds(),
        **additional_output_information
    }

    if isinstance(save_model, str):
        torch.save(best_model.state_dict(), f'{save_model}.pth')

    with open(output_file_path, "w") as file:
        json.dump(data, file, indent=4) 