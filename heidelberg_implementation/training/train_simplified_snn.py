from snntorch import functional as SF
import torch.nn as nn
from util.utils import save_history_plot, save_loss_per_time_step_plot
import torch
from torch.utils.data import DataLoader
from tonic import datasets, transforms
import json
from datetime import datetime
from constants import DEVICE, DTYPE, TIME_STEPS, BATCH_SIZE
from typing import Union
from util.early_stopping import EarlyStopping
import copy
import numpy as np

def compute_epoch_loss_per_time_step_data(loss_per_time_step_hist, epoch):
    last_element = loss_per_time_step_hist[-1]
    first_element = loss_per_time_step_hist[0]
    averaged_element = []

    for i in range(TIME_STEPS):
        averaged_element.append(np.mean(np.array(loss_per_time_step_hist)[:, i]))

    return ({
        'epoch': epoch,
        'loss_per_time_step_last_element': last_element,
        'loss_per_time_step_first_element': first_element,
        'loss_per_time_step_averaged_element': averaged_element
    })

def compute_test_set_accuracy(test_data_generator, net):
    total = 0
    correct = 0

    with torch.no_grad():
        net.eval()
        for data, targets in test_data_generator:
            data = data.to_dense().to(torch.float32).squeeze().permute(1, 0, 2).to(DEVICE)
            targets = targets.to(DEVICE)

            test_spk_recs, _ = net(data)

            output_test_spk = test_spk_recs[-1]

            _, predicted = output_test_spk.sum(dim=0).max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()


    return 100 * correct / total

def calculate_loss_over_all_timesteps(time_steps, mem_rec, targets, loss_function):
    loss_per_time_step = []
    loss_val = torch.zeros((1), dtype=DTYPE, device=DEVICE)

    for step in range(time_steps):
        loss = loss_function(mem_rec[step], targets)
        loss_val += loss
        loss_per_time_step.append(loss.item())

    return loss_val, loss_per_time_step

def calculate_gradient(optimizer, loss_val):
    optimizer.zero_grad()
    loss_val.backward()

def update_weights(optimizer):
    optimizer.step()

def train(data, targets, net, optimizer):
    data = data.to_dense().to(torch.float32).squeeze().permute(1, 0, 2).to(DEVICE)
    targets = targets.to(DEVICE)

    spk_recs, mem_recs = net(data)

    output_spk_rec = spk_recs[-1]
    output_mem_rec = mem_recs[-1]

    total_loss, loss_per_time_step = calculate_loss_over_all_timesteps(TIME_STEPS, output_mem_rec, targets, LOSS_FUNCTION)

    calculate_gradient(optimizer=optimizer, loss_val=total_loss)
    update_weights(optimizer=optimizer)
    
    acc = SF.accuracy_rate(output_spk_rec, targets)

    return total_loss.item(), acc, loss_per_time_step

frame_transform = transforms.ToFrame(
    sensor_size=datasets.SHD.sensor_size,  
    n_time_bins=TIME_STEPS
)

LOSS_FUNCTION = nn.CrossEntropyLoss()

def train_simplified_snn(net, num_epochs, save_model: Union[bool, str]=False, save_plots: Union[bool, str]=False, additional_output_information={}, output_file_path='output/simplified_results.json'):
    early_stopper = EarlyStopping(patience=3, min_delta=0.01)
    
    train_data = datasets.SHD("./data", transform=frame_transform, train=True)
    test_data = datasets.SHD("./data", transform=frame_transform, train=False)
    
    train_data_loader = DataLoader(train_data, shuffle=False, batch_size=BATCH_SIZE)
    test_data_loader = DataLoader(test_data, shuffle=False, batch_size=BATCH_SIZE)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

    loss_history = []
    acc_history = []
    best_model = None
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
            loss_val, acc, loss_per_time_step = train(data, targets, net, optimizer)

            batch_size = data.size(0)

            epoch_loss += loss_val * batch_size
            epoch_acc += acc * batch_size

            total_samples += batch_size

            loss_per_time_step_hist.append(loss_per_time_step)

        epoch_loss_per_time_step.append(compute_epoch_loss_per_time_step_data(loss_per_time_step_hist, epoch=epoch))
        test_accuracy = compute_test_set_accuracy(test_data_loader, net)
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
            

    end = datetime.now()
    time_diff = end - start

    if isinstance(save_plots, str):
        save_history_plot(loss_history, path=f'{save_plots}_simplified_loss')
        save_history_plot(acc_history, path=f'{save_plots}_simplified_accuracy')
        save_loss_per_time_step_plot(epoch_loss_per_time_step, path=f'{save_plots}_loss_per_time_steps')

    test_set_accuracy = compute_test_set_accuracy(test_data_loader, net)

    data = {
        'epochs': early_stopped_number_epoch if num_epochs == 'early_stopping' else num_epochs,
        'training_accuracy': acc_history[-1],
        'test_accuracy': test_set_accuracy,
        'time':  time_diff.total_seconds(),
        **additional_output_information
    }

    if isinstance(save_model, str):
        torch.save(best_model.state_dict(), f'{save_model}.pth')

    with open(output_file_path, "w") as file:
        json.dump(data, file, indent=4) 