from snntorch import functional as SF
import torch.nn as nn
from util.utils import save_history_plot
import torch
from util.utils import get_device
import numpy as np
from torch.utils.data import DataLoader
from tonic import datasets, transforms
import json
from datetime import datetime

def compute_test_set_accuracy(test_data_generator, net):
    total = 0
    correct = 0

    with torch.no_grad():
        net.eval()
        for data, targets in test_data_generator:
            data = data.to_dense().to(torch.float32).squeeze().permute(1, 0, 2).to(device)
            targets = targets.to(device)

            test_spk_recs, _ = net(data)

            output_test_spk = test_spk_recs[-1]

            _, predicted = output_test_spk.sum(dim=0).max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()


    return 100 * correct / total

def calculate_loss_over_all_timesteps(time_steps, mem_rec, targets, loss_function):
    loss_val = torch.zeros((1), dtype=dtype, device=device)

    for step in range(time_steps):
        loss_val += loss_function(mem_rec[step], targets)

    return loss_val

def calculate_gradient(optimizer, loss_val):
    optimizer.zero_grad()
    loss_val.backward()

def update_weights(optimizer):
    optimizer.step()

time_steps = 100
num_inputs = 700
num_hidden = 1000
num_outputs = 20
dtype = torch.float
batch_size = 32

max_time = 1.4
beta = 0.99
device = get_device()

frame_transform = transforms.ToFrame(
    sensor_size=datasets.SHD.sensor_size,  
    n_time_bins=time_steps
)

LOSS_FUNCTION = nn.CrossEntropyLoss()

def train_simplified_snn(net, num_epochs, save_model=False, save_plots=False, additional_output_information={}, output_file_path='output/simplified_results.json'):
    train_data = datasets.SHD("./data", transform=frame_transform, train=True)
    test_data = datasets.SHD("./data", transform=frame_transform, train=False)
    
    train_data_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

    global_loss_hist = []
    global_acc_hist = []

    start = datetime.now()

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")

        for i, (data, targets) in enumerate(train_data_loader):
            loss_hist = []
            acc_hist = []

            data = data.to_dense().to(torch.float32).squeeze().permute(1, 0, 2).to(device)
            targets = targets.to(device)

            spk_recs, mem_recs = net(data)

            output_spk_rec = spk_recs[-1]
            output_mem_rec = mem_recs[-1]

            loss_val = calculate_loss_over_all_timesteps(time_steps, output_mem_rec, targets, LOSS_FUNCTION)

            calculate_gradient(optimizer=optimizer, loss_val=loss_val)
            update_weights(optimizer=optimizer)
            
            acc = SF.accuracy_rate(output_spk_rec, targets)
            acc_hist.append(acc)
            
        print(f"loss {loss_val.item()}")
        print(f"accuracy {np.array(acc_hist).mean()}")
        
        global_loss_hist.append(loss_val.item())
        global_acc_hist.append(np.array(acc_hist).mean())

    end = datetime.now()
    time_diff = end - start

    if isinstance(save_plots, str):
        save_history_plot(global_loss_hist, path=f'{save_plots}_simplified_loss')
        save_history_plot(global_acc_hist, path=f'{save_plots}_simplified_accuracy')

    test_set_accuracy = compute_test_set_accuracy(test_data_loader, net)

    data = {
        'epochs': num_epochs,
        'training_accuracy': global_acc_hist[len(global_acc_hist) - 1],
        'test_accuracy': test_set_accuracy,
        'time':  time_diff.total_seconds(),
        **additional_output_information
    }

    if isinstance(save_model, str):
        torch.save(net.state_dict(), f'{save_model}.pth')

    with open(output_file_path, "w") as file:
        json.dump(data, file, indent=4) 