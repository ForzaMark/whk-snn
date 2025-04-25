import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import spikeplot as splt
from snntorch import utils
import torch.nn as nn
from train_utils import sparse_data_generator_from_hdf5_spikes
from train_utils import get_train_test_data
import torch
from utils import get_device
import numpy as np
import matplotlib.pyplot as plt

def save_loss_plot_png(loss_history):
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Index (Timestep)")
    plt.ylabel("Value")
    plt.title("Array Plot")
    plt.grid(True)

    plt.savefig("./simplified_loss_hist.png")
    plt.close()

time_steps = 100
num_inputs = 700
num_hidden = 1000
num_outputs = 20
dtype = torch.float
batch_size = 32

max_time = 1.4
beta = 0.99
device = get_device()

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(time_steps):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = get_train_test_data()    

    data_generator = list(sparse_data_generator_from_hdf5_spikes(x_train, y_train,
                                       batch_size=batch_size,
                                       nb_steps=time_steps,
                                       nb_units=num_inputs,
                                       max_time=max_time,
                                       device=device))
    
    print("data generated")
    
    net = Net().to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))
    loss = nn.CrossEntropyLoss()

    num_epochs = 3

    global_loss_hist = []

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")

        for i, (data, targets) in enumerate(data_generator):
            loss_hist = []
            acc_hist = []

            data = data.to_dense().permute(1, 0, 2).to(device)
            targets = targets.to(device)

            spk_rec, mem_rec = net.forward(data)
            loss_val = torch.zeros((1), dtype=dtype, device=device)

            # sum loss at every step
            for step in range(time_steps):
                loss_val += loss(mem_rec[step], targets)

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            acc = SF.accuracy_rate(spk_rec, targets)

            global_loss_hist.append(loss_val.item())
            loss_hist.append(loss_val.item())
            acc_hist.append(acc)
            
        print(f"average loss {np.array(loss_hist).mean()}")
        print(f"average accuracy {np.array(acc_hist).mean()}")

    save_loss_plot_png(global_loss_hist)