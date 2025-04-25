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

time_steps = 100
num_inputs = 700
num_hidden = 1000
num_outputs = 20

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
                                       batch_size=1,
                                       nb_steps=time_steps,
                                       nb_units=num_inputs,
                                       max_time=max_time,
                                       device=device))
    
    print("data generated")
    
    net = Net().to(device)

    optimizer = torch.optim.Adam(net.parameters())
    loss_fn = SF.mse_count_loss()

    num_epochs = 30

    loss_hist = []
    acc_hist = []

    # training loop
    print("start of training loop")
    for epoch in range(num_epochs):
        for i, (data, targets) in enumerate(data_generator):
            data = data.to_dense().permute(1, 0, 2)
            data = data.to(device)
            targets = targets.to(device)


            net.train()
            spk_rec, _ = net.forward(data)
            loss_val = loss_fn(spk_rec, targets)

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Store loss history for future plotting
            loss_hist.append(loss_val.item())

            print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")

            acc = SF.accuracy_rate(spk_rec, targets)
            acc_hist.append(acc)
            print(f"Accuracy: {acc * 100:.2f}%\n")