import torch.nn as nn
import snntorch as snn
import torch

num_hidden = 1000

class SingleHiddenLayer1000NeuronsNet(nn.Module):
    def __init__(self, num_inputs, num_outputs, beta, time_steps, sparsity):
        super().__init__()

        self.time_steps = time_steps

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

        with torch.no_grad():
            for name, param in self.named_parameters():
                if "weight" in name:
                    mask = (torch.rand_like(param) > sparsity).float()
                    param.mul_(mask)

    def forward(self, x):
        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk1_rec = []
        mem1_rec = []

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(self.time_steps):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk1_rec.append(spk1)
            mem1_rec.append(mem1)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return [torch.stack(spk1_rec, dim=0), torch.stack(spk2_rec, dim=0)], \
               [torch.stack(mem1_rec, dim=0), torch.stack(mem2_rec, dim=0)]