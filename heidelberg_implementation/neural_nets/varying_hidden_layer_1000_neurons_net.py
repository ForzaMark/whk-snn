import torch
import torch.nn as nn
import snntorch as snn

class VaryingHiddenLayer1000NeuronsNet(nn.Module):
    def __init__(self, num_input, num_hidden, num_output, beta, time_steps, num_hidden_layers, sparsity):
        super().__init__()

        self.time_steps = time_steps
        self.num_hidden_layers = num_hidden_layers

        self.lifs = nn.ModuleList()
        layers = []

        layers.append(nn.Linear(num_input, num_hidden))
        self.lifs.append(snn.Leaky(beta=beta))

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(num_hidden, num_hidden))
            self.lifs.append(snn.Leaky(beta=beta))

        layers.append(nn.Linear(num_hidden, num_output))
        self.lifs.append(snn.Leaky(beta=beta))

        self.linears = nn.ModuleList(layers)

        with torch.no_grad():
            for name, param in self.named_parameters():
                if "weight" in name:
                    mask = (torch.rand_like(param) > sparsity).float()
                    param.mul_(mask)

    
    def forward(self, x):
        mems = []

        for lif in self.lifs:
            mems.append(lif.init_leaky())

        all_spk_recordings = [[] for _ in range(self.num_hidden_layers + 1)]
        all_mem_recordings = [[] for _ in range(self.num_hidden_layers + 1)]

        for step in range(self.time_steps):
            for i, (linear, lif) in enumerate(zip(self.linears, self.lifs)):
                if i == 0:
                    current = linear(x[step])
                    spk, mem = lif(current, mems[0])
                else:
                    current = linear(spk)
                    spk, mem = lif(current, mems[i])

                mems[i] = mem

                all_spk_recordings[i].append(spk)
                all_mem_recordings[i].append(mem)

        return [torch.stack(spk_recording, dim=0) for spk_recording in  all_spk_recordings], \
               [torch.stack(mem_recording, dim=0) for mem_recording in all_mem_recordings]