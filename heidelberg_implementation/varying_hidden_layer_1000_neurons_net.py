import torch
import torch.nn as nn
import snntorch as snn

class VaryingHiddenLayer1000NeuronsNet(nn.Module):
    def __init__(self, num_input, num_hidden, num_output, beta, time_steps, num_hidden_layers=4):
        super().__init__()

        self.time_steps = time_steps

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
    
    def forward(self, x):
        mems = []

        for lif in self.lifs:
            mems.append(lif.init_leaky())

        final_layer_spk_recording = []
        final_layer_mem_recording = []

        for step in range(self.time_steps):
            for i, (linear, lif) in enumerate(zip(self.linears, self.lifs)):
                if i == 0:
                    current = linear(x[step])
                    spk, mem = lif(current, mems[0])
                else:
                    current = linear(spk)
                    spk, mem = lif(current, mems[i])

                mems[i] = mem

                if i == (len(self.lifs) - 1):
                    final_layer_spk_recording.append(spk)
                    final_layer_mem_recording.append(mem)

        return torch.stack(final_layer_spk_recording, dim=0), torch.stack(final_layer_mem_recording, dim=0)
