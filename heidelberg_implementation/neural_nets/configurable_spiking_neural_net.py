from typing import Union

import snntorch as snn
import torch
import torch.nn as nn
from constants import ATAN_ALPHA
from snntorch import surrogate


class ConfigurableSpikingNeuralNet(nn.Module):
    def __init__(self, 
                 number_input_neurons, 
                 number_hidden_neurons, 
                 number_output_neurons, 
                 beta, 
                 threshold,
                 time_steps, 
                 number_hidden_layers,
                 surrogate_approximation = surrogate.atan(alpha = ATAN_ALPHA),
                 population_coding: Union[int, bool] = False):
        super().__init__()

        self.time_steps = time_steps
        self.num_hidden_layers = number_hidden_layers

        self.lifs = nn.ModuleList()
        layers = []

        layers.append(nn.Linear(number_input_neurons, number_hidden_neurons))
        self.lifs.append(snn.Leaky(beta=beta, threshold=threshold, spike_grad=surrogate_approximation))

        for _ in range(number_hidden_layers - 1):
            layers.append(nn.Linear(number_hidden_neurons, number_hidden_neurons))
            self.lifs.append(snn.Leaky(beta=beta, threshold=threshold, spike_grad=surrogate_approximation))

        if population_coding is False:
            layers.append(nn.Linear(number_hidden_neurons, number_output_neurons))
        else:
            assert isinstance(population_coding, int) and population_coding > number_output_neurons
            layers.append(nn.Linear(number_hidden_neurons, population_coding))
        
        self.lifs.append(snn.Leaky(beta=beta, threshold=threshold, spike_grad=surrogate_approximation))

        self.linears = nn.ModuleList(layers)
    
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