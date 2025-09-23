import snntorch as snn
import torch
import torch.nn as nn


class RSNN(nn.Module):
    def __init__(self, input_size=700, hidden_size=20):
        super(RSNN, self).__init__()
        self.hidden_size = hidden_size

        self.input_layer = nn.Linear(input_size, hidden_size)

        self.recurrent_layer = nn.Linear(hidden_size, hidden_size)

        self.remove_autapse()

        self.lif_hidden = snn.Leaky(beta=0.9)

    def remove_autapse(self):
        with torch.no_grad():
            w = self.recurrent_layer.weight
            w.fill_diagonal_(0.0)

    def forward(self, x):
        batch_size, time_steps, _ = x.shape

        h_mem = torch.zeros(batch_size, self.hidden_size, device=x.device)
        h_spk = torch.zeros(batch_size, self.hidden_size, device=x.device)

        hidden_spikes = []
        hidden_membrane_potential = []

        self.lif_hidden.init_leaky()

        for t in range(time_steps):
            x_t = x[:, t, :]

            if t == 0:
                current = self.input_layer(x_t)
                h_spk, h_mem = self.lif_hidden(current, h_mem)

                hidden_spikes.append(h_spk)
                hidden_membrane_potential.append(h_mem)
            else:
                input_layer_current = self.input_layer(x_t)
                previous_hidden_layer_current = self.recurrent_layer(h_spk)

                h_spk, h_mem = self.lif_hidden(
                    input_layer_current + previous_hidden_layer_current, h_mem
                )

                hidden_spikes.append(h_spk)
                hidden_membrane_potential.append(h_mem)

        return [torch.stack(hidden_spikes, dim=0).permute(1, 0, 2)], [
            torch.stack(hidden_membrane_potential, dim=0).permute(1, 0, 2)
        ]
