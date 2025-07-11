import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, input_size=700, hidden_size=256, num_layers=1, num_classes=20):
        super(LSTMClassifier, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(torch.mean(lstm_out, dim=0))
        return out
