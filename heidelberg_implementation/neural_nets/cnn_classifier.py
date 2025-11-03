import torch
import torch.nn as nn


def conv_block():
    return nn.Sequential(
        nn.Conv2d(32, 32, kernel_size=3),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout(0.2),
    )


class CNN_Classifier(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(CNN_Classifier, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=input_channels, out_channels=32, kernel_size=11
        )
        self.relu1 = nn.ReLU()

        self.conv_block1 = conv_block()
        self.conv_block2 = conv_block()
        self.conv_block3 = conv_block()

        self.fc1 = nn.Linear(32 * 3 * 12, 128)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x = torch.flatten(x, 1)

        x = self.relu_fc1(self.fc1(x))
        x = self.fc2(x)
        return x
