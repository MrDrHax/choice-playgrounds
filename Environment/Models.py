import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class BasicCNNModel(nn.Module):
    def __init__(self, input_channels=3, num_actions=6, image_size=(64, 64)):
        super(BasicCNNModel, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

        # Dynamically compute flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, *image_size)
            x = self._forward_conv(dummy_input)
            self.flattened_size = x.view(1, -1).size(1)

        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.output = nn.Linear(128, num_actions)

    def _forward_conv(self, x):
        x = F.relu(self.conv1(x))  # Downsample
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.output(x))  # Independent actions [0, 1]

