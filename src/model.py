import torch
import torch.nn as nn


class CNN(nn.Module):
    """
    3-Stage Convolutional Neural Network for CIFAR-10 Image Classification.

    Architecture:
        - Block 1: Conv2d(3 -> 32, 3x3, p=1) + ReLU + MaxPool2d(2, 2)  --> Output: (B, 32, 16, 16)
        - Block 2: Conv2d(32 -> 64, 3x3, p=1) + ReLU + MaxPool2d(2, 2) --> Output: (B, 64, 8, 8)
        - Block 3: Conv2d(64 -> 128, 3x3, p=1) + ReLU + MaxPool2d(2, 2) --> Output: (B, 128, 4, 4)
        - Flatten: 128 * 4 * 4 = 2,048 features
        - FC 1: Linear(2048 -> 256) + ReLU
        - FC 2: Linear(256 -> 10)
    """

    def __init__(self, num_classes: int = 10):
        super(CNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(4 * 4 * 128, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
