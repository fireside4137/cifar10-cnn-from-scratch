import torch
import torch.nn as nn


class CNN(nn.Module):
    """
    Enhanced 3-Stage Convolutional Neural Network for CIFAR-10.

    Improvements:
        - Batch Normalization (nn.BatchNorm2d) after each Conv2d layer
          to stabilize activations and accelerate training convergence.
        - Spatial Dropout (nn.Dropout2d(0.25)) after pooling to prevent
          co-adaptation across feature maps.
        - Dense Dropout (nn.Dropout(0.5)) in fully-connected head to
          regularize high parameter count and eliminate overfitting.

    Architecture:
        - Block 1: Conv2d(3 -> 32, 3x3, p=1) + BatchNorm2d(32) + ReLU + MaxPool(2,2) + Dropout2d(0.25)
        - Block 2: Conv2d(32 -> 64, 3x3, p=1) + BatchNorm2d(64) + ReLU + MaxPool(2,2) + Dropout2d(0.25)
        - Block 3: Conv2d(64 -> 128, 3x3, p=1) + BatchNorm2d(128) + ReLU + MaxPool(2,2) + Dropout2d(0.25)
        - Flatten: 128 * 4 * 4 = 2,048 features
        - FC 1: Linear(2048 -> 256) + ReLU + Dropout(0.5)
        - FC 2: Linear(256 -> 10)
    """

    def __init__(self, num_classes: int = 10):
        super(CNN, self).__init__()

        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(4 * 4 * 128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
