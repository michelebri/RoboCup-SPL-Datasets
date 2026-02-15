"""CNN model for robot color classification."""

from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F


class ColorClassifierCNN(nn.Module):
    """
    Convolutional Neural Network for robot color classification.

    Architecture:
    - 4 convolutional layers with batch normalization
    - Max pooling after each conv layer
    - 2 fully connected layers with dropout
    - Designed for 128x128 RGB input images
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 7):
        """
        Initialize the CNN model.

        Args:
            in_channels: Number of input channels (3 for RGB).
            num_classes: Number of output classes (robot colors).
        """
        super(ColorClassifierCNN, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        # Fully connected layers
        # After 4 pooling layers on 128x128 input: 128 -> 64 -> 32 -> 16 -> 8
        self.fc1 = nn.Linear(256 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Output logits of shape (batch_size, num_classes).
        """
        # Conv block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        # Conv block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        # Conv block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        # Conv block 4
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def get_feature_dims(self) -> Tuple[int, ...]:
        """
        Get the dimensions of the feature map before FC layers.

        Returns:
            Tuple of (channels, height, width) after all conv layers.
        """
        return (256, 8, 8)

    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f"ColorClassifierCNN(\n"
            f"  in_channels={self.in_channels},\n"
            f"  num_classes={self.num_classes},\n"
            f"  parameters={sum(p.numel() for p in self.parameters()):,}\n"
            f")"
        )


# Alias for backwards compatibility
CNN = ColorClassifierCNN
