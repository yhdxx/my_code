# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ServerTail(nn.Module):
    """
    Server-side shared tail (bigger MLP for CIFAR-10).
    Input: tail_in_features (e.g. 512)
    """
    def __init__(self, in_features=512, hidden=512, out_features=10):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden, hidden)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

class ClientModel(nn.Module):
    """
    Client model: a small convolutional front (private) that outputs a vector of size tail_in_features,
    plus a tail which uses the same structure as ServerTail (a local copy for inference / gradient calc).
    The front is intentionally larger than MNIST MLP to handle CIFAR-10.
    """
    def __init__(self, tail_in_features=512):
        super().__init__()
        # Convolutional front: outputs a flattened feature vector
        self.front = nn.Sequential(
            # conv block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 -> 16

            # conv block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16 -> 8

            # conv block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # 8x8 -> 1x1
            nn.Flatten(),
            # final projection to tail_in_features
            nn.Linear(256, tail_in_features),
            nn.ReLU(),
        )

        # tail: same structure as ServerTail (local copy)
        self.tail = ServerTail(in_features=tail_in_features, hidden=512, out_features=10)

    def forward(self, x):
        x = self.front(x)
        out = self.tail(x)
        return out
