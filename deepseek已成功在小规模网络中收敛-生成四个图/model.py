# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ServerTail(nn.Module):
    """Shared last-n-layers model (server side). We'll keep it as an MLP tail for MNIST."""
    def __init__(self, in_features=128, hidden=64, out_features=10):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class ClientModel(nn.Module):
    """Client full model = private front (varied) + shared tail (same class as ServerTail)."""
    def __init__(self, front_sizes=[784, 256], tail_in_features=128):
        """
        front_sizes: list describing an MLP front: [input_dim, hidden1, hidden2, ..., tail_in_features]
        tail_in_features must match ServerTail.in_features
        """
        super().__init__()
        layers = []
        for i in range(len(front_sizes)-1):
            layers.append(nn.Linear(front_sizes[i], front_sizes[i+1]))
            layers.append(nn.ReLU())
        # The front will output activations of size front_sizes[-1] which should equal tail_in_features
        self.front = nn.Sequential(*layers)
        # The tail is provided separately (server-side); here we'll create a copy to hold parameters locally.
        # But its structure must match ServerTail.
        self.tail = ServerTail(in_features=front_sizes[-1])

    def forward(self, x):
        x = x.view(x.size(0), -1)
        a = self.front(x)
        out = self.tail(a)
        return out
