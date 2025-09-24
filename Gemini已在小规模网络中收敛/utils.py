import torch
import torch.nn as nn
from torchvision import datasets, transforms
import os


# Define the server model (the last n layers)
class ServerModel(nn.Module):
    def __init__(self, n):
        super(ServerModel, self).__init__()
        # We will define this part based on the cut_n later
        # For a simple example, let's assume n=3 and it's a simple classifier
        if n == 3:
            self.fc1 = nn.Linear(512, 128)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(128, 64)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(64, 10)
        else:
            raise ValueError("n must be 3 for this example.")

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


# Define the client model (the full mi layers)
class ClientModel(nn.Module):
    def __init__(self, mi, cut_n):
        super(ClientModel, self).__init__()
        # For this example, let's assume a simple CNN for mi=5 layers and cut_n=2
        # So, first 2 layers are client-only, last 3 are shared
        if mi == 5 and cut_n == 2:
            # Client-only part (mi-n layers)
            self.conv1 = nn.Conv2d(1, 16, 5, 1)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(2)
            self.conv2 = nn.Conv2d(16, 32, 5, 1)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(2)
            self.flatten = nn.Flatten()
            self.fc_client_only = nn.Linear(512, 512)  # Output of client-only part

            # Shared part (the last n layers)
            self.fc1 = nn.Linear(512, 128)
            self.relu3 = nn.ReLU()
            self.fc2 = nn.Linear(128, 64)
            self.relu4 = nn.ReLU()
            self.fc3 = nn.Linear(64, 10)
        else:
            raise ValueError("mi must be 5 and cut_n must be 2 for this example.")

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc_client_only(x)

        # We need to split the forward pass for this method
        # The forward pass will be implemented in the training loop
        # For simplicity, we can have a full forward pass here for testing
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x


def get_data_loaders(num_clients, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Split training data into num_clients parts
    client_datasets = torch.utils.data.random_split(
        train_dataset,
        [len(train_dataset) // num_clients] * num_clients
    )

    train_loaders = [
        torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
        for ds in client_datasets
    ]

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loaders, test_loader


def get_model(mi, cut_n):
    # This is to get the full model for client side
    return ClientModel(mi, cut_n)


def get_server_model(n):
    # This is to get the shared part for server side
    return ServerModel(n)