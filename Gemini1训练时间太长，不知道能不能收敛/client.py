import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import copy


class Client:
    def __init__(self, client_id, model, dataset, batch_size=32):
        self.client_id = client_id
        self.model = model
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def train_and_get_partial_gradient(self, server_model):
        # Update client's back layers with server's parameters
        client_back_layers = self.model.back_layers
        server_layers = server_model.layers

        # Copy parameters from server model to client model's back layers
        for client_layer, server_layer in zip(client_back_layers, server_layers):
            client_layer.load_state_dict(server_layer.state_dict())

        # Train the entire model
        self.model.train()
        total_loss = 0.0
        for data, target in self.dataloader:
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        # Get the gradients of the back layers
        back_grads = {}
        for name, param in self.model.named_parameters():
            if 'back_layers' in name:
                if param.grad is not None:
                    back_grads[name] = copy.deepcopy(param.grad)

        return back_grads, total_loss / len(self.dataloader)