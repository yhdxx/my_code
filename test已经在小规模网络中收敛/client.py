# -------- client.py --------
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

class Client:
    def __init__(self, cid, full_model_cls, cut_n, dataset, device='cpu'):
        self.cid = cid
        self.device = device
        self.model = full_model_cls(cut_n=cut_n).to(device)
        self.cut_n = cut_n
        self.dataset = dataset
        self.loader = DataLoader(dataset, batch_size=32, shuffle=True)

    def compute_last_n_gradients(self, criterion, optimizer=None):
        """Run one pass over the full local dataset and return averaged gradients
           for the last-n (server) parameters only."""
        self.model.train()

        optimizer_local = optimizer
        if optimizer_local is None:
            optimizer_local = optim.SGD(self.model.parameters(), lr=0.01)

        # 先清零梯度
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        total_loss = 0.0
        total_samples = 0

        # 遍历所有 batch
        for x, y in self.loader:
            x, y = x.to(self.device), y.to(self.device)
            optimizer_local.zero_grad()
            logits = self.model(x)
            loss = criterion(logits, y)
            loss.backward()

            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

        # 收集 server 参数
        server_blocks = list(self.model.blocks)[-self.cut_n:] if self.cut_n > 0 else []
        server_params = set([id(p) for blk in server_blocks for p in blk.parameters()])

        # 累积梯度并求平均
        grads = {}
        for name, p in self.model.named_parameters():
            if id(p) in server_params:
                grads[name] = (p.grad.detach().cpu().clone() / len(self.loader))

        avg_loss = total_loss / total_samples
        return grads, avg_loss

    def update_server_params(self, server_state_dict):
        # Load server blocks' weights
        # Make sure to update corresponding parameters in self.model
        server_blocks = self.model.get_server_blocks()
        server_blocks.load_state_dict(server_state_dict)

    def local_update_front(self, epochs=1, lr=0.01):
        """Perform local training updating only the client (front) layers while keeping server layers fixed."""
        # freeze server params
        for p in self.model.get_server_blocks().parameters():
            p.requires_grad = False
        for p in self.model.get_client_blocks().parameters():
            p.requires_grad = True

        opt = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
        crit = nn.CrossEntropyLoss()
        self.model.train()

        for e in range(epochs):
            total_loss, total_samples = 0.0, 0
            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                logits = self.model(x)
                loss = crit(logits, y)
                loss.backward()
                opt.step()
                total_loss += loss.item() * x.size(0)
                total_samples += x.size(0)
            print(f"  [Client {self.cid}] epoch {e} local front loss: {total_loss / total_samples:.4f}")

        # unfreeze everything
        for p in self.model.parameters():
            p.requires_grad = True

