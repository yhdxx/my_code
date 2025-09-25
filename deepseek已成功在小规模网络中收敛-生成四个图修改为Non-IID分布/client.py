# client.py
import copy
import torch
import torch.nn as nn
import torch.optim as optim

class Client:
    def __init__(self, cid, model: nn.Module, device='cpu', lr_front=0.01):
        self.cid = cid
        self.device = device
        self.model = model.to(device)
        # Separate optimizers: front (private) will be updated locally
        front_params = list(self.model.front.parameters())
        self.front_opt = optim.SGD(front_params, lr=lr_front)
        # We do NOT create an optimizer for tail because tail updates come from server.
        self.criterion = nn.CrossEntropyLoss()

    def set_tail_params(self, tail_state_dict):
        """Replace local tail parameters with server-side params (state_dict)."""
        self.model.tail.load_state_dict(tail_state_dict)

    def train_local(self, dataloader, epochs=1, send_every_batch=False):
        """Train locally for some epochs/batches. Return list of (param_name, grad_tensor) for tail grads."""
        self.model.train()
        tail_grads = None
        for epoch in range(epochs):
            for xb, yb in dataloader:
                xb = xb.to(self.device); yb = yb.to(self.device)
                # zero front grads
                self.front_opt.zero_grad()
                # zero tail grads (so .grad is fresh)
                for p in self.model.tail.parameters():
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()
                out = self.model(xb)
                loss = self.criterion(out, yb)
                loss.backward()
                # Update front parameters locally
                self.front_opt.step()
                # Extract tail gradients (clone them)
                tail_grads = {name: p.grad.detach().clone() for name, p in self.model.tail.named_parameters() if p.grad is not None}
                # Optionally send per batch; we'll just return the last observed tail_grads (or could accumulate)
                if send_every_batch:
                    return tail_grads
            # end batches
        return tail_grads

    def evaluate(self, dataloader):
        self.model.eval()
        correct = 0
        total = 0
        loss_total = 0.0
        with torch.no_grad():
            for xb, yb in dataloader:
                xb = xb.to(self.device); yb = yb.to(self.device)
                out = self.model(xb)
                loss = self.criterion(out, yb)
                loss_total += loss.item() * xb.size(0)
                preds = out.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += xb.size(0)
        return loss_total / total, correct / total
