# client.py
import torch
import torch.nn as nn
import torch.optim as optim
import copy

class Client:
    def __init__(self, cid, model: nn.Module, device='cpu', lr_front=1e-3, weight_decay=1e-4):
        """
        model: ClientModel instance (front + tail copy)
        """
        self.cid = cid
        self.device = device
        self.model = model.to(device)
        # front (private) optimizer - use Adam for better stability on larger models
        front_params = list(self.model.front.parameters())
        self.front_opt = optim.Adam(front_params, lr=lr_front, weight_decay=weight_decay)
        # criterion
        self.criterion = nn.CrossEntropyLoss()

    def set_tail_params(self, tail_state_dict):
        """Load server-provided tail parameters into local tail."""
        self.model.tail.load_state_dict(tail_state_dict)

    def train_local(self, dataloader, epochs=1, send_every_batch=False, grad_clip=5.0):
        """
        Train local front for `epochs` over dataloader.
        Collect tail gradients for every batch, average them over batches, and return the averaged grad dict.
        Returns: dict {param_name: averaged_grad_tensor} or None if no grads observed.
        """
        self.model.train()
        device = self.device
        tail_accum = None
        batch_count = 0

        for epoch in range(epochs):
            for xb, yb in dataloader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                # zero front grads and zero existing tail grads
                self.front_opt.zero_grad()
                for p in self.model.tail.parameters():
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()

                out = self.model(xb)
                loss = self.criterion(out, yb)
                loss.backward()

                # gradient clipping on front params for stability
                torch.nn.utils.clip_grad_norm_(self.model.front.parameters(), max_norm=grad_clip)

                # update front parameters
                self.front_opt.step()

                # gather tail gradients for this batch (if exist)
                tail_grads_batch = {name: p.grad.detach().clone().cpu() for name, p in self.model.tail.named_parameters() if p.grad is not None}

                if tail_grads_batch:
                    batch_count += 1
                    if tail_accum is None:
                        tail_accum = {k: v.clone() for k, v in tail_grads_batch.items()}
                    else:
                        for k, v in tail_grads_batch.items():
                            tail_accum[k] += v

                    if send_every_batch:
                        # return averaged up to this batch
                        averaged = {k: (v / batch_count).clone() for k, v in tail_accum.items()}
                        return averaged

        if tail_accum is None:
            return None
        # average across batches
        averaged = {k: (v / batch_count).clone() for k, v in tail_accum.items()}
        return averaged

    def evaluate(self, dataloader):
        """
        Evaluate full local model (front + tail) on given dataloader.
        Returns: (avg_loss, accuracy)
        """
        self.model.eval()
        correct = 0
        total = 0
        loss_total = 0.0
        with torch.no_grad():
            for xb, yb in dataloader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                out = self.model(xb)
                loss = self.criterion(out, yb)
                loss_total += loss.item() * xb.size(0)
                preds = out.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += xb.size(0)

        if total == 0:
            return None, None
        return loss_total / total, correct / total
