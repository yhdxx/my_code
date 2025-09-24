# server.py
import torch
import torch.nn as nn
import torch.optim as optim

class Server:
    def __init__(self, server_model: nn.Module, lr=5e-4, device='cpu', weight_decay=1e-4):
        self.device = device
        self.model = server_model.to(device)
        # Use Adam for server updates (often more robust on aggregated gradients)
        self.opt = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def aggregate_and_update(self, list_of_tail_grads, grad_clip=10.0):
        """
        list_of_tail_grads: list of dicts {param_name: grad_tensor(cpu)} from clients.
        We'll average them (element-wise) and apply as .grad on server model, then optimizer.step()
        """
        if not list_of_tail_grads:
            return

        # initialize accumulator
        avg = {}
        cnt = len(list_of_tail_grads)
        first = list_of_tail_grads[0]
        for param_name in first.keys():
            avg[param_name] = torch.zeros_like(first[param_name], device=self.device)

        # sum
        for gdict in list_of_tail_grads:
            for k, v in gdict.items():
                avg[k] += v.to(self.device)

        # average
        for k in avg:
            avg[k] = avg[k] / cnt

        # set server model .grad
        self.opt.zero_grad()
        name_to_param = dict(self.model.named_parameters())
        for k, avg_grad in avg.items():
            if k in name_to_param:
                # optionally clip gradient per-parameter norm to improve stability
                g = avg_grad.clone()
                # set as .grad
                name_to_param[k].grad = g.clone()
            else:
                raise KeyError(f"Server model missing param {k}")

        # gradient clipping on server parameters (global)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)

        # optimizer step
        self.opt.step()

    def get_tail_state(self):
        """Return CPU state_dict of server tail for sending to clients."""
        return {k: v.cpu() for k, v in self.model.state_dict().items()}
