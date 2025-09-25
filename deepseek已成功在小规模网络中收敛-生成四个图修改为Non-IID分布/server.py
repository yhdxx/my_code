# server.py
import torch
import torch.nn as nn
import torch.optim as optim
import copy

class Server:
    def __init__(self, server_model: nn.Module, lr=0.01, device='cpu'):
        self.device = device
        self.model = server_model.to(device)
        self.opt = optim.SGD(self.model.parameters(), lr=lr)

    def aggregate_and_update(self, list_of_tail_grads):
        """
        list_of_tail_grads: list of dicts {param_name: grad_tensor} from clients.
        We'll average them (element-wise) and apply as .grad on server model, then optimizer.step()
        """
        if len(list_of_tail_grads) == 0:
            return
        # init accumulator
        avg = {}
        cnt = len(list_of_tail_grads)
        for param_name in list_of_tail_grads[0].keys():
            avg[param_name] = torch.zeros_like(list_of_tail_grads[0][param_name], device=self.device)
        # sum
        for gdict in list_of_tail_grads:
            for k, v in gdict.items():
                avg[k] += v.to(self.device)
        # average
        for k in avg:
            avg[k] = avg[k] / cnt
        # set server model .grad
        # zero grads first
        self.opt.zero_grad()
        name_to_param = dict(self.model.named_parameters())
        for k, avg_grad in avg.items():
            if k in name_to_param:
                name_to_param[k].grad = avg_grad.clone()
            else:
                raise KeyError(f"Server model missing param {k}")
        # optimizer step
        self.opt.step()

    def get_tail_state(self):
        return {k: v.cpu() for k, v in self.model.state_dict().items()}
