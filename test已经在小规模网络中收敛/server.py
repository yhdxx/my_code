# -------- server.py --------
import copy
import torch
import torch.optim as optim

class Server:
    def __init__(self, server_model_cls, cut_n, device='cpu'):
        self.device = device
        self.cut_n = cut_n
        self.model = server_model_cls(cut_n=cut_n).to(device)  # same architecture; only last-n will be used
        self.opt = optim.SGD(self.model.get_server_blocks().parameters(), lr=0.05)

    def aggregate_and_update(self, client_grads_list):
        """client_grads_list: list of dicts name->grad tensors (cpu)
        We'll average gradients and apply to server's last-n params via a simple SGD step.
        """
        # get server param names order
        server_state = self.model.get_server_blocks().state_dict()
        # build averaged grads
        avg_grads = {}
        n = len(client_grads_list)
        for gdict in client_grads_list:
            for k,v in gdict.items():
                if k not in avg_grads:
                    avg_grads[k] = v.clone()
                else:
                    avg_grads[k] += v
        for k in avg_grads.keys():
            avg_grads[k] /= float(n)

        # Apply gradient step manually
        # Map grads into server parameters
        # Convert server parameters to list of (name, tensor)
        with torch.no_grad():
            idx = 0
            for name, param in self.model.get_server_blocks().named_parameters():
                key_full = name.replace('0.','') if False else name  # name mapping varies; we use direct mapping below
                # Find matching key in avg_grads (they are named like 'blocks.3.weight' in client)
                # Simpler: iterate over server's named_parameters and pick items from avg_grads using same names
                # We'll assume names align between client and server
                if name in avg_grads:
                    g = avg_grads[name].to(self.device)
                    # SGD: param = param - lr * grad
                    lr = self.opt.param_groups[0]['lr']
                    param -= lr * g
        # return updated server state dict
        return {k:v.cpu() for k,v in self.model.get_server_blocks().state_dict().items()}

