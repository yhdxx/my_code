# Repository: partial_gradient_federated_split
# Files included in this one document (use as separate files if you prefer):
# - model.py
# - client.py
# - server.py
# - run.py
# - README.md

# -------- model.py --------
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN_Full(nn.Module):
    """A simple CNN where we can cut between early (client) and late (server) layers.
    The "cut" index indicates how many layers belong to the *server* (i.e. last n layers).
    """
    def __init__(self, cut_n=2, num_classes=10):
        super().__init__()
        # Define a sequence of blocks; we'll treat them as an ordered list
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)),  # 0
            nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)), # 1
            nn.Flatten(),                                                                 # 2
            nn.Linear(32*7*7, 128),                                                       # 3
            nn.ReLU(),                                                                    # 4
            nn.Linear(128, num_classes)                                                   # 5
        ])
        self.cut_n = cut_n

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

    def get_server_blocks(self):
        # return last cut_n blocks as an nn.Sequential
        if self.cut_n == 0:
            return nn.Sequential()
        server_blocks = list(self.blocks[-self.cut_n:])
        return nn.Sequential(*server_blocks)

    def get_client_blocks(self):
        client_blocks = list(self.blocks[:-self.cut_n]) if self.cut_n>0 else list(self.blocks[:])
        return nn.Sequential(*client_blocks)

    def load_server_state_dict(self, sd):
        server = self.get_server_blocks()
        server.load_state_dict(sd)

    def server_state_dict(self):
        return self.get_server_blocks().state_dict()

    def client_state_dict(self):
        return self.get_client_blocks().state_dict()
