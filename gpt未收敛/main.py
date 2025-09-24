# run.py
# 单机模拟：clients 保持 front (异构) + local server 副本（与 central server server_back 同结构）
# 每轮：客户端本地完整训练 local_epochs，然后上报 server 参数（weights）到 central，
# central 做简单平均（FedAvg）并下发 global server 参数回客户端。
# 记录 loss/acc 到 CSV。

import copy, random, os, csv
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# ---------- 简单模块 ----------
class ClientFront(nn.Module):
    def __init__(self, input_dim=28*28, hidden=128, depth=1):
        super().__init__()
        layers = []
        cur = input_dim
        for i in range(depth):
            layers.append(nn.Linear(cur, hidden))
            layers.append(nn.ReLU(inplace=True))
            cur = hidden
        self.net = nn.Sequential(*layers)
        self.out_dim = cur

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

class ServerBack(nn.Module):
    def __init__(self, in_dim, hidden=64, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class ClientModel(nn.Module):
    def __init__(self, front: ClientFront, server_back: ServerBack):
        super().__init__()
        self.front = front
        # local copy of server part (will be synced from central server periodically)
        self.server = copy.deepcopy(server_back)

    def forward(self, x):
        z = self.front(x)
        out = self.server(z)
        return out

# ---------- 数据划分 ----------
def split_dataset(dataset, d):
    n = len(dataset)
    idx = np.arange(n)
    np.random.shuffle(idx)
    sizes = [n // d] * d
    for i in range(n % d):
        sizes[i] += 1
    parts = []
    cur = 0
    for s in sizes:
        parts.append(idx[cur:cur+s].tolist())
        cur += s
    return parts

# ---------- 训练主流程 ----------
def run(
    num_clients=4,
    rounds=30,
    local_epochs=1,
    local_batch_size=128,
    lr_front=1e-2,
    lr_server_local=1e-2,
    lr_server_global=1e-2,
    device=DEVICE
):
    transform = transforms.Compose([transforms.ToTensor()])
    train_all = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_all = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    parts = split_dataset(train_all, num_clients)
    front_out_dim = 128
    server_back = ServerBack(in_dim=front_out_dim, hidden=64, num_classes=10).to(device)

    # create clients with heterogeneous fronts but same front_out_dim
    clients = []
    client_loaders = []
    for i in range(num_clients):
        depth = random.choice([1, 2])  # heterogeneity
        front = ClientFront(input_dim=28*28, hidden=front_out_dim, depth=depth)
        model = ClientModel(front, server_back)
        model.to(device)
        clients.append(model)
        subset = Subset(train_all, parts[i])
        loader = DataLoader(subset, batch_size=local_batch_size, shuffle=True, drop_last=False)
        client_loaders.append(loader)

    # optimizers: each client has optimizer for full local model (front + server_local)
    client_opts = []
    for m in clients:
        opt = optim.SGD(list(m.front.parameters()) + list(m.server.parameters()), lr=lr_front, momentum=0.9)
        client_opts.append(opt)

    loss_fn = nn.CrossEntropyLoss()

    # For central server, we'll treat server_back as the global server parameters (and give it its own optimizer optionally)
    server_global = server_back

    # logging
    os.makedirs('logs', exist_ok=True)
    csv_path = os.path.join('logs', 'train_log.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['round', 'train_loss_mean', 'test_loss', 'test_acc'])

    # evaluation function: ensemble over clients' front + global server (average logits)
    def evaluate():
        server_global.eval()
        for m in clients: m.eval()
        loader = DataLoader(test_all, batch_size=256, shuffle=False)
        total = 0
        correct = 0
        loss_sum = 0.0
        with torch.no_grad():
            for x,y in loader:
                x,y = x.to(device), y.to(device)
                logits_sum = 0
                for m in clients:
                    z = m.front(x)           # each client's front
                    logits = server_global(z)  # use global server for evaluation
                    logits_sum += logits
                logits_avg = logits_sum / len(clients)
                loss_sum += loss_fn(logits_avg, y).item() * y.size(0)
                preds = logits_avg.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return loss_sum / total, correct / total

    # Training rounds
    for r in range(1, rounds+1):
        # Each client does local_epochs of local training on its own data (updating both front and local server)
        train_losses = []
        for i, (m, loader, opt) in enumerate(zip(clients, client_loaders, client_opts)):
            m.train()
            local_losses = []
            for ep in range(local_epochs):
                for x,y in loader:
                    x,y = x.to(device), y.to(device)
                    opt.zero_grad()
                    logits = m(x)
                    loss = loss_fn(logits, y)
                    loss.backward()
                    opt.step()
                    local_losses.append(loss.item())
                    # limit verbosity / runtime: optional early stop per client (not necessary)
                    # break
            if len(local_losses) > 0:
                train_losses.append(float(np.mean(local_losses)))
            else:
                train_losses.append(0.0)

        # ------------- server aggregation: average server parameters from clients -------------
        # collect server param state dicts
        server_states = [m.server.state_dict() for m in clients]
        # simple federated averaging of parameters (equal weighting)
        new_state = {}
        for key in server_states[0].keys():
            # stack and mean
            stacked = torch.stack([s[key].float().cpu() for s in server_states], dim=0)
            new_state[key] = torch.mean(stacked, dim=0)
        # load averaged params into global server and into each client's server copy
        server_global.load_state_dict({k: new_state[k].to(device) for k in new_state})
        for m in clients:
            m.server.load_state_dict(server_global.state_dict())

        # evaluate
        if r % 1 == 0:
            test_loss, test_acc = evaluate()
            mean_train_loss = float(np.mean(train_losses)) if len(train_losses)>0 else 0.0
            print(f"Round {r}/{rounds} | train_loss {mean_train_loss:.4f} | test_loss {test_loss:.4f} | test_acc {test_acc:.4f}")
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([r, mean_train_loss, test_loss, test_acc])

    print("Training finished. Logs saved to", csv_path)

if __name__ == '__main__':
    run(num_clients=4, rounds=30, local_epochs=1, local_batch_size=128, device=DEVICE)
