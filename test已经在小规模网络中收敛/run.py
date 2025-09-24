# -------- run.py (entry point) --------
import argparse
import torch
import torch.nn as nn
import csv
from torchvision import datasets, transforms
from torch.utils.data import random_split

from model import SimpleCNN_Full
from client import Client
from server import Server


def partition_dataset(dataset, num_parts):
    length = len(dataset)
    part = length // num_parts
    lengths = [part]*num_parts
    # give remainder to last
    lengths[-1] += length - sum(lengths)
    parts = random_split(dataset, lengths)
    return parts


def main(num_clients=3, rounds=5, cut_n=2, device='cpu'):
    transform = transforms.Compose([transforms.ToTensor()])
    mnist = datasets.MNIST('../data', train=True, download=True, transform=transform)
    parts = partition_dataset(mnist, num_clients)

    # Create clients and server
    clients = [Client(i, SimpleCNN_Full, cut_n, parts[i], device=device) for i in range(num_clients)]
    server = Server(SimpleCNN_Full, cut_n, device=device)

    criterion = nn.CrossEntropyLoss()

    # 打开CSV文件准备写入
    with open("loss_log.csv", mode="w", newline="") as f:
        writer = csv.writer(f)
        # 写表头
        header = ["Round"] + [f"Client{i}_loss" for i in range(num_clients)]
        writer.writerow(header)

        for r in range(rounds):
            print(f"\n===== Round {r} =====")
            client_grads = []
            losses = []
            for c in clients:
                grads, avg_loss = c.compute_last_n_gradients(criterion)
                client_grads.append(grads)
                losses.append(avg_loss)

            # 打印
            print(" client average losses (full local dataset):")
            for i, loss in enumerate(losses):
                print(f"   Client {i}: {loss:.4f}")

            # 写入CSV
            row = [r] + losses
            writer.writerow(row)

            # Server aggregates and updates last-n
            updated_server_state = server.aggregate_and_update(client_grads)

            # Server sends updated last-n to clients
            for c in clients:
                c.update_server_params(updated_server_state)

            # Clients perform local update of front layers while keeping last-n fixed
            for c in clients:
                c.local_update_front(epochs=1, lr=0.01)

    print("Done training simulation.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clients', type=int, default=3)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--cut_n', type=int, default=2)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    main(num_clients=args.num_clients, rounds=args.rounds, cut_n=args.cut_n, device=args.device)


