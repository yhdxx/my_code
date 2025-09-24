# run.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random
import numpy as np
import pandas as pd
import os
from datetime import datetime

from model import ClientModel, ServerTail
from client import Client
from server import Server

def partition_dataset(dataset, num_clients, seed=1234):
    data_per_client = len(dataset) // num_clients
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    parts = []
    for i in range(num_clients):
        start = i * data_per_client
        end = start + data_per_client
        parts.append(indices[start:end])
    return parts

def main():
    torch.manual_seed(0)
    np.random.seed(0)
    random_seed = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # --- hyperparams ---
    num_clients = 8
    tail_in_features = 512
    epochs = 200
    local_epochs = 1
    batch_size = 128
    server_lr = 5e-4
    client_lr = 5e-4

    # results dir & csv
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"federated_learning_results_{timestamp}.csv"
    csv_path = os.path.join(results_dir, csv_filename)

    # prepare CSV columns
    df_columns = ['epoch']
    for i in range(num_clients):
        df_columns.extend([f'client_{i}_accuracy', f'client_{i}_loss'])
    df_columns.extend(['avg_accuracy', 'avg_loss'])
    results_df = pd.DataFrame(columns=df_columns)
    results_df.to_csv(csv_path, index=False)
    print(f"Results will be saved to: {csv_path}")

    # CIFAR-10 transforms (data augmentation for training)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                             std=(0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                             std=(0.2470, 0.2435, 0.2616)),
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    parts = partition_dataset(train_dataset, num_clients)

    # Create server tail
    server_tail = ServerTail(in_features=tail_in_features, hidden=512, out_features=10)
    server = Server(server_tail, lr=server_lr, device=device)

    # create clients
    clients = []
    client_loaders = []
    for i in range(num_clients):
        cm = ClientModel(tail_in_features=tail_in_features)
        client = Client(i, cm, device=device, lr_front=client_lr)
        clients.append(client)
        subset = Subset(train_dataset, parts[i])
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        client_loaders.append(loader)

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    # training loop
    results_df = pd.DataFrame(columns=df_columns)
    for epoch in range(epochs):
        print(f"--- Global round {epoch + 1}/{epochs} ---")

        # 1) server -> clients (tail params)
        tail_state = server.get_tail_state()
        for c in clients:
            c.set_tail_params(tail_state)

        # 2) clients train locally & upload averaged tail grads
        tail_grads_all = []
        for i, c in enumerate(clients):
            grads = c.train_local(client_loaders[i], epochs=local_epochs)
            if grads is None:
                continue
            tail_grads_all.append(grads)
            print(f"Client {i} uploaded tail grads (keys): {list(grads.keys())}")

        # 3) server aggregates and updates tail
        server.aggregate_and_update(tail_grads_all)

        # 4) evaluate clients (with updated tail)
        accs = []
        losses = []
        epoch_results = {'epoch': epoch + 1}
        for i, c in enumerate(clients):
            # ensure client has latest tail
            c.set_tail_params(server.get_tail_state())
            loss, acc = c.evaluate(test_loader)
            accs.append(acc)
            losses.append(loss)
            epoch_results[f'client_{i}_accuracy'] = acc
            epoch_results[f'client_{i}_loss'] = loss

        avg_acc = float(np.mean(accs))
        avg_loss = float(np.mean(losses))
        epoch_results['avg_accuracy'] = avg_acc
        epoch_results['avg_loss'] = avg_loss

        print(f"Eval across clients (avg acc): {avg_acc:.4f}, avg loss: {avg_loss:.4f}")

        # append and save CSV
        new_row = pd.DataFrame([epoch_results])
        if results_df.empty:
            results_df = new_row
        else:
            results_df = pd.concat([results_df, new_row], ignore_index=True)
        results_df.to_csv(csv_path, index=False)

    print("Training finished. Results saved to:", csv_path)

    # final summary
    if not results_df.empty:
        max_acc_epoch = results_df['avg_accuracy'].idxmax() + 1
        max_acc = results_df['avg_accuracy'].max()
        print(f"Best avg_accuracy: {max_acc:.4f} (epoch {max_acc_epoch})")

    return csv_path

if __name__ == "__main__":
    csv_file = main()
