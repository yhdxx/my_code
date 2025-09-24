# run.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random
import numpy as np
import pandas as pd
import os
from datetime import datetime
import csv

from model import ClientModel, ServerTail
from client import Client
from server import Server


def partition_dataset(dataset, num_clients, seed=1234):
    # simple equal partition
    data_per_client = len(dataset) // num_clients
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    parts = []
    for i in range(num_clients):
        start = i * data_per_client
        end = start + data_per_client
        parts.append(indices[start:end])
    return parts


def make_client_front_sizes(num_clients, tail_in_features):
    # create varied front architectures but same tail input dim
    out = []
    for i in range(num_clients):
        # front: input 784 -> hidden (random among options) -> tail_in_features
        h = random.choice([128, 256, 200])
        out.append([784, h, tail_in_features])
    return out


def main():
    torch.manual_seed(0)
    random_seed = 0
    device = 'cpu'  # change to 'cuda' if available
    num_clients = 4
    tail_in_features = 128  # cut size
    epochs = 100
    local_epochs = 1
    batch_size = 64

    # 创建结果目录
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # 创建CSV文件路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"federated_learning_results_{timestamp}.csv"
    csv_path = os.path.join(results_dir, csv_filename)

    # 初始化CSV文件并写入表头
    df_columns = ['epoch']
    for i in range(num_clients):
        df_columns.extend([f'client_{i}_accuracy', f'client_{i}_loss'])
    df_columns.extend(['avg_accuracy', 'avg_loss'])

    # 使用更安全的方式初始化DataFrame，避免警告
    results_df = pd.DataFrame(columns=df_columns)
    # 先写入一个空行来避免FutureWarning
    results_df.loc[0] = [None] * len(df_columns)
    results_df.to_csv(csv_path, index=False)
    print(f"结果将保存到: {csv_path}")

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    parts = partition_dataset(train_dataset, num_clients)
    client_fronts = make_client_front_sizes(num_clients, tail_in_features)

    # Create server tail model
    server_tail = ServerTail(in_features=tail_in_features)
    server = Server(server_tail, lr=0.01, device=device)

    # Create clients
    clients = []
    client_loaders = []
    for i in range(num_clients):
        front_sizes = client_fronts[i]
        cm = ClientModel(front_sizes, tail_in_features=tail_in_features)
        client = Client(i, cm, device=device, lr_front=0.01)
        clients.append(client)
        subset = Subset(train_dataset, parts[i])
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        client_loaders.append(loader)

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # 重新初始化空的DataFrame用于收集结果
    results_df = pd.DataFrame(columns=df_columns)

    # Main training loop
    for epoch in range(epochs):
        print(f"--- Global round {epoch + 1}/{epochs} ---")
        # Server distributes tail params
        tail_state = server.get_tail_state()
        for c in clients:
            c.set_tail_params(tail_state)
        # Each client trains locally and collects tail grads
        tail_grads_all = []
        for i, c in enumerate(clients):
            grads = c.train_local(client_loaders[i], epochs=local_epochs)
            if grads is None:
                # no grads? skip
                continue
            tail_grads_all.append(grads)
            print(f"Client {i} uploaded tail grads (shapes): {list(grads.keys())}")

        # Server aggregates and updates tail
        server.aggregate_and_update(tail_grads_all)

        # Evaluate clients (on local model with updated tail)
        accs = []
        losses = []
        epoch_results = {'epoch': epoch + 1}

        for i, c in enumerate(clients):
            # make sure client has latest tail
            c.set_tail_params(server.get_tail_state())
            loss, acc = c.evaluate(test_loader)
            accs.append(acc)
            losses.append(loss)
            epoch_results[f'client_{i}_accuracy'] = acc
            epoch_results[f'client_{i}_loss'] = loss

        avg_acc = np.mean(accs)
        avg_loss = np.mean(losses)
        epoch_results['avg_accuracy'] = avg_acc
        epoch_results['avg_loss'] = avg_loss

        print(f"Eval across clients (avg acc): {avg_acc:.4f}, avg loss: {avg_loss:.4f}")

        # 将本轮结果添加到DataFrame并保存到CSV
        # 使用更安全的拼接方式
        new_row = pd.DataFrame([epoch_results])
        if results_df.empty:
            results_df = new_row
        else:
            results_df = pd.concat([results_df, new_row], ignore_index=True)

        results_df.to_csv(csv_path, index=False)

    print(f"训练完成！结果已保存到: {csv_path}")

    # 打印最终统计信息
    print("\n=== 训练结果统计 ===")
    print(f"最终平均准确率: {avg_acc:.4f}")
    print(f"最终平均损失: {avg_loss:.4f}")
    if not results_df.empty:
        max_acc_epoch = results_df['avg_accuracy'].idxmax() + 1
        max_acc = results_df['avg_accuracy'].max()
        print(f"最高平均准确率: {max_acc:.4f} (第 {max_acc_epoch} 轮)")

    return csv_path


if __name__ == "__main__":
    csv_file = main()