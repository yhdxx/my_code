
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random, numpy as np, pandas as pd, os
from datetime import datetime
import argparse

from model import ClientModel, ServerTail
from client import Client
from server import Server


def partition_dataset(dataset, num_clients, shards_per_client=2, seed=1234):
    np.random.seed(seed)
    labels = dataset.targets.numpy()
    num_classes = labels.max() + 1

    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    for idxs in class_indices:
        np.random.shuffle(idxs)

    shards = []
    for c in range(num_classes):
        shards.extend(np.array_split(class_indices[c], num_clients))
    np.random.shuffle(shards)

    parts = []
    needed = num_clients * shards_per_client
    assert needed <= len(shards), f"shards 数量不足 ({len(shards)})，请调小 shards_per_client ({shards_per_client})"

    for i in range(num_clients):
        client_shards = shards[i * shards_per_client:(i + 1) * shards_per_client]
        client_indices = np.concatenate(client_shards, axis=0)
        parts.append(client_indices.tolist())

    return parts


def make_client_front_sizes(num_clients, tail_in_features):
    out = []
    for i in range(num_clients):
        h = random.choice([128, 256, 200])
        out.append([784, h, tail_in_features])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", type=int, default=10, help="客户端数量")
    parser.add_argument("--front_layers", type=str, default="784,256,128",
                        help="前端层结构, 逗号分隔，例如 784,256,128 或者 784 （表示无隐私中间层）")
    parser.add_argument("--tail_layers", type=int, default=8, help="共享层数")
    parser.add_argument("--epochs", type=int, default=1000, help="全局训练轮数")
    args = parser.parse_args()

    torch.manual_seed(0); random.seed(0); np.random.seed(0)

    device = 'cpu'
    num_clients = args.num_clients

    # 解析 front_layers
    front_sizes = [int(x.strip()) for x in args.front_layers.split(',') if x.strip() != '']
    if len(front_sizes) == 0:
        raise ValueError("--front_layers 必须至少包含输入维度，例如 --front_layers 784 或 --front_layers 784,256,128")

    tail_in_features = front_sizes[-1]

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    parts = partition_dataset(train_dataset, num_clients, shards_per_client=2)

    # 创建服务器共享层
    server_tail = ServerTail(in_features=tail_in_features, num_layers=args.tail_layers)
    server = Server(server_tail, lr=0.005, device=device, use_adam=True)

    clients = []
    client_loaders = []
    for i in range(num_clients):
        cm = ClientModel(front_sizes, tail_layers=args.tail_layers)
        client = Client(i, cm, device=device, lr_front=0.01)
        clients.append(client)
        subset = Subset(train_dataset, parts[i])
        loader = DataLoader(subset, batch_size=64, shuffle=True)
        client_loaders.append(loader)

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(results_dir, f"federated_results_{timestamp}.csv")

    df_columns = ['epoch']
    for i in range(num_clients):
        df_columns.extend([f'client_{i}_accuracy', f'client_{i}_loss'])
    df_columns.extend(['avg_accuracy', 'avg_loss'])

    results_df = pd.DataFrame(columns=df_columns)

    for epoch in range(args.epochs):
        print(f"--- Global round {epoch + 1}/{args.epochs} ---")

        # 下发共享层参数
        tail_state = server.get_tail_state()
        for c in clients:
            c.set_tail_params(tail_state)

        tail_grads_with_weights = []
        for i, c in enumerate(clients):
            grads, n_samples = c.train_local(client_loaders[i], epochs=1)
            if grads is None or n_samples == 0:
                continue
            tail_grads_with_weights.append((grads, n_samples))
            print(f"Client {i} uploaded averaged tail grads, samples={n_samples}")

        server.aggregate_and_update(tail_grads_with_weights)

        accs, losses = [], []
        epoch_results = {'epoch': epoch + 1}
        for i, c in enumerate(clients):
            c.set_tail_params(server.get_tail_state())
            loss, acc = c.evaluate(test_loader)
            accs.append(acc); losses.append(loss)
            epoch_results[f'client_{i}_accuracy'] = acc
            epoch_results[f'client_{i}_loss'] = loss

        avg_acc = float(np.mean(accs)) if accs else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
        epoch_results['avg_accuracy'] = avg_acc
        epoch_results['avg_loss'] = avg_loss
        print(f"Eval across clients (avg acc): {avg_acc:.4f}, avg loss: {avg_loss:.4f}")

        new_row = pd.DataFrame([epoch_results])
        if results_df.empty:
            results_df = new_row
        else:
            results_df = pd.concat([results_df, new_row], ignore_index=True)
        results_df.to_csv(csv_path, index=False)

    print(f"训练完成！结果已保存到: {csv_path}")

    return csv_path


if __name__ == "__main__":
    main()
