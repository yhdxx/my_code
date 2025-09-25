import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random, numpy as np, pandas as pd, os
from datetime import datetime
import argparse

from model import ClientModel, ServerTail
from client import Client
from server import Server


def partition_dataset_dirichlet(dataset, num_clients, alpha=0.5, seed=1234):
    """
    使用狄利克雷分布划分数据集，创建非独立同分布数据
    alpha: 狄利克雷分布参数，控制数据分布的非平衡程度
          alpha越小，数据分布越不平衡；alpha越大，数据分布越平衡
    """
    np.random.seed(seed)
    labels = dataset.targets.numpy() if hasattr(dataset.targets, 'numpy') else np.array(dataset.targets)
    num_classes = len(np.unique(labels))

    # 按类别组织数据索引
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]

    # 为每个类别生成狄利克雷分布
    client_proportions = np.random.dirichlet([alpha] * num_clients, num_classes)

    # 分配数据到各个客户端
    client_indices = [[] for _ in range(num_clients)]

    for class_id in range(num_classes):
        class_data = class_indices[class_id]
        np.random.shuffle(class_data)

        # 根据狄利克雷分布比例分配当前类别的数据
        proportions = client_proportions[class_id]
        proportions = proportions / proportions.sum()  # 确保归一化
        cumulative_prop = np.cumsum(proportions)

        start_idx = 0
        for client_id in range(num_clients):
            if client_id == num_clients - 1:
                # 最后一个客户端获取剩余所有数据
                end_idx = len(class_data)
            else:
                end_idx = int(cumulative_prop[client_id] * len(class_data))

            num_samples = end_idx - start_idx
            if num_samples > 0:
                client_indices[client_id].extend(class_data[start_idx:end_idx].tolist())
            start_idx = end_idx

    # 打乱每个客户端内部的数据顺序
    for indices in client_indices:
        np.random.shuffle(indices)

    return client_indices


def partition_dataset_shards(dataset, num_clients, shards_per_client=2, seed=1234):
    """原有的基于shards的划分方法（保留作为备选）"""
    np.random.seed(seed)
    labels = dataset.targets.numpy() if hasattr(dataset.targets, 'numpy') else np.array(dataset.targets)
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


def analyze_data_distribution(dataset, parts, num_clients):
    """分析数据分布情况"""
    labels = dataset.targets.numpy() if hasattr(dataset.targets, 'numpy') else np.array(dataset.targets)
    num_classes = len(np.unique(labels))

    print("\n数据分布分析:")
    print("客户端\t总样本数\t", end="")
    for i in range(num_classes):
        print(f"类别{i}\t", end="")
    print()

    for client_id in range(num_clients):
        client_indices = parts[client_id]
        client_labels = labels[client_indices]

        print(f"{client_id}\t{len(client_indices)}\t\t", end="")
        for class_id in range(num_classes):
            count = np.sum(client_labels == class_id)
            print(f"{count}\t", end="")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", type=int, default=10, help="客户端数量")
    parser.add_argument("--front_layers", type=str, default="784,256,128",
                        help="前端层结构, 逗号分隔")
    parser.add_argument("--tail_layers", type=int, default=6, help="共享层数")
    parser.add_argument("--epochs", type=int, default=100, help="全局训练轮数")
    parser.add_argument("--local_epochs", type=int, default=3, help="客户端本地训练轮数")
    parser.add_argument("--alpha", type=float, default=0.5, help="狄利克雷分布参数alpha，控制数据不平衡程度")
    parser.add_argument("--partition_method", type=str, default="dirichlet",
                        choices=["dirichlet", "shards"], help="数据划分方法")
    args = parser.parse_args()

    # 稳定的低学习率 (0.005)
    LR_FRONT = 0.005
    LR_TAIL = 0.005

    torch.manual_seed(0);
    random.seed(0);
    np.random.seed(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    num_clients = args.num_clients

    front_sizes = [int(x.strip()) for x in args.front_layers.split(',') if x.strip() != '']
    if len(front_sizes) == 0:
        raise ValueError("--front_layers 必须至少包含输入维度")

    tail_in_features = front_sizes[-1]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # 根据参数选择数据划分方法
    if args.partition_method == "dirichlet":
        print(f"使用狄利克雷分布划分数据，alpha={args.alpha}")
        parts = partition_dataset_dirichlet(train_dataset, num_clients, alpha=args.alpha)
    else:
        print("使用shards方法划分数据")
        parts = partition_dataset_shards(train_dataset, num_clients, shards_per_client=2)

    # 分析数据分布
    analyze_data_distribution(train_dataset, parts, num_clients)

    server_tail = ServerTail(in_features=tail_in_features, num_layers=args.tail_layers)
    server = Server(server_tail, device=device)

    clients = []
    client_loaders = []
    for i in range(num_clients):
        cm = ClientModel(front_sizes, tail_layers=args.tail_layers)
        # 传入 LR_FRONT 和 LR_TAIL
        client = Client(i, cm, device=device, lr_front=LR_FRONT, lr_tail=LR_TAIL)
        clients.append(client)
        subset = Subset(train_dataset, parts[i])
        loader = DataLoader(subset, batch_size=128, shuffle=True)
        client_loaders.append(loader)

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(results_dir, f"federated_learning_results_{timestamp}.csv")

    df_columns = ['epoch']
    for i in range(num_clients):
        df_columns.extend([f'client_{i}_accuracy', f'client_{i}_loss'])
    df_columns.extend(['avg_accuracy', 'avg_loss'])

    results_df = pd.DataFrame(columns=df_columns)

    for epoch in range(args.epochs):
        print(f"\n--- Global round {epoch + 1}/{args.epochs} ---")

        tail_state = server.get_tail_state()
        for c in clients:
            c.set_tail_params(tail_state)

        tail_params_with_weights = []
        for i, c in enumerate(clients):
            # client.train_local() 现在返回的是参数 (FedAvg)
            params, n_samples = c.train_local(client_loaders[i], epochs=args.local_epochs)
            if params is None or n_samples == 0:
                continue
            tail_params_with_weights.append((params, n_samples))

        # server.aggregate_and_update() 对参数进行 FedAvg
        server.aggregate_and_update(tail_params_with_weights)
        print("Server aggregated and updated shared tail parameters.")

        # --------------------- 评估并打印 ---------------------
        accs, losses = [], []
        epoch_results = {'epoch': epoch + 1}

        for i, c in enumerate(clients):
            c.set_tail_params(server.get_tail_state())
            loss, acc = c.evaluate(test_loader)

            print(f"Client {i} - Test Acc: {acc:.4f}, Loss: {loss:.4f}")

            accs.append(acc);
            losses.append(loss)
            epoch_results[f'client_{i}_accuracy'] = acc
            epoch_results[f'client_{i}_loss'] = loss

        avg_acc = float(np.mean(accs)) if accs else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
        epoch_results['avg_accuracy'] = avg_acc
        epoch_results['avg_loss'] = avg_loss
        print(f"--- Global Average: Acc {avg_acc:.4f}, Loss {avg_loss:.4f} ---")

        new_row = pd.DataFrame([epoch_results])
        if results_df.empty:
            results_df = new_row
        else:
            results_df = pd.concat([results_df, new_row], ignore_index=True)
        results_df.to_csv(csv_path, index=False)

    print(f"\n训练完成！结果已保存到: {csv_path}")

    return csv_path


if __name__ == "__main__":
    main()
'''
# 使用狄利克雷分布，alpha=0.1（高度不平衡）
python run.py --partition_method dirichlet --alpha 0.1

# 使用狄利克雷分布，alpha=1.0（中等不平衡）
python run.py --partition_method dirichlet --alpha 1.0

# 使用狄利克雷分布，alpha=10.0（相对平衡）
python run.py --partition_method dirichlet --alpha 10.0

# 使用原来的shards方法
python run.py --partition_method shards
'''