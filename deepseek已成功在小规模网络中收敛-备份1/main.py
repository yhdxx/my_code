import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from server import FederatedServer
from client import FederatedClient
from config import Config


def load_datasets():
    """加载并划分MNIST数据集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载训练集和测试集
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)

    return trainset, testset


def create_client_datasets(trainset, testset, num_clients):
    """为每个客户端创建数据子集"""
    client_trainsets = []
    client_testsets = []

    # 划分训练数据
    data_indices = list(range(len(trainset)))
    np.random.shuffle(data_indices)
    client_data_splits = np.array_split(data_indices, num_clients)

    for i in range(num_clients):
        client_trainset = Subset(trainset, client_data_splits[i])
        # 每个客户端使用相同的测试集进行评估
        client_testset = testset

        client_trainsets.append(client_trainset)
        client_testsets.append(client_testset)

    return client_trainsets, client_testsets


def plot_accuracy(accuracy_history, config):
    """绘制精度曲线图"""
    plt.figure(figsize=(12, 8))

    # 绘制每个客户端的精度曲线
    for client_id in range(config.NUM_CLIENTS):
        rounds = [r for r, acc_dict in accuracy_history if client_id in acc_dict]
        accuracies = [accuracy_history[i][1][client_id] for i in range(len(rounds))]

        plt.plot(rounds, accuracies,
                 marker='o',
                 label=f'Client {client_id} ({config.CLIENT_LAYERS[client_id]} layers)',
                 linewidth=2)

    plt.xlabel('Training Round', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Federated Learning - Client Accuracy over Rounds', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存图像
    plt.savefig('federated_learning_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 打印最终精度
    print("\n=== Final Accuracy Summary ===")
    if accuracy_history:
        last_round, last_accuracies = accuracy_history[-1]
        for client_id, accuracy in last_accuracies.items():
            print(f"Client {client_id}: {accuracy:.2f}%")


def main():
    config = Config()

    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)

    # 加载数据
    print("Loading datasets...")
    trainset, testset = load_datasets()
    client_trainsets, client_testsets = create_client_datasets(
        trainset, testset, config.NUM_CLIENTS)

    # 创建服务器和客户端
    server = FederatedServer(config)
    clients = []

    for i in range(config.NUM_CLIENTS):
        train_loader = DataLoader(client_trainsets[i], batch_size=config.BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(client_testsets[i], batch_size=config.BATCH_SIZE, shuffle=False)

        client = FederatedClient(i, train_loader, test_loader, config)
        clients.append(client)
        print(f"Client {i} created with {config.CLIENT_LAYERS[i]} layers")

    print("Starting federated learning...")

    # 初始化精度记录
    accuracy_history = []  # 格式: [(round_num, {client_id: accuracy})]

    # 联邦学习训练循环
    for round_num in range(config.NUM_ROUNDS):
        print(f"\n--- Round {round_num + 1}/{config.NUM_ROUNDS} ---")

        # 选择部分客户端参与训练
        selected_indices = np.random.choice(
            range(config.NUM_CLIENTS),
            size=max(1, int(config.FRACTION * config.NUM_CLIENTS)),
            replace=False
        )
        selected_clients = [clients[i] for i in selected_indices]

        print(f"Selected clients: {selected_indices}")

        # 客户端本地训练
        for client in selected_clients:
            try:
                # 步骤1: 用服务器模型更新共享层
                server_model_state = server.get_server_model_state()
                client.update_shared_layers(server_model_state)

                # 步骤2: 训练私有层
                client.train_private_layers()

                # 步骤3: 训练共享层并获取梯度
                gradients = client.train_shared_layers_and_get_gradients()

                # 上传梯度到服务器
                server.receive_gradients(gradients)
                print(f"Client {client.client_id} gradients uploaded")

            except Exception as e:
                print(f"Error training client {client.client_id}: {e}")
                continue

        # 服务器聚合梯度
        server.aggregate_gradients()
        print("Gradients aggregated by server")

        # 每轮都进行评估，但只记录部分轮次的结果以减少存储和绘图复杂度
        if (round_num + 1) % 5 == 0 or round_num == 0 or round_num == config.NUM_ROUNDS - 1:
            round_accuracies = {}
            valid_clients = 0
            total_accuracy = 0

            for client in clients:
                try:
                    accuracy = client.evaluate()
                    round_accuracies[client.client_id] = accuracy
                    total_accuracy += accuracy
                    valid_clients += 1
                    print(f"Client {client.client_id} accuracy: {accuracy:.2f}%")
                except Exception as e:
                    print(f"Error evaluating client {client.client_id}: {e}")
                    continue

            if valid_clients > 0:
                avg_accuracy = total_accuracy / valid_clients
                accuracy_history.append((round_num + 1, round_accuracies))
                print(f"Round {round_num + 1}: Average Client Accuracy = {avg_accuracy:.2f}%")

    # 绘制精度曲线
    plot_accuracy(accuracy_history, config)


if __name__ == "__main__":
    main()