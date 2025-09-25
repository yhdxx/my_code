import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import glob


def plot_federated_learning_results(csv_file):
    """
    读取联邦学习结果CSV文件并绘制图表
    """
    # 读取数据
    df = pd.read_csv(csv_file)

    # 设置中文字体（如果需要显示中文）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # 获取客户端数量
    client_columns = [col for col in df.columns if 'client_' in col and 'accuracy' in col]
    num_clients = len(client_columns)

    # 绘制准确率图表 - 将pandas Series转换为numpy数组
    epochs = df['epoch'].values  # 添加.values转换为numpy数组

    # 自动生成颜色（支持任意数量客户端）
    cmap = plt.cm.get_cmap("tab20", num_clients)
    colors = [cmap(i) for i in range(num_clients)]

    # 图1: 所有客户端的准确率
    for i in range(num_clients):
        acc_col = f'client_{i}_accuracy'
        # 转换为numpy数组
        acc_values = df[acc_col].values
        ax1.plot(epochs, acc_values,
                 label=f'Client {i}',
                 color=colors[i % len(colors)],
                 alpha=0.7,
                 linewidth=2)

    # 绘制平均准确率 - 转换为numpy数组
    avg_acc_values = df['avg_accuracy'].values
    ax1.plot(epochs, avg_acc_values, label='Average', color='black', linewidth=3, linestyle='--')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Federated Learning - Accuracy over Epochs')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # 图2: 所有客户端的损失
    for i in range(num_clients):
        loss_col = f'client_{i}_loss'
        # 转换为numpy数组
        loss_values = df[loss_col].values
        ax2.plot(epochs, loss_values,
                 label=f'Client {i}',
                 color=colors[i % len(colors)],
                 alpha=0.7,
                 linewidth=2)

    # 绘制平均损失 - 转换为numpy数组
    avg_loss_values = df['avg_loss'].values
    ax2.plot(epochs, avg_loss_values, label='Average', color='black', linewidth=3, linestyle='--')

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Federated Learning - Loss over Epochs')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 图3: 平均准确率和损失（双Y轴）
    ax3_twin = ax3.twinx()

    # 转换为numpy数组
    accuracy_line = ax3.plot(epochs, avg_acc_values, label='Avg Accuracy', color='blue', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy', color='blue')
    ax3.tick_params(axis='y', labelcolor='blue')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)

    loss_line = ax3_twin.plot(epochs, avg_loss_values, label='Avg Loss', color='red', linewidth=2)
    ax3_twin.set_ylabel('Loss', color='red')
    ax3_twin.tick_params(axis='y', labelcolor='red')

    lines = accuracy_line + loss_line
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper left')
    ax3.set_title('Average Accuracy and Loss')

    # 图4: 最终轮次各客户端性能对比
    final_accuracies = [df[f'client_{i}_accuracy'].iloc[-1] for i in range(num_clients)]
    final_losses = [df[f'client_{i}_loss'].iloc[-1] for i in range(num_clients)]

    x_pos = np.arange(num_clients)
    width = 0.35

    bars1 = ax4.bar(x_pos - width / 2, final_accuracies, width, label='Accuracy', alpha=0.7, color='lightblue')
    bars2 = ax4.bar(x_pos + width / 2, final_losses, width, label='Loss', alpha=0.7, color='lightcoral')

    ax4.set_xlabel('Client ID')
    ax4.set_ylabel('Value')
    ax4.set_title('Final Epoch: Client Performance Comparison')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f'Client {i}' for i in range(num_clients)])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    for bar in bars1:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.3f}', ha='center', va='bottom')

    for bar in bars2:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.3f}', ha='center', va='bottom')

    plt.tight_layout()

    plot_filename = csv_file.replace('.csv', '_plot.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"图表已保存为: {plot_filename}")

    plt.show()

    print("\n=== 详细训练结果统计 ===")
    print(f"训练轮次: {len(df)}")
    print(f"最终平均准确率: {df['avg_accuracy'].iloc[-1]:.4f}")
    print(f"最终平均损失: {df['avg_loss'].iloc[-1]:.4f}")
    print(f"最高平均准确率: {df['avg_accuracy'].max():.4f} (第 {df['avg_accuracy'].idxmax() + 1} 轮)")

    print("\n各客户端最终表现:")
    for i in range(num_clients):
        print(f"Client {i}: 准确率={df[f'client_{i}_accuracy'].iloc[-1]:.4f}, 损失={df[f'client_{i}_loss'].iloc[-1]:.4f}")

    return fig


def find_latest_csv(results_dir="results"):
    """查找最新的CSV文件"""
    if not os.path.exists(results_dir):
        return None

    csv_files = glob.glob(os.path.join(results_dir, "federated_learning_results_*.csv"))
    if not csv_files:
        return None

    csv_files.sort(key=os.path.getmtime, reverse=True)
    return csv_files[0]


def main():
    parser = argparse.ArgumentParser(description='绘制联邦学习结果图表')
    parser.add_argument('--csv', type=str, default=None,
                        help='CSV文件路径（如果未指定，将自动查找最新的结果文件）')
    parser.add_argument('--results-dir', type=str, default="results",
                        help='结果文件目录（默认: results）')

    args = parser.parse_args()

    if args.csv:
        csv_file = args.csv
    else:
        csv_file = find_latest_csv(args.results_dir)
        if csv_file:
            print(f"自动选择最新文件: {csv_file}")
        else:
            print("未找到CSV文件，请先运行训练脚本")
            return

    if not os.path.exists(csv_file):
        print(f"文件不存在: {csv_file}")
        return

    plot_federated_learning_results(csv_file)


if __name__ == "__main__":
    main()