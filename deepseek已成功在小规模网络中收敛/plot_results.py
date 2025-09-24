import matplotlib.pyplot as plt
import pandas as pd
import sys

def plot_results(csv_path, save_path=None):
    # 读取CSV文件
    df = pd.read_csv(csv_path)

    # 检查必须的列
    if 'avg_accuracy' not in df.columns or 'avg_loss' not in df.columns:
        raise ValueError("CSV文件中缺少 avg_accuracy 或 avg_loss 列")

    epochs = df['epoch']
    avg_acc = df['avg_accuracy']
    avg_loss = df['avg_loss']

    # 找出客户端的准确率列（匹配 client_X_accuracy）
    client_acc_cols = [col for col in df.columns if col.startswith("client_") and col.endswith("_accuracy")]

    # --- 绘制图 ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # (1) 客户端准确率 + 平均准确率
    for col in client_acc_cols:
        ax1.plot(epochs, df[col], alpha=0.5, label=col)
    ax1.plot(epochs, avg_acc, color='black', linewidth=2, label='avg_accuracy')
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1.0)
    ax1.set_title("Client Accuracies and Average Accuracy")
    ax1.legend(loc='lower right', fontsize=8)

    # (2) 平均Loss曲线
    ax2.plot(epochs, avg_loss, color='red', label='avg_loss')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Average Loss")
    ax2.set_title("Average Loss Across Clients")
    ax2.legend(loc='upper right')

    plt.tight_layout()

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"图像已保存到: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python plot_results.py <csv文件路径> [输出图片路径]")
        sys.exit(1)

    csv_file = sys.argv[1]
    save_file = sys.argv[2] if len(sys.argv) >= 3 else None

    plot_results(csv_file, save_file)
