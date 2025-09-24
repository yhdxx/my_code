import pandas as pd
import matplotlib.pyplot as plt

# 读取日志文件
log_file = "logs/train_log.csv"
df = pd.read_csv(log_file)

print("CSV 列：", df.columns)

plt.figure(figsize=(10,5))

# 子图1: 训练 loss
plt.subplot(1,2,1)
plt.plot(df["round"], df["train_loss_mean"], marker="o", label="Train Loss (mean)")
plt.xlabel("Round")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.legend()

# 子图2: 测试准确率
plt.subplot(1,2,2)
plt.plot(df["round"], df["test_acc"], marker="s", color="orange", label="Test Accuracy")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.title("Test Accuracy Curve")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("logs/training_curves.png", dpi=300)
plt.close()

print("绘图完成，结果已保存到 logs/training_curves.png")
