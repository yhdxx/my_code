import csv
import matplotlib.pyplot as plt

# 读取CSV文件
rounds = []
client_losses = {}

with open("loss_log.csv", mode="r") as f:
    reader = csv.reader(f)
    header = next(reader)  # 第一行表头
    client_names = header[1:]  # 跳过 "Round"

    # 初始化存储
    for name in client_names:
        client_losses[name] = []

    # 逐行读取
    for row in reader:
        rounds.append(int(row[0]))
        for i, name in enumerate(client_names):
            client_losses[name].append(float(row[i + 1]))

# 画图
plt.figure(figsize=(8, 5))
for name, losses in client_losses.items():
    plt.plot(rounds, losses, marker="o", label=name)

plt.xlabel("Round")
plt.ylabel("Average Loss")
plt.title("Federated Clients' Loss Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
