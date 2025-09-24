import matplotlib.pyplot as plt
import csv

def read_loss_log(filename="loss_log.csv"):
    rounds = []
    losses = []
    with open(filename, "r") as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头
        for row in reader:
            rounds.append(int(row[0]))
            losses.append(float(row[1]))
    return rounds, losses

def plot_loss(rounds, losses):
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, losses, marker="o", linewidth=2)
    plt.xlabel("Round")
    plt.ylabel("Average Train Loss")
    plt.title("Federated Learning Training Loss Curve")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    rounds, losses = read_loss_log("loss_log.csv")
    plot_loss(rounds, losses)
