## README / 运行说明

1. 环境：Python 3.8+, PyTorch, torchvision
2. 在项目目录中保存上述文件为 `model.py`, `client.py`, `server.py`, `run.py`。
3. 运行：

```bash
python run.py --num_clients 5 --front_layers 784,256,128 --tail_layers 6 --epochs 50
```

4. 脚本会生成 `loss_log.csv`，其中每一行是每轮（round）每个客户端的平均 loss，可以用 Excel 或 matplotlib 绘图。

---

**注意：上面的实现为教学与实验用途，做了大量简化（例如没有考虑 secure aggregation、没有可靠的超参、没有严格对齐的命名约定等）。它用于演示你的方法的工作流程与可行性。**
