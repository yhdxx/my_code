
import torch
import torch.nn as nn
import torch.optim as optim

class Client:
    def __init__(self, cid, model: nn.Module, device='cpu', lr_front=0.01):
        self.cid = cid
        self.device = device
        self.model = model.to(device)

        # 只有在前端存在可训练参数时才创建 optimizer
        front_params = list(self.model.front.parameters())
        if len(front_params) > 0:
            self.front_opt = optim.SGD(front_params, lr=lr_front)
        else:
            # 前端无参数（例如 --front_layers 784），跳过前端优化器
            self.front_opt = None

        self.criterion = nn.CrossEntropyLoss()

    def set_tail_params(self, tail_state_dict):
        """接收服务器下发的共享层参数"""
        self.model.tail.load_state_dict(tail_state_dict)

    def train_local(self, dataloader, epochs=1, send_every_batch=False):
        """
        本地训练：更新前端（若存在），收集并返回尾端梯度的平均值以及样本数量
        返回: (tail_grads_dict, num_samples)
        """
        self.model.train()
        tail_grad_sum = {}
        batch_count = 0
        total_samples = 0

        for epoch in range(epochs):
            for xb, yb in dataloader:
                xb = xb.to(self.device); yb = yb.to(self.device)
                batch_size = xb.size(0)
                total_samples += batch_size

                # 清空前端梯度（如果有 optimizer）
                if self.front_opt is not None:
                    self.front_opt.zero_grad()

                # 清空尾端梯度
                for p in self.model.tail.parameters():
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()

                out = self.model(xb)
                loss = self.criterion(out, yb)
                loss.backward()

                # 更新前端（如果存在）
                if self.front_opt is not None:
                    self.front_opt.step()

                # 收集尾端梯度（clone 到 CPU）
                for name, p in self.model.tail.named_parameters():
                    if p.grad is not None:
                        g = p.grad.detach().clone().cpu()
                        if name in tail_grad_sum:
                            tail_grad_sum[name] += g
                        else:
                            tail_grad_sum[name] = g.clone()
                batch_count += 1

                if send_every_batch:
                    avg = {k: (v.clone() / 1.0) for k, v in tail_grad_sum.items()}
                    return avg, total_samples

        if batch_count == 0:
            return None, 0

        tail_grads_avg = {k: (v / float(batch_count)) for k, v in tail_grad_sum.items()}
        return tail_grads_avg, total_samples

    def evaluate(self, dataloader):
        self.model.eval()
        correct = 0
        total = 0
        loss_total = 0.0
        with torch.no_grad():
            for xb, yb in dataloader:
                xb = xb.to(self.device); yb = yb.to(self.device)
                out = self.model(xb)
                loss = self.criterion(out, yb)
                loss_total += loss.item() * xb.size(0)
                preds = out.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += xb.size(0)
        if total == 0:
            return 0.0, 0.0
        return loss_total / total, correct / total
