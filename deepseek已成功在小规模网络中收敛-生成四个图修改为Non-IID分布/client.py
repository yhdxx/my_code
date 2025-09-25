import torch
import torch.nn as nn
import torch.optim as optim
from collections import ChainMap


class Client:
    # 接收 lr_front 和 lr_tail 参数
    def __init__(self, cid, model: nn.Module, device='cpu', lr_front=0.005, lr_tail=0.005):
        self.cid = cid
        self.device = device
        self.model = model.to(device)
        self.criterion = nn.CrossEntropyLoss()

        front_params = list(self.model.front.parameters())
        tail_params = list(self.model.tail.parameters())

        # 1. 前端优化器 (使用 lr_front)
        if len(front_params) > 0:
            self.front_opt = optim.SGD(front_params, lr=lr_front, momentum=0.9, weight_decay=1e-4)
        else:
            self.front_opt = None

        # 2. 尾端优化器 (使用 lr_tail)
        self.tail_opt = optim.SGD(tail_params, lr=lr_tail, momentum=0.9, weight_decay=1e-4)

        # 合并所有需要清零和步进的优化器
        self.all_opts = [opt for opt in [self.front_opt, self.tail_opt] if opt is not None]

    def set_tail_params(self, tail_state_dict):
        """接收服务器下发的共享层参数"""
        self.model.tail.load_state_dict(tail_state_dict)

    # 关键修改：返回参数 (state_dict) 而不是梯度
    def train_local(self, dataloader, epochs=3):
        """
        本地训练：同时更新前端 F_i 和尾端 T。
        返回: (tail_params_dict, num_samples)
        """
        self.model.train()
        total_samples = 0

        for epoch in range(epochs):
            for xb, yb in dataloader:
                xb = xb.to(self.device);
                yb = yb.to(self.device)
                batch_size = xb.size(0)
                total_samples += batch_size

                # 清空所有参数梯度
                for opt in self.all_opts:
                    opt.zero_grad()

                out = self.model(xb)
                loss = self.criterion(out, yb)
                loss.backward()

                # 更新所有参数 (前端和尾端)
                for opt in self.all_opts:
                    # 梯度裁剪 (针对整个模型)
                    all_params = ChainMap(*[opt.param_groups[0]['params'] for opt in self.all_opts])
                    torch.nn.utils.clip_grad_norm_(list(all_params.keys()), max_norm=1.0)
                    opt.step()

        if total_samples == 0:
            return None, 0

        # 返回训练后的共享层参数字典 (FedAvg 要求的)
        return {k: v.cpu() for k, v in self.model.tail.state_dict().items()}, total_samples

    def evaluate(self, dataloader):
        self.model.eval()
        correct = 0
        total = 0
        loss_total = 0.0
        with torch.no_grad():
            for xb, yb in dataloader:
                xb = xb.to(self.device);
                yb = yb.to(self.device)
                out = self.model(xb)
                loss = self.criterion(out, yb)
                loss_total += loss.item() * xb.size(0)
                preds = out.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += xb.size(0)
        if total == 0:
            return 0.0, 0.0
        return loss_total / total, correct / total