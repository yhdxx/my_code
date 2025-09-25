
import torch
import torch.nn as nn

class ServerTail(nn.Module):
    """服务器端共享层 (尾部)，支持灵活设置层数"""
    def __init__(self, in_features=128, hidden=64, out_features=10, num_layers=8):
        super().__init__()
        layers = []
        input_dim = in_features

        for i in range(num_layers - 1):
            layers.append(nn.Linear(input_dim, hidden))
            layers.append(nn.ReLU())
            input_dim = hidden

        # 输出层
        layers.append(nn.Linear(input_dim, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ClientModel(nn.Module):
    """客户端完整模型 = 前端(隐私层) + 尾端(共享层)"""
    def __init__(self, front_sizes=[784, 256, 128], tail_layers=8, num_classes=10):
        """
        front_sizes: list，例如 [784, 256, 128]
                     最后一个元素为共享层输入维度
        tail_layers: int，共享层层数
        num_classes: 分类类别数
        """
        super().__init__()
        layers = []
        for i in range(len(front_sizes) - 1):
            layers.append(nn.Linear(front_sizes[i], front_sizes[i+1]))
            layers.append(nn.ReLU())

        # 如果没有前端中间层（例如 front_sizes == [784]），使用 Identity
        if len(layers) > 0:
            self.front = nn.Sequential(*layers)
        else:
            # 输入已经会在 forward 中被 flatten 为 (batch, 784)
            self.front = nn.Identity()

        # 共享尾部（结构与服务器端一致）
        self.tail = ServerTail(in_features=front_sizes[-1], num_layers=tail_layers, out_features=num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        a = self.front(x)
        out = self.tail(a)
        return out

