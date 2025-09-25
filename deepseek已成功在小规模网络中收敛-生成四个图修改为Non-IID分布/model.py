import torch
import torch.nn as nn
import torch.nn.init as init


class ServerTail(nn.Module):
    """服务器端共享层 (尾部)，支持灵活设置层数"""
    def __init__(self, in_features=128, hidden=256, out_features=10, num_layers=6):
        super().__init__()
        layers = []
        input_dim = in_features

        effective_layers = max(1, num_layers)

        for i in range(effective_layers - 1):
            layers.append(nn.Linear(input_dim, hidden))
            # 引入 BatchNorm1d
            layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            input_dim = hidden

        layers.append(nn.Linear(input_dim, out_features))
        self.net = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)


class ClientModel(nn.Module):
    """客户端完整模型 = 前端(隐私层) + 尾端(共享层)"""
    def __init__(self, front_sizes=[784, 256, 128], tail_layers=6, num_classes=10):
        super().__init__()
        layers = []
        for i in range(len(front_sizes) - 1):
            layers.append(nn.Linear(front_sizes[i], front_sizes[i + 1]))
            # 引入 BatchNorm1d
            layers.append(nn.BatchNorm1d(front_sizes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))

        if len(layers) > 0:
            self.front = nn.Sequential(*layers)
        else:
            self.front = nn.Identity()

        self.tail = ServerTail(in_features=front_sizes[-1], hidden=256, num_layers=tail_layers, out_features=num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        a = self.front(x)
        out = self.tail(a)
        return out