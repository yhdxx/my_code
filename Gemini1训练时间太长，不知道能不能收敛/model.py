import torch.nn as nn

#定义客户端本地模型，分为前端层和后端层
class ClientNet(nn.Module):
    def __init__(self, mi=100, num_classes=10):
        super(ClientNet, self).__init__()
        self.mi = mi

        # 定义前端 mi-n 层
        self.front_layers = nn.ModuleList()
        # 简化为只有一个输入层，其余都是512->512
        self.front_layers.append(nn.Linear(784, 512))
        self.front_layers.append(nn.ReLU())
        # 其余层
        for i in range(mi - 2):  # mi-2个线性层
            self.front_layers.append(nn.Linear(512, 512))
            self.front_layers.append(nn.ReLU())

        # 定义后端 n 层
        self.back_layers = nn.ModuleList([
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        ])

    def forward(self, x):
        x = x.view(-1, 784)
        for layer in self.front_layers:
            x = layer(x)
        for layer in self.back_layers:
            x = layer(x)
        return x


class ServerNet(nn.Module):
    def __init__(self, n, num_classes=10):
        super(ServerNet, self).__init__()
        self.n = n

        self.layers = nn.ModuleList()
        # 定义服务器模型层，它与客户端的后n层完全相同
        # 由于客户端后n层是固定的，服务器模型也应该是固定的
        self.layers.append(nn.Linear(512, 512))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(512, num_classes))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x