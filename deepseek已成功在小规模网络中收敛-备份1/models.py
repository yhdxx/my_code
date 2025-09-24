import torch
import torch.nn as nn
from config import Config


class ServerModel(nn.Module):
    """服务器模型（后n层）"""

    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(ServerModel, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        # 构建后n层网络
        current_size = input_size
        for i in range(num_layers):
            if i == num_layers - 1:
                self.layers.append(nn.Linear(current_size, output_size))
            else:
                self.layers.append(nn.Linear(current_size, hidden_size))
                self.layers.append(nn.ReLU())
                current_size = hidden_size

    #前向传播
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

#客户端的完整模型
class ClientModel(nn.Module):
    """客户端完整模型（mi层）"""

    def __init__(self, input_size, hidden_size, output_size, total_layers, server_layers):
        super(ClientModel, self).__init__()
        self.total_layers = total_layers
        self.server_layers = server_layers
        self.client_layers = total_layers - server_layers

        # 客户端私有层（前mi-n层）
        self.private_layers = nn.ModuleList()
        # 服务器共享层（后n层）
        self.shared_layers = nn.ModuleList()

        # 构建客户端私有层
        current_size = input_size
        for i in range(self.client_layers):
            if i == self.client_layers - 1:
                self.private_layers.append(nn.Linear(current_size, hidden_size))
            else:
                self.private_layers.append(nn.Linear(current_size, hidden_size))
                self.private_layers.append(nn.ReLU())
            current_size = hidden_size

        # 构建共享层（结构与服务器相同）
        shared_input_size = hidden_size if self.client_layers > 0 else input_size
        for i in range(server_layers):
            if i == server_layers - 1:
                self.shared_layers.append(nn.Linear(shared_input_size, output_size))
            else:
                self.shared_layers.append(nn.Linear(shared_input_size, hidden_size))
                self.shared_layers.append(nn.ReLU())
                shared_input_size = hidden_size

    def forward_private(self, x):
        """前向传播通过私有层"""
        for layer in self.private_layers:
            x = layer(x)
        return x

    def forward_shared(self, x):
        """前向传播通过共享层"""
        for layer in self.shared_layers:
            x = layer(x)
        return x

    #先过私有层再过共享层
    def forward(self, x):
        x = self.forward_private(x)
        x = self.forward_shared(x)
        return x

    #导出共享层权重
    def get_shared_state_dict(self):
        """获取共享层的状态字典（与服务器模型兼容）"""
        shared_state_dict = {}
        for i, layer in enumerate(self.shared_layers):
            if hasattr(layer, 'weight'):
                shared_state_dict[f'layers.{2 * i}.weight'] = layer.weight.data
                shared_state_dict[f'layers.{2 * i}.bias'] = layer.bias.data
        return shared_state_dict

    #从服务器下发的参数更新共享层
    def load_shared_state_dict(self, state_dict):
        """加载共享层的状态字典"""
        shared_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('layers.'):
                # 转换服务器模型键名到客户端共享层键名
                parts = key.split('.')
                layer_idx = int(parts[1])
                param_name = parts[2]

                if layer_idx % 2 == 0:  # 线性层
                    actual_layer_idx = layer_idx // 2
                    if actual_layer_idx < len(self.shared_layers):
                        new_key = f'{actual_layer_idx}.{param_name}'
                        shared_state_dict[new_key] = value

        # 加载转换后的状态字典
        if shared_state_dict:
            self.shared_layers.load_state_dict(shared_state_dict, strict=False)