import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models import ClientModel
from config import Config

#表示一个联邦客户端
class FederatedClient:
    def __init__(self, client_id, train_loader, test_loader, config):
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config

        self.total_layers = config.CLIENT_LAYERS[client_id]

        self.model = ClientModel(
            input_size=config.INPUT_SIZE,
            hidden_size=config.HIDDEN_SIZE,
            output_size=config.NUM_CLASSES,
            total_layers=self.total_layers,
            server_layers=config.SERVER_LAYERS
        ).to(config.DEVICE)

        self.criterion = nn.CrossEntropyLoss()

    def train_private_layers(self):
        """训练客户端私有层（固定共享层）"""
        # 冻结共享层
        for param in self.model.shared_layers.parameters():
            param.requires_grad = False

        # 解冻私有层
        for param in self.model.private_layers.parameters():
            param.requires_grad = True

        #为私有层的参数新建一个SGD优化器
        optimizer = optim.SGD(self.model.private_layers.parameters(),
                              lr=self.config.LEARNING_RATE)

        #把整个模型设为训练模式
        self.model.train()
        
        for epoch in range(self.config.CLIENT_EPOCHS):
            for data, target in self.train_loader:
                data, target = data.to(self.config.DEVICE), target.to(self.config.DEVICE)
                data = data.view(data.size(0), -1)

                optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()

    def train_shared_layers_and_get_gradients(self):
        """训练共享层并获取梯度"""
        # 冻结私有层
        for param in self.model.private_layers.parameters():
            param.requires_grad = False

        # 解冻共享层
        for param in self.model.shared_layers.parameters():
            param.requires_grad = True

        optimizer = optim.SGD(self.model.shared_layers.parameters(),
                              lr=self.config.LEARNING_RATE)

        gradients_dict = {}
        self.model.train()

        # 使用一个batch训练并获取梯度
        for data, target in self.train_loader:
            data, target = data.to(self.config.DEVICE), target.to(self.config.DEVICE)
            data = data.view(data.size(0), -1)

            optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()

            # 修复梯度收集：正确识别线性层和激活层
            layer_idx = 0
            for i, layer in enumerate(self.model.shared_layers):
                if isinstance(layer, nn.Linear):
                    # 权重梯度
                    server_weight_key = f'layers.{layer_idx}.weight'
                    gradients_dict[server_weight_key] = layer.weight.grad.clone()

                    # 偏置梯度
                    if layer.bias is not None:
                        server_bias_key = f'layers.{layer_idx}.bias'
                        gradients_dict[server_bias_key] = layer.bias.grad.clone()

                    layer_idx += 1  # 只有线性层会增加服务器模型索引

            optimizer.step()
            break  # 只用一个batch

        return gradients_dict

    def update_shared_layers(self, server_model_state):
        """更新客户端共享层参数"""
        self.model.load_shared_state_dict(server_model_state)

    def evaluate(self):
        """评估客户端模型"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.config.DEVICE), target.to(self.config.DEVICE)
                data = data.view(data.size(0), -1)

                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total if total > 0 else 0
        return accuracy