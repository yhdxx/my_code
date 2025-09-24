import torch
import torch.nn as nn
from models import ServerModel
from config import Config


class FederatedServer:
    def __init__(self, config):
        self.config = config
        # 服务器模型的输入尺寸需要匹配客户端私有层的输出
        self.model = ServerModel(
            input_size=config.HIDDEN_SIZE,
            hidden_size=config.HIDDEN_SIZE,
            output_size=config.NUM_CLASSES,
            num_layers=config.SERVER_LAYERS
        ).to(config.DEVICE)

        self.global_round = 0
        self.client_gradients = []

    def aggregate_gradients(self):
        """聚合客户端梯度"""
        if not self.client_gradients:
            return

        # 平均梯度
        avg_gradients = {}
        for param_name in self.client_gradients[0].keys():
            gradients = [client_grads[param_name] for client_grads in self.client_gradients
                         if param_name in client_grads and client_grads[param_name] is not None]
            if gradients:
                avg_gradients[param_name] = torch.stack(gradients).mean(dim=0)

        # 应用平均梯度
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.LEARNING_RATE)
        optimizer.zero_grad()

        # 手动设置梯度
        for name, param in self.model.named_parameters():
            if name in avg_gradients:
                param.grad = avg_gradients[name].clone()

        optimizer.step()
        self.client_gradients = []
        self.global_round += 1

    def receive_gradients(self, gradients):
        """接收客户端上传的梯度"""
        self.client_gradients.append(gradients)

    def get_server_model_state(self):
        """获取服务器模型状态"""
        return self.model.state_dict()