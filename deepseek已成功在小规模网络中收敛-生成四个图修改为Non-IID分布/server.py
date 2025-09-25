# server.py
import torch
import torch.nn as nn
from collections import OrderedDict


class Server:
    def __init__(self, server_model, device='cpu'):
        self.device = device
        self.model = server_model.to(device)

    def aggregate_and_update(self, list_of_tail_params_with_weights):
        """
        聚合客户端上传的尾端参数并更新服务器模型 (FedAvg)
        list_of_tail_params_with_weights: list of tuples (param_dict, weight)
        """
        if not list_of_tail_params_with_weights:
            return

        total_weight = sum(float(w) for _, w in list_of_tail_params_with_weights)
        if total_weight == 0:
            return

        # 获取第一个客户端的参数字典作为模板
        first_params_dict = list_of_tail_params_with_weights[0][0]

        # 1. 初始化平均参数字典
        avg_state_dict = OrderedDict()
        for k, v in first_params_dict.items():
            # 复制第一个客户端的值作为起始，并移动到设备上
            avg_state_dict[k] = v.to(self.device).clone()

            # 如果是浮点型参数 (权重/BN统计量)，将其归零，准备进行加权平均
            if v.is_floating_point():
                avg_state_dict[k].zero_()

        # 2. 对浮点型参数进行加权平均
        for params_dict, w in list_of_tail_params_with_weights:
            weight = float(w) / total_weight
            for k, v in params_dict.items():
                if v.is_floating_point():
                    # 仅对浮点型参数执行 FedAvg 操作
                    avg_state_dict[k] += v.to(self.device) * weight
                # 对于非浮点型参数 (如 num_batches_tracked)，它们已经在第一步中从第一个客户端处复制，
                # 且无需进行加权平均，故在此步骤跳过。

        # 将平均后的参数加载到服务器模型
        self.model.load_state_dict(avg_state_dict)

    def get_tail_state(self):
        # 确保返回 CPU 上的参数
        return {k: v.cpu() for k, v in self.model.state_dict().items()}