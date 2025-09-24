import torch
from collections import defaultdict
import copy
from model import ServerNet


class Server:
    def __init__(self, n, num_clients):
        # 这里n的值不再影响ServerNet的结构，因为ClientNet的后n层是固定的
        self.server_model = ServerNet(n=3)  # n在这里只是一个标识符，实际模型固定为3层
        self.num_clients = num_clients
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.server_model.parameters(), lr=0.01)

    def aggregate_gradients(self, client_grads_list):
        aggregated_grads = defaultdict(lambda: None)

        for client_grads in client_grads_list:
            for name, grad in client_grads.items():
                if grad is not None:
                    # 获取与服务器模型参数相对应的名称
                    server_param_name = '.'.join(name.split('.')[1:])
                    if aggregated_grads[server_param_name] is None:
                        aggregated_grads[server_param_name] = torch.zeros_like(grad)
                    aggregated_grads[server_param_name] += grad

        for name, grad in aggregated_grads.items():
            if grad is not None:
                aggregated_grads[name] /= self.num_clients

        return aggregated_grads

    def update_and_get_params(self, aggregated_grads):
        # Manually update server model parameters using the aggregated grads
        for name, param in self.server_model.named_parameters():
            if name in aggregated_grads and aggregated_grads[name] is not None:
                param.data -= self.optimizer.param_groups[0]['lr'] * aggregated_grads[name]

        return copy.deepcopy(self.server_model)