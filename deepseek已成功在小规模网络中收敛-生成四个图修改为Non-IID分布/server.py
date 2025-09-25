
# server.py
import torch
import torch.optim as optim

class Server:
    def __init__(self, server_model, lr=0.01, device='cpu', use_adam=True):
        self.device = device
        self.model = server_model.to(device)
        if use_adam:
            self.opt = optim.Adam(self.model.parameters(), lr=lr)
        else:
            self.opt = optim.SGD(self.model.parameters(), lr=lr)

    def aggregate_and_update(self, list_of_tail_grads_with_weights):
        """
        聚合客户端上传的尾端梯度并更新服务器模型
        list_of_tail_grads_with_weights: list of tuples (gdict, weight)
        """
        if not list_of_tail_grads_with_weights:
            return

        total_weight = sum(float(w) for _, w in list_of_tail_grads_with_weights)
        if total_weight == 0:
            return

        first_gdict = list_of_tail_grads_with_weights[0][0]
        avg = {k: torch.zeros_like(v, device=self.device) for k, v in first_gdict.items()}

        for gdict, w in list_of_tail_grads_with_weights:
            weight = float(w) / total_weight
            for k, v in gdict.items():
                avg[k] += v.to(self.device) * weight

        # 设置模型梯度
        self.opt.zero_grad()
        name_to_param = dict(self.model.named_parameters())
        for k, avg_grad in avg.items():
            if k in name_to_param:
                name_to_param[k].grad = avg_grad.clone()
            else:
                raise KeyError(f"Server model missing param {k}")

        self.opt.step()

    def get_tail_state(self):
        return {k: v.cpu() for k, v in self.model.state_dict().items()}
