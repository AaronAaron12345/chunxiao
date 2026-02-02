import torch
import torch.nn as nn
from .hypernet import HyperNet
from .target_network import TargetNetwork


# 定义超网络融合（HyperNetFusion）类
class HyperNetFusion(nn.Module):
    def __init__(self, hypernet_input_dim, hypernet_hidden_dim, target_net_input_dim, target_net_hidden_dim,
                 target_net_output_dim, num_target_nets):
        super(HyperNetFusion, self).__init__()

        # 计算每个目标网络所需的参数数量
        single_target_net = TargetNetwork(target_net_input_dim, target_net_hidden_dim, target_net_output_dim)
        param_dim_per_target = sum(p.numel() for p in single_target_net.parameters())  # 计算每个目标网络的参数数量
        total_param_dim = param_dim_per_target * num_target_nets  # 计算所有目标网络参数的总维度

        # 定义超网络，输入维度为hypernet_input_dim，输出维度为所有目标网络参数的总维度
        self.hypernet = HyperNet(hypernet_input_dim, hypernet_hidden_dim, total_param_dim)
        # 创建目标网络的列表
        self.target_nets = nn.ModuleList([
            TargetNetwork(target_net_input_dim, target_net_hidden_dim, target_net_output_dim)
            for _ in range(num_target_nets)
        ])

    def forward(self, x, hypernet_input):
        # 通过超网络生成目标网络的参数
        params = self.hypernet(hypernet_input)  # [batch_size, total_param_dim]
        # 将生成的参数拆分为每个目标网络的参数
        param_dim_per_target = sum(p.numel() for p in self.target_nets[0].parameters())
        params = params.view(params.size(0), len(self.target_nets),
                             param_dim_per_target)  # [batch_size, num_target_nets, param_dim_per_target]

        outputs = []
        for i, target_net in enumerate(self.target_nets):
            # 每个目标网络通过自己的参数进行计算
            output = target_net(x, params[:, i, :])  # [batch_size, output_dim]
            outputs.append(output)

        # 融合多个目标网络的输出，例如通过平均融合
        outputs = torch.stack(outputs, dim=0).mean(dim=0)
        return outputs
