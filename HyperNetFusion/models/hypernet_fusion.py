# models/hypernet_fusion.py
import torch
import torch.nn as nn
from .hypernet import HyperNet
from .target_network import TargetNetwork


class HyperNetFusion(nn.Module):
    def __init__(self, hypernet_input_dim, hypernet_hidden_dim, target_net_input_dim, target_net_hidden_dim,
                 target_net_output_dim, num_target_nets):
        super(HyperNetFusion, self).__init__()

        # Calculate the number of parameters required for each TargetNetwork
        single_target_net = TargetNetwork(target_net_input_dim, target_net_hidden_dim, target_net_output_dim)
        param_dim_per_target = sum(p.numel() for p in single_target_net.parameters())
        total_param_dim = param_dim_per_target * num_target_nets

        self.hypernet = HyperNet(hypernet_input_dim, hypernet_hidden_dim, total_param_dim)
        self.target_nets = nn.ModuleList([
            TargetNetwork(target_net_input_dim, target_net_hidden_dim, target_net_output_dim)
            for _ in range(num_target_nets)
        ])

    def forward(self, x, hypernet_input):
        params = self.hypernet(hypernet_input)  # [batch_size, total_param_dim]
        # Split the parameters for each TargetNetwork
        param_dim_per_target = sum(p.numel() for p in self.target_nets[0].parameters())
        params = params.view(params.size(0), len(self.target_nets),
                             param_dim_per_target)  # [batch_size, num_target_nets, param_dim_per_target]
        outputs = []
        for i, target_net in enumerate(self.target_nets):
            output = target_net(x, params[:, i, :])  # [batch_size, output_dim]
            outputs.append(output)
        # Fuse the outputs of multiple target networks, e.g., by averaging
        outputs = torch.stack(outputs, dim=0).mean(dim=0)
        return outputs
