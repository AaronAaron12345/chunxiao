# models/target_network.py
import torch
import torch.nn as nn


class TargetNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TargetNetwork, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        ])

    def forward(self, x, params):
        # params: [batch_size, param_size]
        current = 0
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                weight_num = layer.weight.numel()
                bias_num = layer.bias.numel()

                # Extract the current layer's weights and biases
                weight = params[:, current:current + weight_num].view(-1, layer.weight.size(1), layer.weight.size(
                    0))  # [batch_size, in_features, out_features]
                bias = params[:, current + weight_num:current + weight_num + bias_num].view(-1, layer.bias.size(
                    0))  # [batch_size, out_features]

                # Matrix multiplication: [batch_size, 1, in_features] * [batch_size, in_features, out_features] = [batch_size, 1, out_features]
                x = torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias  # [batch_size, out_features]

                current += weight_num + bias_num
            elif isinstance(layer, nn.ReLU):
                x = layer(x)
        return x
