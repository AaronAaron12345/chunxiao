import torch
import torch.nn as nn


# 定义目标网络（TargetNetwork）类
class TargetNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TargetNetwork, self).__init__()
        # 定义目标网络的层次结构
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),  # 输入到隐藏层的线性变换
            nn.ReLU(),  # ReLU激活函数
            nn.Linear(hidden_dim, output_dim)  # 隐藏层到输出层的线性变换
        ])

    def forward(self, x, params):
        # params：是来自超网络的参数 [batch_size, param_size]
        current = 0
        for layer in self.layers:
            if isinstance(layer, nn.Linear):  # 如果是线性层
                weight_num = layer.weight.numel()  # 获取权重的元素数量
                bias_num = layer.bias.numel()  # 获取偏置的元素数量

                # 提取当前层的权重和偏置
                weight = params[:, current:current + weight_num].view(-1, layer.weight.size(1), layer.weight.size(0))  # [batch_size, in_features, out_features]
                bias = params[:, current + weight_num:current + weight_num + bias_num].view(-1, layer.bias.size(0))  # [batch_size, out_features]

                # 矩阵乘法： [batch_size, 1, in_features] * [batch_size, in_features, out_features] = [batch_size, 1, out_features]
                x = torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias  # [batch_size, out_features]

                current += weight_num + bias_num  # 更新当前参数的索引
            elif isinstance(layer, nn.ReLU):  # 如果是ReLU激活层
                x = layer(x)
        return x
