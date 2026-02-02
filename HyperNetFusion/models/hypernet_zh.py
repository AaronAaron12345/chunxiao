import torch
import torch.nn as nn


# 定义超网络（HyperNet）类
class HyperNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, param_dim):
        super(HyperNet, self).__init__()  # 初始化父类（nn.Module）
        # 定义一个全连接层的序列
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 从输入维度到隐藏层维度的线性变换
            nn.ReLU(),  # ReLU激活函数
            nn.Linear(hidden_dim, param_dim)  # 从隐藏层到输出参数维度的线性变换（这些参数将会被用作目标网络的权重和偏置）
        )

    def forward(self, x):
        # 前向传播：传入输入x，得到参数
        params = self.network(x)  # [batch_size, param_dim]
        return params  # 返回生成的参数
