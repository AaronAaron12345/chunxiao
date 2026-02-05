# -*- coding: utf-8 -*-
"""
HyperNetwork and Target Network Module
======================================
超网络生成多个目标子网络的权重，实现动态参数生成
"""

import torch
import torch.nn as nn
import numpy as np


class TargetNetwork(nn.Module):
    """
    目标网络（子网络）
    接收超网络生成的参数，进行分类预测
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TargetNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 定义网络结构（权重由超网络提供）
        self.fc1_weight_size = input_dim * hidden_dim
        self.fc1_bias_size = hidden_dim
        self.fc2_weight_size = hidden_dim * output_dim
        self.fc2_bias_size = output_dim
        
        self.total_params = (self.fc1_weight_size + self.fc1_bias_size + 
                            self.fc2_weight_size + self.fc2_bias_size)
    
    def forward(self, x, params):
        """
        前向传播
        x: 输入数据 [batch_size, input_dim]
        params: 超网络生成的参数 [batch_size, total_params]
        """
        batch_size = x.size(0)
        
        # 解析参数
        idx = 0
        fc1_weight = params[:, idx:idx + self.fc1_weight_size].view(batch_size, self.hidden_dim, self.input_dim)
        idx += self.fc1_weight_size
        fc1_bias = params[:, idx:idx + self.fc1_bias_size].view(batch_size, self.hidden_dim)
        idx += self.fc1_bias_size
        fc2_weight = params[:, idx:idx + self.fc2_weight_size].view(batch_size, self.output_dim, self.hidden_dim)
        idx += self.fc2_weight_size
        fc2_bias = params[:, idx:idx + self.fc2_bias_size].view(batch_size, self.output_dim)
        
        # 批量矩阵乘法
        x = x.unsqueeze(2)  # [batch, input_dim, 1]
        h = torch.bmm(fc1_weight, x).squeeze(2) + fc1_bias  # [batch, hidden_dim]
        h = torch.relu(h)
        h = h.unsqueeze(2)  # [batch, hidden_dim, 1]
        out = torch.bmm(fc2_weight, h).squeeze(2) + fc2_bias  # [batch, output_dim]
        
        return out


class HyperNetwork(nn.Module):
    """
    超网络
    根据输入特征动态生成目标网络的所有参数
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(HyperNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)


class HyperNetFusion(nn.Module):
    """
    VAE-HyperNetFusion 主模型
    
    架构:
    1. 超网络接收输入特征
    2. 生成多个目标子网络的参数
    3. 多个子网络分别处理输入，输出取平均（集成）
    
    优势:
    - 动态参数生成，适应不同数据特征
    - 多网络集成，提高泛化能力
    - 减少可训练参数（只需训练超网络）
    """
    
    def __init__(self, input_dim, hypernet_hidden_dim=128, 
                 target_hidden_dim=32, num_classes=2, num_target_nets=10):
        super(HyperNetFusion, self).__init__()
        
        self.input_dim = input_dim
        self.num_target_nets = num_target_nets
        self.num_classes = num_classes
        
        # 创建目标网络模板（获取参数数量）
        self.target_template = TargetNetwork(input_dim, target_hidden_dim, num_classes)
        param_dim_per_target = self.target_template.total_params
        total_param_dim = param_dim_per_target * num_target_nets
        
        # 超网络
        self.hypernet = HyperNetwork(input_dim, hypernet_hidden_dim, total_param_dim)
        
        # 目标网络列表
        self.target_nets = nn.ModuleList([
            TargetNetwork(input_dim, target_hidden_dim, num_classes)
            for _ in range(num_target_nets)
        ])
        
        self.param_dim_per_target = param_dim_per_target
    
    def forward(self, x):
        """
        前向传播
        x: 输入特征 [batch_size, input_dim]
        """
        batch_size = x.size(0)
        
        # 超网络生成所有目标网络的参数
        all_params = self.hypernet(x)  # [batch_size, total_param_dim]
        
        # 分配参数给各个目标网络并计算输出
        outputs = []
        for i, target_net in enumerate(self.target_nets):
            start_idx = i * self.param_dim_per_target
            end_idx = start_idx + self.param_dim_per_target
            params = all_params[:, start_idx:end_idx]
            out = target_net(x, params)
            outputs.append(out)
        
        # 集成：取平均
        ensemble_output = torch.stack(outputs, dim=0).mean(dim=0)
        
        return ensemble_output
    
    def predict_proba(self, x):
        """返回预测概率"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            proba = torch.softmax(logits, dim=1)
        return proba


if __name__ == '__main__':
    # 测试代码
    batch_size = 16
    input_dim = 4
    num_classes = 2
    
    model = HyperNetFusion(input_dim=input_dim, num_classes=num_classes, num_target_nets=5)
    x = torch.randn(batch_size, input_dim)
    output = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"超网络参数量: {sum(p.numel() for p in model.hypernet.parameters())}")
