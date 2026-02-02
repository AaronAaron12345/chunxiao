# models/hypernet.py
import torch
import torch.nn as nn


class HyperNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, param_dim):
        super(HyperNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, param_dim)  # Output parameters: weights and biases
        )

    def forward(self, x):
        params = self.network(x)
        return params
