# -*- coding: utf-8 -*-
"""
VAE Data Augmentation Module
============================
使用变分自编码器(VAE)进行数据增强，核心思想：
1. VAE学习数据的潜在分布
2. 在原始数据和VAE重构数据之间进行插值
3. 生成多样化的增强样本，解决小数据集问题
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class VAE(nn.Module):
    """变分自编码器 - PyTorch实现"""
    
    def __init__(self, input_dim, latent_dim=2, hidden_dim=128):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon_x, x, mu, logvar, kl_weight=1.0):
    """VAE损失函数 = 重构损失 + KL散度"""
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kl_loss


class VAEDataAugmentation:
    """
    VAE数据增强类
    
    核心流程：
    1. 对每个类别分别训练VAE
    2. 对原始样本进行编码-解码得到重构样本
    3. 在原始样本和重构样本之间线性插值生成新样本
    """
    
    def __init__(self, latent_dim=2, hidden_dim=128, epochs=100, 
                 learning_rate=0.001, kl_weight=0.5, num_interpolation_points=5):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.kl_weight = kl_weight
        self.num_interpolation_points = num_interpolation_points
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = MinMaxScaler()
    
    def _train_vae(self, X_class):
        """为单个类别训练VAE"""
        input_dim = X_class.shape[1]
        vae = VAE(input_dim, self.latent_dim, self.hidden_dim).to(self.device)
        optimizer = optim.Adam(vae.parameters(), lr=self.learning_rate)
        
        X_tensor = torch.FloatTensor(X_class).to(self.device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=min(32, len(X_class)), shuffle=True)
        
        vae.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in dataloader:
                x = batch[0]
                optimizer.zero_grad()
                recon_x, mu, logvar = vae(x)
                loss = vae_loss(recon_x, x, mu, logvar, self.kl_weight)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        return vae
    
    def _linear_interpolate(self, point_a, point_b, num_points):
        """在两点之间进行线性插值"""
        return np.linspace(point_a, point_b, num=num_points + 2)[1:-1]  # 排除端点
    
    def augment(self, X, y):
        """
        执行数据增强
        
        参数:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签向量 (n_samples,)
        
        返回:
            X_augmented: 增强后的特征矩阵
            y_augmented: 增强后的标签
        """
        # 数据标准化到[0,1]
        X_scaled = self.scaler.fit_transform(X)
        
        unique_classes = np.unique(y)
        augmented_data = []
        augmented_labels = []
        
        print(f"[VAE Augmentation] 开始数据增强...")
        print(f"  - 原始样本数: {len(X)}")
        print(f"  - 类别数: {len(unique_classes)}")
        print(f"  - 插值点数: {self.num_interpolation_points}")
        
        for cls in unique_classes:
            X_class = X_scaled[y == cls]
            print(f"  - 类别 {cls}: {len(X_class)} 个样本，训练VAE中...")
            
            # 训练该类别的VAE
            vae = self._train_vae(X_class)
            
            # 获取重构数据
            vae.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_class).to(self.device)
                recon_X, _, _ = vae(X_tensor)
                recon_X = recon_X.cpu().numpy()
            
            # 在原始数据和重构数据之间插值
            for orig, recon in zip(X_class, recon_X):
                interpolated = self._linear_interpolate(orig, recon, self.num_interpolation_points)
                augmented_data.extend(interpolated)
                augmented_labels.extend([cls] * len(interpolated))
        
        X_augmented = np.array(augmented_data)
        y_augmented = np.array(augmented_labels)
        
        # 逆标准化回原始尺度
        X_augmented = self.scaler.inverse_transform(X_augmented)
        
        print(f"  - 增强后样本数: {len(X_augmented)}")
        
        return X_augmented, y_augmented
    
    def save_augmented_data(self, X_aug, y_aug, filepath, feature_names=None):
        """保存增强后的数据到CSV"""
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(X_aug.shape[1])]
        
        df = pd.DataFrame(X_aug, columns=feature_names)
        df['Label'] = y_aug
        df.to_csv(filepath, index=False)
        print(f"  - 增强数据已保存到: {filepath}")


if __name__ == '__main__':
    # 测试代码
    np.random.seed(42)
    X_test = np.random.randn(50, 4) * 10 + 50
    y_test = np.array([0] * 25 + [1] * 25)
    
    augmenter = VAEDataAugmentation(num_interpolation_points=5)
    X_aug, y_aug = augmenter.augment(X_test, y_test)
    print(f"增强结果: {X_test.shape} -> {X_aug.shape}")
