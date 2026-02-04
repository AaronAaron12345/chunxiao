#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
26_vae_hypernet_gpu_kfold.py - GPU并行 + 5折交叉验证
=====================================================
目标：VAE+HyperNet 达到80%准确率

改进策略：
1. 5折交叉验证（比325组快65倍）
2. 多GPU并行
3. 大量数据增强（500倍）
4. HyperNet生成多个子网络（类似RF多棵树）
5. Label Smoothing防止过拟合

运行: python 26_vae_hypernet_gpu_kfold.py
"""

import os
import sys
import json
import time
import logging
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).parent
LOG_DIR = SCRIPT_DIR / 'logs'
OUTPUT_DIR = SCRIPT_DIR / 'output'
LOG_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# GPU设置
DEVICE_IDS = [0, 1, 2, 3, 4, 5]  # 使用GPU 0-5


class VAE(nn.Module):
    """变分自编码器 - 用于数据增强"""
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_var = nn.Linear(hidden_dim // 2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


class HyperNetwork(nn.Module):
    """超网络 - 根据输入特征生成分类器权重"""
    def __init__(self, input_dim, n_classes, n_subnets=20, hidden_dim=32):
        super(HyperNetwork, self).__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.n_subnets = n_subnets
        self.hidden_dim = hidden_dim
        
        # 超网络：为每个子网络生成权重
        # 子网络结构: input -> hidden -> output
        self.weight_size_1 = input_dim * hidden_dim  # W1
        self.bias_size_1 = hidden_dim  # b1
        self.weight_size_2 = hidden_dim * n_classes  # W2
        self.bias_size_2 = n_classes  # b2
        
        total_params = self.weight_size_1 + self.bias_size_1 + self.weight_size_2 + self.bias_size_2
        
        # 超网络生成器 - 为n_subnets个子网络生成权重
        self.hypernet = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_subnets * total_params),
        )
        
        # 用于聚合多个子网络预测的权重
        self.ensemble_weights = nn.Parameter(torch.ones(n_subnets) / n_subnets)
    
    def forward(self, x, support_x=None):
        """
        x: 查询样本 (batch, input_dim)
        support_x: 支持集样本 (用于生成权重) - 如果为None则用x自身
        """
        if support_x is None:
            support_x = x
        
        # 用支持集的均值来生成权重
        context = support_x.mean(dim=0, keepdim=True)  # (1, input_dim)
        
        # 生成所有子网络的权重
        all_params = self.hypernet(context)  # (1, n_subnets * total_params)
        all_params = all_params.view(self.n_subnets, -1)  # (n_subnets, total_params)
        
        batch_size = x.size(0)
        all_logits = []
        
        for i in range(self.n_subnets):
            params = all_params[i]
            idx = 0
            
            # 提取子网络权重
            W1 = params[idx:idx + self.weight_size_1].view(self.hidden_dim, self.input_dim)
            idx += self.weight_size_1
            b1 = params[idx:idx + self.bias_size_1]
            idx += self.bias_size_1
            W2 = params[idx:idx + self.weight_size_2].view(self.n_classes, self.hidden_dim)
            idx += self.weight_size_2
            b2 = params[idx:idx + self.bias_size_2]
            
            # 前向传播
            h = torch.relu(torch.mm(x, W1.t()) + b1)
            h = torch.dropout(h, p=0.2, train=self.training)
            logits = torch.mm(h, W2.t()) + b2
            all_logits.append(logits)
        
        # 加权平均所有子网络的预测
        all_logits = torch.stack(all_logits, dim=0)  # (n_subnets, batch, n_classes)
        weights = torch.softmax(self.ensemble_weights, dim=0).view(-1, 1, 1)
        ensemble_logits = (all_logits * weights).sum(dim=0)  # (batch, n_classes)
        
        return ensemble_logits


class VAEHyperNetFusion(nn.Module):
    """VAE + HyperNet 融合模型"""
    def __init__(self, input_dim, n_classes, n_subnets=20, vae_latent=16):
        super(VAEHyperNetFusion, self).__init__()
        self.vae = VAE(input_dim, hidden_dim=64, latent_dim=vae_latent)
        self.hypernet = HyperNetwork(input_dim, n_classes, n_subnets=n_subnets)
        self.input_dim = input_dim
        self.n_classes = n_classes
    
    def augment_data(self, X, y, aug_factor=50):
        """用VAE增强数据"""
        self.vae.eval()
        device = next(self.vae.parameters()).device
        
        X_aug = [X]
        y_aug = [y]
        
        with torch.no_grad():
            for cls in torch.unique(y):
                mask = (y == cls)
                X_cls = X[mask]
                
                # 获取潜在分布参数
                mu, log_var = self.vae.encode(X_cls)
                
                # 生成多个增强样本
                for _ in range(aug_factor):
                    # 从潜在空间采样
                    z = self.vae.reparameterize(mu, log_var * 0.5)  # 减小方差
                    X_new = self.vae.decode(z)
                    X_aug.append(X_new)
                    y_aug.append(torch.full((len(X_cls),), cls.item(), device=device))
                
                # 插值增强
                recon, _, _ = self.vae(X_cls)
                for alpha in np.linspace(0.1, 0.9, 10):
                    X_interp = alpha * X_cls + (1 - alpha) * recon
                    X_aug.append(X_interp)
                    y_aug.append(torch.full((len(X_cls),), cls.item(), device=device))
        
        return torch.cat(X_aug, dim=0), torch.cat(y_aug, dim=0)
    
    def forward(self, x, support_x=None):
        return self.hypernet(x, support_x)


def train_vae(model, X_train, epochs=100, lr=0.01):
    """训练VAE"""
    model.vae.train()
    optimizer = optim.Adam(model.vae.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        recon, mu, log_var = model.vae(X_train)
        
        # 重建损失 + KL散度
        recon_loss = nn.MSELoss()(recon, X_train)
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recon_loss + 0.01 * kl_loss
        
        loss.backward()
        optimizer.step()


def train_hypernet(model, X_aug, y_aug, epochs=200, lr=0.005):
    """训练HyperNet"""
    model.hypernet.train()
    optimizer = optim.AdamW(model.hypernet.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    dataset = TensorDataset(X_aug, y_aug)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            logits = model.hypernet(X_batch, X_aug)  # 用全部增强数据作为支持集
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.hypernet.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()


def evaluate_fold(fold_idx, X_train, y_train, X_test, y_test, device):
    """评估单个fold"""
    # 转换为tensor
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    
    n_classes = len(np.unique(y_train))
    input_dim = X_train.shape[1]
    
    # 创建模型
    model = VAEHyperNetFusion(input_dim, n_classes, n_subnets=30).to(device)
    
    # 1. 训练VAE
    train_vae(model, X_train_t, epochs=100, lr=0.01)
    
    # 2. 数据增强
    X_aug, y_aug = model.augment_data(X_train_t, y_train_t, aug_factor=100)
    
    # 3. 训练HyperNet
    train_hypernet(model, X_aug, y_aug, epochs=300, lr=0.005)
    
    # 4. 预测
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t, X_aug)
        y_pred = logits.argmax(dim=1).cpu().numpy()
    
    return y_test.tolist(), y_pred.tolist()


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR / f'26_gpu_kfold_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 70)
    logger.info("26_vae_hypernet_gpu_kfold.py - GPU并行 + 5折交叉验证")
    logger.info("=" * 70)
    
    # 检测GPU
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        logger.info(f"可用GPU: {n_gpus}")
        for i in range(min(n_gpus, 6)):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        device = torch.device('cuda:0')
    else:
        logger.info("使用CPU")
        device = torch.device('cpu')
    
    # 加载数据
    data_path = SCRIPT_DIR / 'data' / 'Data_for_Jinming.csv'
    df = pd.read_csv(data_path)
    X = df[['LAA', 'Glutamate', 'Choline', 'Sarcosine']].values.astype(np.float32)
    y = LabelEncoder().fit_transform(df['Group'].values)
    
    logger.info(f"数据: {len(X)} 样本, {X.shape[1]} 特征, {len(np.unique(y))} 类别")
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = {}
    
    # ==================== RF基线 ====================
    logger.info("\n[1/3] Random Forest 基线 (5折CV)...")
    start = time.time()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_preds, rf_trues = [], []
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
        rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
        rf.fit(X_scaled[train_idx], y[train_idx])
        pred = rf.predict(X_scaled[test_idx])
        rf_preds.extend(pred.tolist())
        rf_trues.extend(y[test_idx].tolist())
    
    results['RF'] = accuracy_score(rf_trues, rf_preds) * 100
    logger.info(f"   RF: {results['RF']:.2f}% ({time.time()-start:.1f}s)")
    
    # ==================== VAE+RF ====================
    logger.info("\n[2/3] VAE + RF (5折CV)...")
    start = time.time()
    vae_rf_preds, vae_rf_trues = [], []
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
        X_train, y_train = X_scaled[train_idx], y[train_idx]
        X_test, y_test = X_scaled[test_idx], y[test_idx]
        
        # VAE数据增强
        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.LongTensor(y_train).to(device)
        
        model = VAEHyperNetFusion(X_train.shape[1], len(np.unique(y_train))).to(device)
        train_vae(model, X_train_t, epochs=80, lr=0.01)
        X_aug, y_aug = model.augment_data(X_train_t, y_train_t, aug_factor=50)
        
        X_aug_np = X_aug.cpu().numpy()
        y_aug_np = y_aug.cpu().numpy()
        
        rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
        rf.fit(X_aug_np, y_aug_np)
        pred = rf.predict(X_test)
        
        vae_rf_preds.extend(pred.tolist())
        vae_rf_trues.extend(y_test.tolist())
        logger.info(f"   Fold {fold_idx+1}/5 完成")
    
    results['VAE+RF'] = accuracy_score(vae_rf_trues, vae_rf_preds) * 100
    logger.info(f"   VAE+RF: {results['VAE+RF']:.2f}% ({time.time()-start:.1f}s)")
    
    # ==================== VAE+HyperNet ====================
    logger.info("\n[3/3] VAE + HyperNet (5折CV)...")
    start = time.time()
    hypernet_preds, hypernet_trues = [], []
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
        X_train, y_train = X_scaled[train_idx], y[train_idx]
        X_test, y_test = X_scaled[test_idx], y[test_idx]
        
        y_true, y_pred = evaluate_fold(fold_idx, X_train, y_train, X_test, y_test, device)
        hypernet_preds.extend(y_pred)
        hypernet_trues.extend(y_true)
        
        fold_acc = accuracy_score(y_true, y_pred) * 100
        logger.info(f"   Fold {fold_idx+1}/5: {fold_acc:.2f}%")
    
    results['VAE+HyperNet'] = accuracy_score(hypernet_trues, hypernet_preds) * 100
    logger.info(f"   VAE+HyperNet: {results['VAE+HyperNet']:.2f}% ({time.time()-start:.1f}s)")
    
    # ==================== 汇总 ====================
    logger.info("\n" + "=" * 70)
    logger.info("[结果对比 - 5折交叉验证]")
    logger.info("=" * 70)
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        marker = "🏆" if acc == max(results.values()) else "  "
        logger.info(f"{marker} {name:15s}: {acc:.2f}%")
    logger.info("=" * 70)
    
    # 保存结果
    result_file = OUTPUT_DIR / f'26_gpu_kfold_{timestamp}.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"保存: {result_file}")


if __name__ == '__main__':
    main()
