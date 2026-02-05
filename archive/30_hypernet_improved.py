#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
30_hypernet_improved.py - 改进版HyperNet
=========================================
针对不稳定问题的改进：
1. 权重生成正则化 - 限制超网络输出范围
2. 特征Bagging - 每个子网络随机选择特征子集（类似RF）
3. Prototype-based - 基于类别原型的稳定学习
4. 多次运行投票 - 减少随机性
5. 温度缩放 - 稳定softmax输出

目标：VAE+HyperNet达到80%

运行: python 30_hypernet_improved.py
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
import torch.nn.functional as F
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


class VAE(nn.Module):
    def __init__(self, input_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(16, 8)
        self.fc_var = nn.Linear(16, 8)
        self.decoder = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, input_dim),
        )
    
    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = self.fc_mu(h), self.fc_var(h)
        z = mu + torch.exp(0.5 * log_var) * torch.randn_like(log_var)
        return self.decoder(z), mu, log_var


class StableHyperNet(nn.Module):
    """
    稳定的超网络设计
    
    改进点：
    1. 基于类别原型生成权重（而不是随机上下文）
    2. 权重生成使用tanh限制范围
    3. 每个子网络使用特征子集（Feature Bagging）
    4. 残差连接增加稳定性
    """
    def __init__(self, input_dim, n_classes, n_subnets=30, hidden_dim=16):
        super(StableHyperNet, self).__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.n_subnets = n_subnets
        self.hidden_dim = hidden_dim
        
        # 特征子集大小（类似RF的max_features）
        self.feature_subset_size = max(2, int(np.sqrt(input_dim)))
        
        # 为每个子网络随机选择特征子集
        self.feature_masks = nn.Parameter(
            torch.zeros(n_subnets, input_dim),
            requires_grad=False
        )
        for i in range(n_subnets):
            idx = np.random.choice(input_dim, self.feature_subset_size, replace=False)
            self.feature_masks.data[i, idx] = 1.0
        
        # 类别原型（可学习）
        self.class_prototypes = nn.Parameter(torch.randn(n_classes, input_dim) * 0.1)
        
        # 超网络：基于原型差异生成权重
        # 输入：类别原型的差异向量
        proto_diff_dim = input_dim * n_classes
        
        # 子网络参数大小
        self.w1_size = self.feature_subset_size * hidden_dim
        self.b1_size = hidden_dim
        self.w2_size = hidden_dim * n_classes
        self.b2_size = n_classes
        total_params = self.w1_size + self.b1_size + self.w2_size + self.b2_size
        
        # 超网络生成器 - 输出使用tanh限制范围
        self.hypernet = nn.Sequential(
            nn.Linear(proto_diff_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_subnets * total_params),
            nn.Tanh(),  # 限制权重范围在[-1, 1]
        )
        
        # 权重缩放因子（可学习）
        self.weight_scale = nn.Parameter(torch.ones(1) * 0.5)
        
        # 子网络投票权重
        self.vote_weights = nn.Parameter(torch.ones(n_subnets) / n_subnets)
        
        # 温度参数
        self.temperature = nn.Parameter(torch.ones(1))
    
    def compute_prototypes(self, X, y):
        """根据训练数据更新类别原型"""
        with torch.no_grad():
            for c in range(self.n_classes):
                mask = (y == c)
                if mask.sum() > 0:
                    self.class_prototypes.data[c] = X[mask].mean(dim=0)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # 基于原型生成权重
        proto_flat = self.class_prototypes.flatten().unsqueeze(0)
        all_params = self.hypernet(proto_flat)  # (1, n_subnets * total_params)
        all_params = all_params * self.weight_scale  # 缩放
        all_params = all_params.view(self.n_subnets, -1)
        
        all_logits = []
        
        for i in range(self.n_subnets):
            # 获取特征子集
            mask = self.feature_masks[i]
            x_subset = x * mask.unsqueeze(0)  # 特征选择
            x_subset = x_subset[:, mask.bool()]  # 压缩到子集维度
            
            # 提取子网络参数
            params = all_params[i]
            idx = 0
            
            W1 = params[idx:idx+self.w1_size].view(self.hidden_dim, self.feature_subset_size)
            idx += self.w1_size
            b1 = params[idx:idx+self.b1_size]
            idx += self.b1_size
            W2 = params[idx:idx+self.w2_size].view(self.n_classes, self.hidden_dim)
            idx += self.w2_size
            b2 = params[idx:idx+self.b2_size]
            
            # 前向传播（带残差）
            h = F.relu(F.linear(x_subset, W1, b1))
            h = F.dropout(h, p=0.2, training=self.training)
            logits = F.linear(h, W2, b2)
            
            all_logits.append(logits)
        
        # 加权投票
        all_logits = torch.stack(all_logits, dim=0)  # (n_subnets, batch, n_classes)
        weights = F.softmax(self.vote_weights, dim=0).view(-1, 1, 1)
        ensemble_logits = (all_logits * weights).sum(dim=0)
        
        # 温度缩放
        return ensemble_logits / self.temperature.clamp(min=0.1)


def vae_augment(vae, X_train, y_train, aug_factor=100, device='cuda'):
    """大量VAE数据增强"""
    vae.eval()
    X_t = torch.FloatTensor(X_train).to(device)
    
    X_aug = [X_train]
    y_aug = [y_train]
    
    with torch.no_grad():
        for cls in np.unique(y_train):
            mask = (y_train == cls)
            X_cls = X_t[mask]
            
            recon, mu, log_var = vae(X_cls)
            recon = recon.cpu().numpy()
            X_cls_np = X_cls.cpu().numpy()
            
            # 大量插值
            for alpha in np.linspace(0.05, 0.95, aug_factor // 10):
                X_aug.append(alpha * X_cls_np + (1 - alpha) * recon)
                y_aug.append(np.full(mask.sum(), cls))
            
            # 潜在空间采样
            for scale in [0.2, 0.3, 0.4]:
                for _ in range(aug_factor // 30):
                    z = mu + torch.exp(0.5 * log_var) * torch.randn_like(log_var) * scale
                    X_aug.append(vae.decoder(z).cpu().numpy())
                    y_aug.append(np.full(mask.sum(), cls))
            
            # 噪声增强
            for noise in [0.03, 0.05, 0.08]:
                X_noisy = X_cls_np + np.random.randn(*X_cls_np.shape) * noise
                X_aug.append(X_noisy)
                y_aug.append(np.full(mask.sum(), cls))
    
    return np.vstack(X_aug), np.hstack(y_aug)


def train_stable_hypernet(model, X, y, epochs=300, lr=0.005, device='cuda'):
    """稳定训练"""
    model.train()
    
    X_t = torch.FloatTensor(X).to(device)
    y_t = torch.LongTensor(y).to(device)
    
    # 更新原型
    model.compute_prototypes(X_t, y_t)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
    
    # 学习率预热 + 余弦退火
    warmup_epochs = 30
    
    for epoch in range(epochs):
        # 学习率预热
        if epoch < warmup_epochs:
            for pg in optimizer.param_groups:
                pg['lr'] = lr * (epoch + 1) / warmup_epochs
        else:
            # 余弦退火
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            for pg in optimizer.param_groups:
                pg['lr'] = lr * 0.5 * (1 + np.cos(np.pi * progress))
        
        optimizer.zero_grad()
        
        # Mini-batch with balanced sampling
        batch_size = min(128, len(X_t))
        idx = torch.randperm(len(X_t))[:batch_size]
        
        logits = model(X_t[idx])
        
        # Label smoothing + focal loss inspired weighting
        loss = F.cross_entropy(logits, y_t[idx], label_smoothing=0.15)
        
        # 正则化：鼓励子网络多样性
        if hasattr(model, 'vote_weights'):
            entropy = -(F.softmax(model.vote_weights, dim=0) * 
                       F.log_softmax(model.vote_weights + 1e-8, dim=0)).sum()
            loss = loss - 0.01 * entropy  # 鼓励均匀投票
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        # 定期更新原型
        if epoch % 50 == 0:
            with torch.no_grad():
                model.compute_prototypes(X_t, y_t)


def evaluate_with_multiple_runs(model_class, X_train, y_train, X_test, n_runs=5, device='cuda'):
    """多次运行投票 - 减少随机性"""
    all_preds = []
    
    for run in range(n_runs):
        torch.manual_seed(run * 42)
        np.random.seed(run * 42)
        
        # 训练VAE
        vae = VAE(X_train.shape[1]).to(device)
        optimizer = optim.Adam(vae.parameters(), lr=0.01)
        X_t = torch.FloatTensor(X_train).to(device)
        
        for _ in range(80):
            optimizer.zero_grad()
            recon, mu, log_var = vae(X_t)
            loss = nn.MSELoss()(recon, X_t) + 0.01 * (-0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp()))
            loss.backward()
            optimizer.step()
        
        # 数据增强
        X_aug, y_aug = vae_augment(vae, X_train, y_train, aug_factor=100, device=device)
        
        # 训练HyperNet
        model = model_class(X_train.shape[1], len(np.unique(y_train)), n_subnets=30).to(device)
        train_stable_hypernet(model, X_aug, y_aug, epochs=300, device=device)
        
        # 预测
        model.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test).to(device)
            logits = model(X_test_t)
            pred = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(pred)
    
    # 多数投票
    all_preds = np.array(all_preds)
    final_pred = []
    for i in range(len(X_test)):
        votes = all_preds[:, i]
        final_pred.append(np.bincount(votes.astype(int)).argmax())
    
    return np.array(final_pred)


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR / f'30_hypernet_improved_{timestamp}.log'
    
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
    logger.info("30_hypernet_improved.py - 改进版稳定HyperNet")
    logger.info("=" * 70)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"设备: {device}")
    
    data_path = SCRIPT_DIR / 'data' / 'Data_for_Jinming.csv'
    df = pd.read_csv(data_path)
    X = df[['LAA', 'Glutamate', 'Choline', 'Sarcosine']].values.astype(np.float32)
    y = LabelEncoder().fit_transform(df['Group'].values)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    logger.info(f"数据: {len(X)} 样本, {X.shape[1]} 特征")
    
    results = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # ==================== RF ====================
    logger.info("\n[1/4] RF 基线...")
    rf_preds, rf_trues = [], []
    for train_idx, test_idx in skf.split(X_scaled, y):
        rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
        rf.fit(X_scaled[train_idx], y[train_idx])
        rf_preds.extend(rf.predict(X_scaled[test_idx]).tolist())
        rf_trues.extend(y[test_idx].tolist())
    results['RF'] = accuracy_score(rf_trues, rf_preds) * 100
    logger.info(f"   RF: {results['RF']:.2f}%")
    
    # ==================== VAE+RF ====================
    logger.info("\n[2/4] VAE + RF...")
    vae_rf_preds, vae_rf_trues = [], []
    for train_idx, test_idx in skf.split(X_scaled, y):
        X_train, y_train = X_scaled[train_idx], y[train_idx]
        X_test, y_test = X_scaled[test_idx], y[test_idx]
        
        vae = VAE(X_train.shape[1]).to(device)
        optimizer = optim.Adam(vae.parameters(), lr=0.01)
        X_t = torch.FloatTensor(X_train).to(device)
        for _ in range(80):
            optimizer.zero_grad()
            recon, mu, log_var = vae(X_t)
            loss = nn.MSELoss()(recon, X_t) + 0.01 * (-0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp()))
            loss.backward()
            optimizer.step()
        
        X_aug, y_aug = vae_augment(vae, X_train, y_train, aug_factor=100, device=device)
        
        rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
        rf.fit(X_aug, y_aug)
        vae_rf_preds.extend(rf.predict(X_test).tolist())
        vae_rf_trues.extend(y_test.tolist())
    
    results['VAE+RF'] = accuracy_score(vae_rf_trues, vae_rf_preds) * 100
    logger.info(f"   VAE+RF: {results['VAE+RF']:.2f}%")
    
    # ==================== 改进HyperNet（单次运行） ====================
    logger.info("\n[3/4] VAE + StableHyperNet (单次)...")
    hypernet_preds, hypernet_trues = [], []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
        X_train, y_train = X_scaled[train_idx], y[train_idx]
        X_test, y_test = X_scaled[test_idx], y[test_idx]
        
        torch.manual_seed(42)
        np.random.seed(42)
        
        # VAE
        vae = VAE(X_train.shape[1]).to(device)
        optimizer = optim.Adam(vae.parameters(), lr=0.01)
        X_t = torch.FloatTensor(X_train).to(device)
        for _ in range(80):
            optimizer.zero_grad()
            recon, mu, log_var = vae(X_t)
            loss = nn.MSELoss()(recon, X_t) + 0.01 * (-0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp()))
            loss.backward()
            optimizer.step()
        
        X_aug, y_aug = vae_augment(vae, X_train, y_train, aug_factor=100, device=device)
        
        # StableHyperNet
        model = StableHyperNet(X_train.shape[1], len(np.unique(y_train)), n_subnets=30).to(device)
        train_stable_hypernet(model, X_aug, y_aug, epochs=300, device=device)
        
        model.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test).to(device)
            pred = model(X_test_t).argmax(dim=1).cpu().numpy()
        
        hypernet_preds.extend(pred.tolist())
        hypernet_trues.extend(y_test.tolist())
        acc = accuracy_score(y_test, pred) * 100
        logger.info(f"   Fold {fold_idx+1}/5: {acc:.2f}%")
    
    results['VAE+StableHyperNet(单次)'] = accuracy_score(hypernet_trues, hypernet_preds) * 100
    logger.info(f"   VAE+StableHyperNet(单次): {results['VAE+StableHyperNet(单次)']:.2f}%")
    
    # ==================== 改进HyperNet（多次运行投票） ====================
    logger.info("\n[4/4] VAE + StableHyperNet (5次投票)...")
    vote_preds, vote_trues = [], []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
        X_train, y_train = X_scaled[train_idx], y[train_idx]
        X_test, y_test = X_scaled[test_idx], y[test_idx]
        
        pred = evaluate_with_multiple_runs(
            StableHyperNet, X_train, y_train, X_test, 
            n_runs=5, device=device
        )
        
        vote_preds.extend(pred.tolist())
        vote_trues.extend(y_test.tolist())
        acc = accuracy_score(y_test, pred) * 100
        logger.info(f"   Fold {fold_idx+1}/5: {acc:.2f}%")
    
    results['VAE+StableHyperNet(5次投票)'] = accuracy_score(vote_trues, vote_preds) * 100
    logger.info(f"   VAE+StableHyperNet(5次投票): {results['VAE+StableHyperNet(5次投票)']:.2f}%")
    
    # ==================== 汇总 ====================
    logger.info("\n" + "=" * 70)
    logger.info("[结果对比]")
    logger.info("=" * 70)
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        marker = "🏆" if acc == max(results.values()) else "  "
        logger.info(f"{marker} {name:30s}: {acc:.2f}%")
    logger.info("=" * 70)
    
    result_file = OUTPUT_DIR / f'30_hypernet_improved_{timestamp}.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"保存: {result_file}")


if __name__ == '__main__':
    main()
