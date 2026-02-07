#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
29_vae_tabnet_style.py - VAE + TabNet风格注意力网络
==================================================
TabNet核心思想：
1. 稀疏特征选择（注意力机制）
2. 分步决策（类似树的分层结构）
3. 共享和独立表示

目标：让神经网络达到80%

运行: python 29_vae_tabnet_style.py
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


class FeatureAttention(nn.Module):
    """特征注意力模块 - 稀疏特征选择"""
    def __init__(self, input_dim, hidden_dim=16):
        super(FeatureAttention, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
    
    def forward(self, x, prior_scales=None):
        # 计算注意力分数
        scores = self.fc(x)
        if prior_scales is not None:
            scores = scores * prior_scales
        # Sparsemax近似（使用softmax + 温度）
        attention = F.softmax(scores / 0.5, dim=-1)
        return attention


class DecisionStep(nn.Module):
    """单个决策步骤"""
    def __init__(self, input_dim, output_dim, shared_fc, n_independent=2):
        super(DecisionStep, self).__init__()
        self.shared_fc = shared_fc
        self.independent = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )
        self.attention = FeatureAttention(input_dim)
    
    def forward(self, x, prior_scales):
        # 注意力加权特征
        attention = self.attention(x, prior_scales)
        masked_x = x * attention
        
        # 共享 + 独立处理
        shared_out = self.shared_fc(masked_x)
        independent_out = self.independent(masked_x)
        
        # 组合
        output = shared_out + independent_out
        
        # 更新先验（减少已选择特征的重要性）
        new_prior = prior_scales * (1 - attention)
        
        return output, attention, new_prior


class TabNetStyleClassifier(nn.Module):
    """TabNet风格的分类器"""
    def __init__(self, input_dim, n_classes, n_steps=3, hidden_dim=32):
        super(TabNetStyleClassifier, self).__init__()
        self.input_dim = input_dim
        self.n_steps = n_steps
        
        # 共享的特征变换
        self.shared_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        
        # 多个决策步骤
        self.steps = nn.ModuleList([
            DecisionStep(input_dim, hidden_dim, self.shared_fc)
            for _ in range(n_steps)
        ])
        
        # 最终分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * n_steps, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_classes),
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # 初始先验（所有特征同等重要）
        prior_scales = torch.ones(batch_size, self.input_dim, device=x.device)
        
        step_outputs = []
        total_attention = 0
        
        for step in self.steps:
            step_out, attention, prior_scales = step(x, prior_scales)
            step_outputs.append(step_out)
            total_attention = total_attention + attention
        
        # 聚合所有步骤的输出
        aggregated = torch.cat(step_outputs, dim=-1)
        
        return self.classifier(aggregated)


class EnsembleTabNet(nn.Module):
    """TabNet集成 - 类似RF的多模型"""
    def __init__(self, input_dim, n_classes, n_models=10):
        super(EnsembleTabNet, self).__init__()
        self.models = nn.ModuleList([
            TabNetStyleClassifier(input_dim, n_classes, n_steps=3, hidden_dim=32)
            for _ in range(n_models)
        ])
    
    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return torch.stack(outputs, dim=0).mean(dim=0)


def vae_augment(vae, X_train, y_train, aug_factor=50, device='cuda'):
    """VAE数据增强"""
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
            
            # 插值
            for alpha in np.linspace(0.1, 0.9, aug_factor // 5):
                X_aug.append(alpha * X_cls_np + (1 - alpha) * recon)
                y_aug.append(np.full(mask.sum(), cls))
            
            # 采样
            for _ in range(aug_factor // 5):
                z = mu + torch.exp(0.5 * log_var) * torch.randn_like(log_var) * 0.3
                X_aug.append(vae.decoder(z).cpu().numpy())
                y_aug.append(np.full(mask.sum(), cls))
            
            # 噪声
            for noise in [0.05, 0.1]:
                X_noisy = X_cls_np + np.random.randn(*X_cls_np.shape) * noise
                X_aug.append(X_noisy)
                y_aug.append(np.full(mask.sum(), cls))
    
    return np.vstack(X_aug), np.hstack(y_aug)


def train_model(model, X, y, epochs=200, lr=0.01, device='cuda'):
    """训练模型"""
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    X_t = torch.FloatTensor(X).to(device)
    y_t = torch.LongTensor(y).to(device)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Mini-batch
        idx = torch.randperm(len(X_t))[:min(64, len(X_t))]
        logits = model(X_t[idx])
        loss = F.cross_entropy(logits, y_t[idx], label_smoothing=0.1)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR / f'29_tabnet_{timestamp}.log'
    
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
    logger.info("29_vae_tabnet_style.py - VAE + TabNet风格网络")
    logger.info("=" * 70)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"设备: {device}")
    
    data_path = SCRIPT_DIR / 'data' / 'Data_for_Jinming.csv'
    df = pd.read_csv(data_path)
    X = df[['LAA', 'Glutamate', 'Choline', 'Sarcosine']].values.astype(np.float32)
    y = LabelEncoder().fit_transform(df['Group'].values)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    logger.info(f"数据: {len(X)} 样本")
    
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
        for _ in range(60):
            optimizer.zero_grad()
            recon, mu, log_var = vae(X_t)
            loss = nn.MSELoss()(recon, X_t) + 0.01 * (-0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp()))
            loss.backward()
            optimizer.step()
        
        X_aug, y_aug = vae_augment(vae, X_train, y_train, aug_factor=50, device=device)
        
        rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
        rf.fit(X_aug, y_aug)
        vae_rf_preds.extend(rf.predict(X_test).tolist())
        vae_rf_trues.extend(y_test.tolist())
    
    results['VAE+RF'] = accuracy_score(vae_rf_trues, vae_rf_preds) * 100
    logger.info(f"   VAE+RF: {results['VAE+RF']:.2f}%")
    
    # ==================== TabNet（无VAE） ====================
    logger.info("\n[3/4] TabNet (无VAE)...")
    tab_preds, tab_trues = [], []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
        X_train, y_train = X_scaled[train_idx], y[train_idx]
        X_test, y_test = X_scaled[test_idx], y[test_idx]
        
        model = EnsembleTabNet(X_train.shape[1], len(np.unique(y_train)), n_models=10).to(device)
        train_model(model, X_train, y_train, epochs=200, lr=0.01, device=device)
        
        model.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test).to(device)
            pred = model(X_test_t).argmax(dim=1).cpu().numpy()
        
        tab_preds.extend(pred.tolist())
        tab_trues.extend(y_test.tolist())
        logger.info(f"   Fold {fold_idx+1}/5: {accuracy_score(y_test, pred)*100:.2f}%")
    
    results['TabNet'] = accuracy_score(tab_trues, tab_preds) * 100
    logger.info(f"   TabNet: {results['TabNet']:.2f}%")
    
    # ==================== VAE + TabNet ====================
    logger.info("\n[4/4] VAE + TabNet...")
    vae_tab_preds, vae_tab_trues = [], []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
        X_train, y_train = X_scaled[train_idx], y[train_idx]
        X_test, y_test = X_scaled[test_idx], y[test_idx]
        
        # VAE
        vae = VAE(X_train.shape[1]).to(device)
        optimizer = optim.Adam(vae.parameters(), lr=0.01)
        X_t = torch.FloatTensor(X_train).to(device)
        for _ in range(60):
            optimizer.zero_grad()
            recon, mu, log_var = vae(X_t)
            loss = nn.MSELoss()(recon, X_t) + 0.01 * (-0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp()))
            loss.backward()
            optimizer.step()
        
        X_aug, y_aug = vae_augment(vae, X_train, y_train, aug_factor=100, device=device)
        
        # TabNet
        model = EnsembleTabNet(X_train.shape[1], len(np.unique(y_train)), n_models=15).to(device)
        train_model(model, X_aug, y_aug, epochs=300, lr=0.01, device=device)
        
        model.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test).to(device)
            pred = model(X_test_t).argmax(dim=1).cpu().numpy()
        
        vae_tab_preds.extend(pred.tolist())
        vae_tab_trues.extend(y_test.tolist())
        logger.info(f"   Fold {fold_idx+1}/5: {accuracy_score(y_test, pred)*100:.2f}%")
    
    results['VAE+TabNet'] = accuracy_score(vae_tab_trues, vae_tab_preds) * 100
    logger.info(f"   VAE+TabNet: {results['VAE+TabNet']:.2f}%")
    
    # ==================== 汇总 ====================
    logger.info("\n" + "=" * 70)
    logger.info("[结果对比]")
    logger.info("=" * 70)
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        marker = "🏆" if acc == max(results.values()) else "  "
        logger.info(f"{marker} {name:20s}: {acc:.2f}%")
    logger.info("=" * 70)
    
    result_file = OUTPUT_DIR / f'29_tabnet_{timestamp}.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"保存: {result_file}")


if __name__ == '__main__':
    main()
