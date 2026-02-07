#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
27_hypernet_rf_distillation.py - RF知识蒸馏到HyperNet
=====================================================
策略：
1. 先训练RF教师（VAE增强后达到88%）
2. 用RF的软标签训练HyperNet学生
3. 结合硬标签和软标签

这样HyperNet可以学习RF的决策边界！

运行: python 27_hypernet_rf_distillation.py
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


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_var = nn.Linear(hidden_dim // 2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
    
    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = self.fc_mu(h), self.fc_var(h)
        z = mu + torch.exp(0.5 * log_var) * torch.randn_like(log_var)
        return self.decoder(z), mu, log_var


class HyperNetStudent(nn.Module):
    """HyperNet学生模型 - 学习RF教师的决策"""
    def __init__(self, input_dim, n_classes, n_subnets=30, hidden_dim=32):
        super(HyperNetStudent, self).__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.n_subnets = n_subnets
        self.hidden_dim = hidden_dim
        
        # 子网络参数大小
        total_params = input_dim * hidden_dim + hidden_dim + hidden_dim * n_classes + n_classes
        
        # 超网络 - 根据数据分布生成子网络权重
        self.context_encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 64), nn.ReLU(),
        )
        
        self.weight_generator = nn.Sequential(
            nn.Linear(64, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, n_subnets * total_params),
        )
        
        self.ensemble_weights = nn.Parameter(torch.ones(n_subnets) / n_subnets)
        
        # 记录参数大小
        self.w1_size = input_dim * hidden_dim
        self.b1_size = hidden_dim
        self.w2_size = hidden_dim * n_classes
        self.b2_size = n_classes
    
    def forward(self, x, context=None):
        if context is None:
            context = x.mean(dim=0, keepdim=True)
        
        # 编码上下文
        ctx = self.context_encoder(context)
        
        # 生成所有子网络权重
        all_params = self.weight_generator(ctx).view(self.n_subnets, -1)
        
        all_logits = []
        for i in range(self.n_subnets):
            params = all_params[i]
            idx = 0
            W1 = params[idx:idx+self.w1_size].view(self.hidden_dim, self.input_dim)
            idx += self.w1_size
            b1 = params[idx:idx+self.b1_size]
            idx += self.b1_size
            W2 = params[idx:idx+self.w2_size].view(self.n_classes, self.hidden_dim)
            idx += self.w2_size
            b2 = params[idx:idx+self.b2_size]
            
            h = F.relu(F.linear(x, W1, b1))
            h = F.dropout(h, p=0.1, training=self.training)
            logits = F.linear(h, W2, b2)
            all_logits.append(logits)
        
        all_logits = torch.stack(all_logits, dim=0)
        weights = F.softmax(self.ensemble_weights, dim=0).view(-1, 1, 1)
        return (all_logits * weights).sum(dim=0)


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
                new = vae.decoder(z).cpu().numpy()
                X_aug.append(new)
                y_aug.append(np.full(mask.sum(), cls))
    
    return np.vstack(X_aug), np.hstack(y_aug)


def train_with_distillation(student, X_aug, y_aug, rf_probs, epochs=300, lr=0.005, device='cuda'):
    """用RF软标签训练HyperNet"""
    student.train()
    optimizer = optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    X_t = torch.FloatTensor(X_aug).to(device)
    y_t = torch.LongTensor(y_aug).to(device)
    rf_probs_t = torch.FloatTensor(rf_probs).to(device)
    
    context = X_t.mean(dim=0, keepdim=True)
    
    dataset = TensorDataset(X_t, y_t, rf_probs_t)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    temperature = 3.0  # 蒸馏温度
    alpha = 0.7  # 软标签权重
    
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch, rf_batch in loader:
            optimizer.zero_grad()
            
            logits = student(X_batch, context)
            
            # 硬标签损失
            hard_loss = F.cross_entropy(logits, y_batch, label_smoothing=0.1)
            
            # 软标签损失（KL散度）
            soft_logits = F.log_softmax(logits / temperature, dim=1)
            soft_targets = F.softmax(rf_batch / temperature, dim=1)
            soft_loss = F.kl_div(soft_logits, soft_targets, reduction='batchmean') * (temperature ** 2)
            
            # 组合损失
            loss = alpha * soft_loss + (1 - alpha) * hard_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
    
    return student


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR / f'27_distillation_{timestamp}.log'
    
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
    logger.info("27_hypernet_rf_distillation.py - RF知识蒸馏到HyperNet")
    logger.info("=" * 70)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"设备: {device}")
    
    # 加载数据
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
        
        # 增强
        X_aug, y_aug = vae_augment(vae, X_train, y_train, aug_factor=50, device=device)
        
        rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
        rf.fit(X_aug, y_aug)
        vae_rf_preds.extend(rf.predict(X_test).tolist())
        vae_rf_trues.extend(y_test.tolist())
    
    results['VAE+RF'] = accuracy_score(vae_rf_trues, vae_rf_preds) * 100
    logger.info(f"   VAE+RF: {results['VAE+RF']:.2f}%")
    
    # ==================== VAE+HyperNet (无蒸馏) ====================
    logger.info("\n[3/4] VAE + HyperNet (无蒸馏)...")
    hypernet_preds, hypernet_trues = [], []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
        X_train, y_train = X_scaled[train_idx], y[train_idx]
        X_test, y_test = X_scaled[test_idx], y[test_idx]
        
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
        
        # 增强
        X_aug, y_aug = vae_augment(vae, X_train, y_train, aug_factor=50, device=device)
        
        # 训练HyperNet
        student = HyperNetStudent(X_train.shape[1], len(np.unique(y_train))).to(device)
        student.train()
        optimizer = optim.AdamW(student.parameters(), lr=0.005, weight_decay=0.01)
        
        X_aug_t = torch.FloatTensor(X_aug).to(device)
        y_aug_t = torch.LongTensor(y_aug).to(device)
        context = X_aug_t.mean(dim=0, keepdim=True)
        
        for _ in range(200):
            optimizer.zero_grad()
            logits = student(X_aug_t, context)
            loss = F.cross_entropy(logits, y_aug_t, label_smoothing=0.1)
            loss.backward()
            optimizer.step()
        
        # 预测
        student.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test).to(device)
            pred = student(X_test_t, context).argmax(dim=1).cpu().numpy()
        
        hypernet_preds.extend(pred.tolist())
        hypernet_trues.extend(y_test.tolist())
        logger.info(f"   Fold {fold_idx+1}/5: {accuracy_score(y_test, pred)*100:.2f}%")
    
    results['VAE+HyperNet'] = accuracy_score(hypernet_trues, hypernet_preds) * 100
    logger.info(f"   VAE+HyperNet: {results['VAE+HyperNet']:.2f}%")
    
    # ==================== VAE+HyperNet (RF蒸馏) ====================
    logger.info("\n[4/4] VAE + HyperNet (RF蒸馏)...")
    distill_preds, distill_trues = [], []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
        X_train, y_train = X_scaled[train_idx], y[train_idx]
        X_test, y_test = X_scaled[test_idx], y[test_idx]
        
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
        
        # 增强
        X_aug, y_aug = vae_augment(vae, X_train, y_train, aug_factor=50, device=device)
        
        # 训练RF教师，获取软标签
        rf_teacher = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
        rf_teacher.fit(X_aug, y_aug)
        rf_probs = rf_teacher.predict_proba(X_aug)  # 软标签
        
        # 用蒸馏训练HyperNet学生
        student = HyperNetStudent(X_train.shape[1], len(np.unique(y_train))).to(device)
        student = train_with_distillation(student, X_aug, y_aug, rf_probs, epochs=300, device=device)
        
        # 预测
        student.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test).to(device)
            X_aug_t = torch.FloatTensor(X_aug).to(device)
            context = X_aug_t.mean(dim=0, keepdim=True)
            pred = student(X_test_t, context).argmax(dim=1).cpu().numpy()
        
        distill_preds.extend(pred.tolist())
        distill_trues.extend(y_test.tolist())
        logger.info(f"   Fold {fold_idx+1}/5: {accuracy_score(y_test, pred)*100:.2f}%")
    
    results['VAE+HyperNet(蒸馏)'] = accuracy_score(distill_trues, distill_preds) * 100
    logger.info(f"   VAE+HyperNet(蒸馏): {results['VAE+HyperNet(蒸馏)']:.2f}%")
    
    # ==================== 汇总 ====================
    logger.info("\n" + "=" * 70)
    logger.info("[结果对比]")
    logger.info("=" * 70)
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        marker = "🏆" if acc == max(results.values()) else "  "
        logger.info(f"{marker} {name:20s}: {acc:.2f}%")
    logger.info("=" * 70)
    
    result_file = OUTPUT_DIR / f'27_distillation_{timestamp}.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"保存: {result_file}")


if __name__ == '__main__':
    main()
