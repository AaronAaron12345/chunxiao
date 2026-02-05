#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
31_hypernet_simple_stable.py - 简化稳定版HyperNet
=================================================
核心改进思路：
1. 简化超网络结构 - 越简单越稳定
2. 直接预测类别概率而不是生成网络权重
3. 使用注意力机制代替复杂的权重生成
4. 原型网络思想 - 基于距离的分类

目标：稳定达到80%

运行: python 31_hypernet_simple_stable.py
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


class PrototypeHyperNet(nn.Module):
    """
    基于原型的超网络 - 更稳定的设计
    
    思想：
    1. 学习每个类别的多个原型（prototype）
    2. 基于样本与原型的距离进行分类
    3. 超网络动态调整距离度量
    """
    def __init__(self, input_dim, n_classes, n_prototypes_per_class=5):
        super(PrototypeHyperNet, self).__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.n_prototypes = n_prototypes_per_class
        
        # 特征嵌入
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
        )
        
        # 类别原型 (n_classes * n_prototypes, embed_dim)
        self.prototypes = nn.Parameter(
            torch.randn(n_classes * n_prototypes_per_class, 16) * 0.1
        )
        
        # 原型重要性权重
        self.prototype_weights = nn.Parameter(
            torch.ones(n_classes, n_prototypes_per_class) / n_prototypes_per_class
        )
        
        # 温度参数
        self.temperature = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x):
        # 编码输入
        z = self.encoder(x)  # (batch, 16)
        
        # 计算到所有原型的距离
        # prototypes: (n_classes * n_prototypes, 16)
        dists = torch.cdist(z, self.prototypes)  # (batch, n_classes * n_prototypes)
        
        # 重塑为 (batch, n_classes, n_prototypes)
        dists = dists.view(-1, self.n_classes, self.n_prototypes)
        
        # 加权平均每个类别的原型距离
        weights = F.softmax(self.prototype_weights, dim=1)  # (n_classes, n_prototypes)
        class_dists = (dists * weights.unsqueeze(0)).sum(dim=2)  # (batch, n_classes)
        
        # 转换为logits（距离越小，概率越大）
        logits = -class_dists / self.temperature.clamp(min=0.1)
        
        return logits
    
    def init_prototypes(self, X, y):
        """用训练数据初始化原型"""
        with torch.no_grad():
            z = self.encoder(X)
            for c in range(self.n_classes):
                mask = (y == c)
                if mask.sum() > 0:
                    class_embeds = z[mask]
                    # K-means style初始化
                    for p in range(self.n_prototypes):
                        if p < len(class_embeds):
                            self.prototypes.data[c * self.n_prototypes + p] = class_embeds[p]
                        else:
                            # 随机扰动
                            idx = np.random.randint(len(class_embeds))
                            self.prototypes.data[c * self.n_prototypes + p] = (
                                class_embeds[idx] + torch.randn(16, device=X.device) * 0.1
                            )


class EnsemblePrototypeNet(nn.Module):
    """原型网络集成"""
    def __init__(self, input_dim, n_classes, n_models=10):
        super(EnsemblePrototypeNet, self).__init__()
        self.models = nn.ModuleList([
            PrototypeHyperNet(input_dim, n_classes, n_prototypes_per_class=3 + i % 5)
            for i in range(n_models)
        ])
    
    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return torch.stack(outputs, dim=0).mean(dim=0)
    
    def init_all_prototypes(self, X, y):
        for model in self.models:
            model.init_prototypes(X, y)


def vae_augment(vae, X_train, y_train, aug_factor=80, device='cuda'):
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
            for alpha in np.linspace(0.1, 0.9, aug_factor // 8):
                X_aug.append(alpha * X_cls_np + (1 - alpha) * recon)
                y_aug.append(np.full(mask.sum(), cls))
            
            # 采样
            for scale in [0.2, 0.3]:
                for _ in range(aug_factor // 16):
                    z = mu + torch.exp(0.5 * log_var) * torch.randn_like(log_var) * scale
                    X_aug.append(vae.decoder(z).cpu().numpy())
                    y_aug.append(np.full(mask.sum(), cls))
    
    return np.vstack(X_aug), np.hstack(y_aug)


def train_prototype_net(model, X, y, epochs=200, lr=0.01, device='cuda'):
    """训练原型网络"""
    model.train()
    
    X_t = torch.FloatTensor(X).to(device)
    y_t = torch.LongTensor(y).to(device)
    
    # 初始化原型
    if hasattr(model, 'init_prototypes'):
        model.init_prototypes(X_t, y_t)
    elif hasattr(model, 'init_all_prototypes'):
        model.init_all_prototypes(X_t, y_t)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
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
    log_file = LOG_DIR / f'31_prototype_{timestamp}.log'
    
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
    logger.info("31_hypernet_simple_stable.py - 原型网络版HyperNet")
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
    logger.info("\n[1/5] RF 基线...")
    rf_preds, rf_trues = [], []
    for train_idx, test_idx in skf.split(X_scaled, y):
        rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
        rf.fit(X_scaled[train_idx], y[train_idx])
        rf_preds.extend(rf.predict(X_scaled[test_idx]).tolist())
        rf_trues.extend(y[test_idx].tolist())
    results['RF'] = accuracy_score(rf_trues, rf_preds) * 100
    logger.info(f"   RF: {results['RF']:.2f}%")
    
    # ==================== VAE+RF ====================
    logger.info("\n[2/5] VAE + RF...")
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
        
        X_aug, y_aug = vae_augment(vae, X_train, y_train, aug_factor=80, device=device)
        
        rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
        rf.fit(X_aug, y_aug)
        vae_rf_preds.extend(rf.predict(X_test).tolist())
        vae_rf_trues.extend(y_test.tolist())
    
    results['VAE+RF'] = accuracy_score(vae_rf_trues, vae_rf_preds) * 100
    logger.info(f"   VAE+RF: {results['VAE+RF']:.2f}%")
    
    # ==================== 原型网络（无VAE） ====================
    logger.info("\n[3/5] PrototypeNet (无VAE)...")
    proto_preds, proto_trues = [], []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
        X_train, y_train = X_scaled[train_idx], y[train_idx]
        X_test, y_test = X_scaled[test_idx], y[test_idx]
        
        torch.manual_seed(42)
        model = EnsemblePrototypeNet(X_train.shape[1], len(np.unique(y_train)), n_models=10).to(device)
        train_prototype_net(model, X_train, y_train, epochs=200, device=device)
        
        model.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test).to(device)
            pred = model(X_test_t).argmax(dim=1).cpu().numpy()
        
        proto_preds.extend(pred.tolist())
        proto_trues.extend(y_test.tolist())
        logger.info(f"   Fold {fold_idx+1}/5: {accuracy_score(y_test, pred)*100:.2f}%")
    
    results['PrototypeNet'] = accuracy_score(proto_trues, proto_preds) * 100
    logger.info(f"   PrototypeNet: {results['PrototypeNet']:.2f}%")
    
    # ==================== VAE+原型网络 ====================
    logger.info("\n[4/5] VAE + PrototypeNet...")
    vae_proto_preds, vae_proto_trues = [], []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
        X_train, y_train = X_scaled[train_idx], y[train_idx]
        X_test, y_test = X_scaled[test_idx], y[test_idx]
        
        torch.manual_seed(42)
        
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
        
        X_aug, y_aug = vae_augment(vae, X_train, y_train, aug_factor=80, device=device)
        
        # 原型网络
        model = EnsemblePrototypeNet(X_train.shape[1], len(np.unique(y_train)), n_models=15).to(device)
        train_prototype_net(model, X_aug, y_aug, epochs=300, device=device)
        
        model.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test).to(device)
            pred = model(X_test_t).argmax(dim=1).cpu().numpy()
        
        vae_proto_preds.extend(pred.tolist())
        vae_proto_trues.extend(y_test.tolist())
        logger.info(f"   Fold {fold_idx+1}/5: {accuracy_score(y_test, pred)*100:.2f}%")
    
    results['VAE+PrototypeNet'] = accuracy_score(vae_proto_trues, vae_proto_preds) * 100
    logger.info(f"   VAE+PrototypeNet: {results['VAE+PrototypeNet']:.2f}%")
    
    # ==================== 多次运行取最佳 ====================
    logger.info("\n[5/5] VAE + PrototypeNet (多次运行取最佳)...")
    best_preds, best_trues = [], []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
        X_train, y_train = X_scaled[train_idx], y[train_idx]
        X_test, y_test = X_scaled[test_idx], y[test_idx]
        
        best_pred = None
        best_train_acc = 0
        
        for run in range(5):
            torch.manual_seed(run * 100)
            np.random.seed(run * 100)
            
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
            
            X_aug, y_aug = vae_augment(vae, X_train, y_train, aug_factor=80, device=device)
            
            model = EnsemblePrototypeNet(X_train.shape[1], len(np.unique(y_train)), n_models=15).to(device)
            train_prototype_net(model, X_aug, y_aug, epochs=300, device=device)
            
            # 选择训练集上表现最好的模型
            model.eval()
            with torch.no_grad():
                train_pred = model(torch.FloatTensor(X_aug).to(device)).argmax(dim=1).cpu().numpy()
                train_acc = accuracy_score(y_aug, train_pred)
                
                if train_acc > best_train_acc:
                    best_train_acc = train_acc
                    X_test_t = torch.FloatTensor(X_test).to(device)
                    best_pred = model(X_test_t).argmax(dim=1).cpu().numpy()
        
        best_preds.extend(best_pred.tolist())
        best_trues.extend(y_test.tolist())
        logger.info(f"   Fold {fold_idx+1}/5: {accuracy_score(y_test, best_pred)*100:.2f}%")
    
    results['VAE+PrototypeNet(最佳)'] = accuracy_score(best_trues, best_preds) * 100
    logger.info(f"   VAE+PrototypeNet(最佳): {results['VAE+PrototypeNet(最佳)']:.2f}%")
    
    # ==================== 汇总 ====================
    logger.info("\n" + "=" * 70)
    logger.info("[结果对比]")
    logger.info("=" * 70)
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        marker = "🏆" if acc == max(results.values()) else "  "
        logger.info(f"{marker} {name:25s}: {acc:.2f}%")
    logger.info("=" * 70)
    
    result_file = OUTPUT_DIR / f'31_prototype_{timestamp}.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"保存: {result_file}")


if __name__ == '__main__':
    main()
