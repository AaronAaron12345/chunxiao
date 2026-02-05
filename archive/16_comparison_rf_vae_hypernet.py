#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
16_comparison_rf_vae_hypernet.py - 完整对比：RF vs VAE+RF vs VAE+HyperNet
=========================================================================
目标：让VAE+HyperNet达到RF水平(~82%)

核心改进：
1. 更简单的超网络设计（参考HyperTab思路）
2. 直接用增强数据训练简单分类器
3. 多个子网络+投票（不是超网络生成权重，而是直接集成）

运行: python 16_comparison_rf_vae_hypernet.py
"""

import os
import sys
import json
import time
import logging
import warnings
from datetime import datetime
from itertools import combinations
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

# 路径配置
SCRIPT_DIR = Path(__file__).parent
LOG_DIR = SCRIPT_DIR / 'logs'
OUTPUT_DIR = SCRIPT_DIR / 'output'
LOG_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ==================== VAE ====================
class VAE(nn.Module):
    """简化的VAE"""
    def __init__(self, input_dim, hidden_dim=64, latent_dim=8):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_var = nn.Linear(hidden_dim // 2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = self.fc_mu(h), self.fc_var(h)
        std = torch.exp(0.5 * log_var)
        z = mu + std * torch.randn_like(std)
        return self.decoder(z), mu, log_var


def train_vae_and_augment(X_train, y_train, n_interp=5, vae_epochs=30):
    """训练VAE并生成增强数据"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X_train.shape[1]
    
    # 归一化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    X_aug_list = [X_scaled]
    y_aug_list = [y_train]
    
    # 为每个类别训练VAE
    for cls in np.unique(y_train):
        X_cls = X_scaled[y_train == cls]
        if len(X_cls) < 3:
            continue
        
        vae = VAE(input_dim, hidden_dim=64, latent_dim=8).to(device)
        optimizer = torch.optim.Adam(vae.parameters(), lr=0.005)
        X_tensor = torch.FloatTensor(X_cls).to(device)
        
        vae.train()
        for _ in range(vae_epochs):
            optimizer.zero_grad()
            recon, mu, log_var = vae(X_tensor)
            recon_loss = nn.MSELoss()(recon, X_tensor)
            kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
            loss = recon_loss + 0.01 * kl_loss
            loss.backward()
            optimizer.step()
        
        vae.eval()
        with torch.no_grad():
            recon = vae(X_tensor)[0].cpu().numpy()
        
        # 插值生成新样本
        for alpha in np.linspace(0.2, 0.8, n_interp):
            aug = alpha * X_cls + (1 - alpha) * recon
            X_aug_list.append(aug)
            y_aug_list.append(np.full(len(X_cls), cls))
    
    X_aug = np.vstack(X_aug_list)
    y_aug = np.hstack(y_aug_list)
    
    return X_aug, y_aug, scaler


# ==================== 简单神经网络分类器 ====================
class SimpleClassifier(nn.Module):
    """简单的两层分类器"""
    def __init__(self, input_dim, hidden_dim, n_classes):
        super(SimpleClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, n_classes)
        )
    
    def forward(self, x):
        return self.net(x)


class EnsembleClassifier:
    """集成分类器 - 多个神经网络投票"""
    def __init__(self, input_dim, n_classes, n_estimators=5, hidden_dim=32):
        self.n_estimators = n_estimators
        self.models = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for i in range(n_estimators):
            # 每个子网络略有不同
            h_dim = hidden_dim + (i % 3) * 8
            model = SimpleClassifier(input_dim, h_dim, n_classes).to(self.device)
            self.models.append(model)
    
    def fit(self, X, y, epochs=50, lr=0.01):
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        criterion = nn.CrossEntropyLoss()
        
        for i, model in enumerate(self.models):
            # 每个模型用不同的数据子集（bootstrap）
            np.random.seed(42 + i)
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_boot = X_tensor[indices]
            y_boot = y_tensor[indices]
            
            optimizer = optim.Adam(model.parameters(), lr=lr * (0.8 + 0.4 * np.random.random()))
            model.train()
            
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = model(X_boot)
                loss = criterion(outputs, y_boot)
                loss.backward()
                optimizer.step()
    
    def predict(self, X):
        X_tensor = torch.FloatTensor(X).to(self.device)
        all_probs = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                outputs = model(X_tensor)
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs)
        
        # 投票
        avg_probs = torch.stack(all_probs).mean(dim=0)
        return avg_probs.argmax(dim=1).cpu().numpy()


# ==================== 处理单个fold ====================
def process_fold_rf(args):
    """纯RF - 基线"""
    fold_idx, X_train, y_train, X_test, y_test = args
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)
    
    return {'fold_idx': fold_idx, 'y_true': y_test.tolist(), 'y_pred': y_pred.tolist(), 'method': 'RF'}


def process_fold_vae_rf(args):
    """VAE增强 + RF"""
    fold_idx, X_train, y_train, X_test, y_test = args
    
    # VAE增强
    X_aug, y_aug, scaler = train_vae_and_augment(X_train, y_train, n_interp=5, vae_epochs=30)
    X_test_scaled = scaler.transform(X_test)
    
    # RF在增强数据上训练
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    rf.fit(X_aug, y_aug)
    y_pred = rf.predict(X_test_scaled)
    
    return {'fold_idx': fold_idx, 'y_true': y_test.tolist(), 'y_pred': y_pred.tolist(), 'method': 'VAE+RF'}


def process_fold_vae_ensemble(args):
    """VAE增强 + 神经网络集成"""
    fold_idx, X_train, y_train, X_test, y_test = args
    
    # VAE增强
    X_aug, y_aug, scaler = train_vae_and_augment(X_train, y_train, n_interp=5, vae_epochs=30)
    X_test_scaled = scaler.transform(X_test)
    
    # 神经网络集成
    n_classes = len(np.unique(y_train))
    ensemble = EnsembleClassifier(
        input_dim=X_train.shape[1], 
        n_classes=n_classes, 
        n_estimators=5, 
        hidden_dim=32
    )
    ensemble.fit(X_aug, y_aug, epochs=50, lr=0.01)
    y_pred = ensemble.predict(X_test_scaled)
    
    return {'fold_idx': fold_idx, 'y_true': y_test.tolist(), 'y_pred': y_pred.tolist(), 'method': 'VAE+Ensemble'}


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR / f'16_comparison_{timestamp}.log'
    
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
    logger.info("16_comparison_rf_vae_hypernet.py - RF vs VAE+RF vs VAE+神经网络集成")
    logger.info("=" * 70)
    
    # 加载数据
    data_path = SCRIPT_DIR / 'data' / 'Data_for_Jinming.csv'
    df = pd.read_csv(data_path)
    feature_cols = ['LAA', 'Glutamate', 'Choline', 'Sarcosine']
    X = df[feature_cols].values.astype(np.float32)
    y_raw = df['Group'].values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    n_samples = len(X)
    logger.info(f"数据: {n_samples} 样本, {X.shape[1]} 特征")
    
    # 生成所有fold
    all_indices = np.arange(n_samples)
    leave_p_out = 2
    test_combos = list(combinations(all_indices, leave_p_out))
    n_folds = len(test_combos)
    logger.info(f"Leave-{leave_p_out}-Out: {n_folds} 个 folds")
    
    # 准备fold数据
    fold_datas = []
    for fold_idx, test_idx in enumerate(test_combos):
        test_idx = np.array(test_idx)
        train_idx = np.setdiff1d(all_indices, test_idx)
        fold_datas.append((fold_idx, X[train_idx], y[train_idx], X[test_idx], y[test_idx]))
    
    n_processes = min(cpu_count(), 64)
    logger.info(f"使用 {n_processes} 个进程并行")
    
    results = {}
    
    # 1. 纯RF基线
    logger.info("\n[1/3] 运行 RF 基线...")
    start = time.time()
    with Pool(n_processes) as pool:
        rf_results = list(pool.imap(process_fold_rf, fold_datas, chunksize=10))
    rf_time = time.time() - start
    
    all_true = [item for r in rf_results for item in r['y_true']]
    all_pred = [item for r in rf_results for item in r['y_pred']]
    rf_acc = accuracy_score(all_true, all_pred) * 100
    logger.info(f"   RF 准确率: {rf_acc:.2f}%  用时: {rf_time:.1f}秒")
    results['RF'] = {'accuracy': rf_acc, 'time': rf_time}
    
    # 2. VAE + RF
    logger.info("\n[2/3] 运行 VAE + RF...")
    start = time.time()
    with Pool(n_processes) as pool:
        vae_rf_results = list(pool.imap(process_fold_vae_rf, fold_datas, chunksize=10))
    vae_rf_time = time.time() - start
    
    all_true = [item for r in vae_rf_results for item in r['y_true']]
    all_pred = [item for r in vae_rf_results for item in r['y_pred']]
    vae_rf_acc = accuracy_score(all_true, all_pred) * 100
    logger.info(f"   VAE+RF 准确率: {vae_rf_acc:.2f}%  用时: {vae_rf_time:.1f}秒")
    results['VAE+RF'] = {'accuracy': vae_rf_acc, 'time': vae_rf_time}
    
    # 3. VAE + 神经网络集成
    logger.info("\n[3/3] 运行 VAE + 神经网络集成...")
    start = time.time()
    with Pool(n_processes) as pool:
        vae_ensemble_results = list(pool.imap(process_fold_vae_ensemble, fold_datas, chunksize=10))
    vae_ensemble_time = time.time() - start
    
    all_true = [item for r in vae_ensemble_results for item in r['y_true']]
    all_pred = [item for r in vae_ensemble_results for item in r['y_pred']]
    vae_ensemble_acc = accuracy_score(all_true, all_pred) * 100
    logger.info(f"   VAE+Ensemble 准确率: {vae_ensemble_acc:.2f}%  用时: {vae_ensemble_time:.1f}秒")
    results['VAE+Ensemble'] = {'accuracy': vae_ensemble_acc, 'time': vae_ensemble_time}
    
    # 总结
    logger.info("\n" + "=" * 70)
    logger.info("[对比结果]")
    logger.info("=" * 70)
    logger.info(f"  RF (基线):        {rf_acc:.2f}%")
    logger.info(f"  VAE + RF:         {vae_rf_acc:.2f}%")
    logger.info(f"  VAE + 神经网络:   {vae_ensemble_acc:.2f}%")
    logger.info("=" * 70)
    
    # 保存结果
    result_file = OUTPUT_DIR / f'16_comparison_{timestamp}.json'
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'n_samples': n_samples,
            'n_folds': n_folds,
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"结果已保存: {result_file}")


if __name__ == '__main__':
    main()
