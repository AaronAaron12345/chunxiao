#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
23_vae_hypernet_gpu_v2.py - VAE + HyperNet (GPU并行修复版)
=========================================================
修复CUDA multiprocessing问题，使用spawn模式

核心思路：
1. VAE: 大量数据增强
2. HyperNet: 模仿RF的结构 - 多子网络 + 特征bagging + 投票

运行: python 23_vae_hypernet_gpu_v2.py
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
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

# 必须在import torch之后、使用multiprocessing之前设置
mp.set_start_method('spawn', force=True)

SCRIPT_DIR = Path(__file__).parent
LOG_DIR = SCRIPT_DIR / 'logs'
OUTPUT_DIR = SCRIPT_DIR / 'output'
LOG_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


class VAE(nn.Module):
    """变分自编码器"""
    def __init__(self, input_dim, hidden_dim=32, latent_dim=8):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
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


class HyperNetEnsemble(nn.Module):
    """HyperNet集成 - 模仿RF"""
    def __init__(self, input_dim, n_classes=2, n_estimators=30, hidden_dim=16, dropout=0.2):
        super(HyperNetEnsemble, self).__init__()
        
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.n_estimators = n_estimators
        
        # 特征子集 (类似RF的max_features)
        self.n_features = max(2, int(np.sqrt(input_dim)))
        self.feature_indices = [
            np.random.choice(input_dim, self.n_features, replace=True)
            for _ in range(n_estimators)
        ]
        
        # 子网络
        self.sub_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.n_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes),
            )
            for _ in range(n_estimators)
        ])
    
    def forward(self, x):
        outputs = []
        for i, subnet in enumerate(self.sub_networks):
            x_subset = x[:, self.feature_indices[i]]
            outputs.append(subnet(x_subset))
        return torch.stack(outputs).mean(dim=0)
    
    def predict_vote(self, x):
        votes = []
        for i, subnet in enumerate(self.sub_networks):
            x_subset = x[:, self.feature_indices[i]]
            pred = subnet(x_subset).argmax(dim=1)
            votes.append(pred)
        votes = torch.stack(votes, dim=0)
        return [torch.bincount(votes[:, j], minlength=self.n_classes).argmax().item() 
                for j in range(x.shape[0])]


def vae_augment(X_train, y_train, device, aug_factor=30):
    """VAE数据增强"""
    input_dim = X_train.shape[1]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    X_aug, y_aug = [X_scaled], [y_train]
    
    for cls in np.unique(y_train):
        X_cls = X_scaled[y_train == cls]
        if len(X_cls) < 2:
            continue
        
        vae = VAE(input_dim).to(device)
        optimizer = optim.Adam(vae.parameters(), lr=0.01)
        X_tensor = torch.FloatTensor(X_cls).to(device)
        
        vae.train()
        for _ in range(80):
            optimizer.zero_grad()
            recon, mu, log_var = vae(X_tensor)
            loss = nn.MSELoss()(recon, X_tensor) + 0.01 * (-0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp()))
            loss.backward()
            optimizer.step()
        
        vae.eval()
        with torch.no_grad():
            recon = vae(X_tensor)[0].cpu().numpy()
            # 插值
            for alpha in np.linspace(0.1, 0.9, aug_factor // 3):
                X_aug.append(alpha * X_cls + (1 - alpha) * recon)
                y_aug.append(np.full(len(X_cls), cls))
            
            # 潜在采样
            mu, log_var = vae.encode(X_tensor)
            for _ in range(aug_factor // 3):
                z = vae.reparameterize(mu, log_var * 0.3)
                X_aug.append(vae.decode(z).cpu().numpy())
                y_aug.append(np.full(len(X_cls), cls))
            
            # 噪声
            for std in [0.05, 0.1, 0.15]:
                X_aug.append(X_cls + np.random.randn(*X_cls.shape) * std)
                y_aug.append(np.full(len(X_cls), cls))
    
    return np.vstack(X_aug), np.hstack(y_aug), scaler


def process_gpu_fold(args):
    """处理单个fold (GPU)"""
    fold_idx, train_idx, test_idx, X_all, y_all, gpu_id = args
    
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]
    
    # VAE增强
    X_aug, y_aug, scaler = vae_augment(X_train, y_train, device, aug_factor=30)
    X_test_s = scaler.transform(X_test)
    
    # 训练HyperNet
    n_classes = len(np.unique(y_aug))
    model = HyperNetEnsemble(X_aug.shape[1], n_classes, n_estimators=30).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    X_tensor = torch.FloatTensor(X_aug).to(device)
    y_tensor = torch.LongTensor(y_aug).to(device)
    
    model.train()
    for _ in range(80):
        idx = np.random.choice(len(X_aug), len(X_aug), replace=True)
        optimizer.zero_grad()
        loss = criterion(model(X_tensor[idx]), y_tensor[idx])
        loss.backward()
        optimizer.step()
    
    # 预测
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_s).to(device)
        y_pred = model.predict_vote(X_test_tensor)
    
    return {'fold_idx': fold_idx, 'y_true': y_test.tolist(), 'y_pred': y_pred}


def process_rf_fold(args):
    """RF基线"""
    fold_idx, train_idx, test_idx, X_all, y_all = args
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    rf.fit(X_train_s, y_train)
    return {'y_true': y_test.tolist(), 'y_pred': rf.predict(X_test_s).tolist()}


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR / f'23_vae_hypernet_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding='utf-8')]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 70)
    logger.info("23_vae_hypernet_gpu_v2.py - VAE + HyperNet (GPU)")
    logger.info("=" * 70)
    
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    logger.info(f"GPU: {n_gpus} 个")
    
    data_path = SCRIPT_DIR / 'data' / 'Data_for_Jinming.csv'
    df = pd.read_csv(data_path)
    X = df[['LAA', 'Glutamate', 'Choline', 'Sarcosine']].values.astype(np.float32)
    y = LabelEncoder().fit_transform(df['Group'].values)
    
    n_samples = len(X)
    logger.info(f"数据: {n_samples} 样本")
    
    test_combos = list(combinations(range(n_samples), 2))
    n_folds = len(test_combos)
    logger.info(f"Leave-2-Out: {n_folds} folds")
    
    fold_list = []
    for fold_idx, test_idx in enumerate(test_combos):
        test_idx_arr = np.array(test_idx)
        train_idx = np.setdiff1d(np.arange(n_samples), test_idx_arr)
        fold_list.append((fold_idx, train_idx, test_idx_arr))
    
    results = {}
    
    # RF基线
    logger.info("\n[1/2] RF...")
    start = time.time()
    rf_args = [(f[0], f[1], f[2], X, y) for f in fold_list]
    with mp.Pool(64) as pool:
        rf_results = list(pool.imap(process_rf_fold, rf_args, chunksize=10))
    results['RF'] = accuracy_score(
        [i for r in rf_results for i in r['y_true']],
        [i for r in rf_results for i in r['y_pred']]
    ) * 100
    logger.info(f"   RF: {results['RF']:.2f}% ({time.time()-start:.1f}s)")
    
    # VAE+HyperNet
    logger.info("\n[2/2] VAE+HyperNet (GPU)...")
    start = time.time()
    
    use_gpus = min(n_gpus, 6) if n_gpus > 0 else 1
    gpu_args = [(f[0], f[1], f[2], X, y, f[0] % use_gpus) for f in fold_list]
    
    # 串行处理避免CUDA问题
    hypernet_results = []
    for i, arg in enumerate(gpu_args):
        result = process_gpu_fold(arg)
        hypernet_results.append(result)
        if (i + 1) % 50 == 0:
            logger.info(f"   进度: {i+1}/{n_folds} ({100*(i+1)/n_folds:.1f}%)")
    
    results['VAE+HyperNet'] = accuracy_score(
        [i for r in hypernet_results for i in r['y_true']],
        [i for r in hypernet_results for i in r['y_pred']]
    ) * 100
    logger.info(f"   VAE+HyperNet: {results['VAE+HyperNet']:.2f}% ({time.time()-start:.1f}s)")
    
    logger.info("\n" + "=" * 70)
    logger.info("[结果]")
    for name, acc in results.items():
        marker = "🏆" if acc == max(results.values()) else "  "
        logger.info(f"{marker} {name}: {acc:.2f}%")
    logger.info("=" * 70)
    
    with open(OUTPUT_DIR / f'23_vae_hypernet_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
