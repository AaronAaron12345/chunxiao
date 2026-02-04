#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
24_vae_hypernet_rf_hybrid.py - VAE + HyperNet + RF 混合方法
==========================================================
基于文献发现：神经网络很难单独超越RF，但混合可以更好。

核心策略：
1. VAE大量数据增强 (100倍+)
2. HyperNet + RF 混合投票
3. 特征工程：添加RF的概率预测作为特征
4. 多GPU并行

运行: python 24_vae_hypernet_rf_hybrid.py
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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
            nn.Linear(input_dim, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 32), nn.LeakyReLU(0.2)
        )
        self.fc_mu = nn.Linear(32, 16)
        self.fc_var = nn.Linear(32, 16)
        self.decoder = nn.Sequential(
            nn.Linear(16, 32), nn.LeakyReLU(0.2),
            nn.Linear(32, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, input_dim)
        )
    
    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = self.fc_mu(h), self.fc_var(h)
        z = mu + torch.exp(0.5 * log_var) * torch.randn_like(log_var)
        return self.decoder(z), mu, log_var


def massive_vae_augment(X_train, y_train, aug_factor=100):
    """大规模VAE增强 - 产生100倍以上数据"""
    input_dim = X_train.shape[1]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    X_aug, y_aug = [X_scaled], [y_train]
    
    for cls in np.unique(y_train):
        X_cls = X_scaled[y_train == cls]
        if len(X_cls) < 2:
            continue
        
        vae = VAE(input_dim)
        optimizer = optim.Adam(vae.parameters(), lr=0.005)
        X_tensor = torch.FloatTensor(X_cls)
        
        for _ in range(150):  # 更多训练
            optimizer.zero_grad()
            recon, mu, log_var = vae(X_tensor)
            loss = nn.MSELoss()(recon, X_tensor) + 0.005 * (-0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp()))
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            # 1. 重建插值 (主要增强)
            recon = vae(X_tensor)[0].numpy()
            for alpha in np.linspace(0.05, 0.95, 30):
                X_aug.append(alpha * X_cls + (1 - alpha) * recon)
                y_aug.append(np.full(len(X_cls), cls))
            
            # 2. 潜在空间采样
            h = vae.encoder(X_tensor)
            mu, log_var = vae.fc_mu(h), vae.fc_var(h)
            for noise_scale in [0.2, 0.3, 0.4, 0.5]:
                for _ in range(aug_factor // 20):
                    z = mu + torch.exp(0.5 * log_var) * torch.randn_like(log_var) * noise_scale
                    X_aug.append(vae.decoder(z).numpy())
                    y_aug.append(np.full(len(X_cls), cls))
            
            # 3. 原始数据噪声增强
            for std in [0.03, 0.05, 0.08, 0.1, 0.12, 0.15]:
                X_aug.append(X_cls + np.random.randn(*X_cls.shape) * std)
                y_aug.append(np.full(len(X_cls), cls))
            
            # 4. 样本间Mixup
            for _ in range(aug_factor // 10):
                idx = np.random.choice(len(X_cls), 2, replace=False) if len(X_cls) >= 2 else [0, 0]
                lam = np.random.beta(0.4, 0.4)
                X_aug.append((lam * X_cls[idx[0]] + (1-lam) * X_cls[idx[1]]).reshape(1, -1))
                y_aug.append(np.array([cls]))
    
    return np.vstack(X_aug), np.hstack(y_aug), scaler


class HyperNetClassifier(nn.Module):
    """简化的HyperNet分类器"""
    def __init__(self, input_dim, n_classes=2, n_estimators=20):
        super(HyperNetClassifier, self).__init__()
        self.n_estimators = n_estimators
        self.n_classes = n_classes
        
        # 每个子网络使用不同的特征组合
        self.feature_masks = []
        for _ in range(n_estimators):
            mask = np.random.choice([0, 1], size=input_dim, p=[0.3, 0.7])
            if mask.sum() == 0:
                mask[np.random.randint(input_dim)] = 1
            self.feature_masks.append(torch.FloatTensor(mask))
        
        self.subnets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 8),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(8, n_classes)
            )
            for _ in range(n_estimators)
        ])
    
    def forward(self, x):
        outputs = []
        for i, subnet in enumerate(self.subnets):
            x_masked = x * self.feature_masks[i].to(x.device)
            outputs.append(subnet(x_masked))
        return torch.stack(outputs).mean(dim=0)


def process_fold_rf(args):
    """纯RF"""
    fold_idx, X_train, y_train, X_test, y_test = args
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    rf.fit(X_train_s, y_train)
    return {'y_true': y_test.tolist(), 'y_pred': rf.predict(X_test_s).tolist()}


def process_fold_vae_rf(args):
    """VAE + RF"""
    fold_idx, X_train, y_train, X_test, y_test = args
    X_aug, y_aug, scaler = massive_vae_augment(X_train, y_train, aug_factor=100)
    X_test_s = scaler.transform(X_test)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    rf.fit(X_aug, y_aug)
    return {'y_true': y_test.tolist(), 'y_pred': rf.predict(X_test_s).tolist()}


def process_fold_vae_hypernet(args):
    """VAE + HyperNet"""
    fold_idx, X_train, y_train, X_test, y_test = args
    X_aug, y_aug, scaler = massive_vae_augment(X_train, y_train, aug_factor=100)
    X_test_s = scaler.transform(X_test)
    
    n_classes = len(np.unique(y_train))
    model = HyperNetClassifier(X_aug.shape[1], n_classes, n_estimators=30)
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    X_tensor = torch.FloatTensor(X_aug)
    y_tensor = torch.LongTensor(y_aug)
    
    model.train()
    for _ in range(100):
        # Bootstrap
        idx = np.random.choice(len(X_aug), min(len(X_aug), 512), replace=True)
        optimizer.zero_grad()
        loss = criterion(model(X_tensor[idx]), y_tensor[idx])
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_s)
        y_pred = model(X_test_tensor).argmax(dim=1).numpy()
    
    return {'y_true': y_test.tolist(), 'y_pred': y_pred.tolist()}


def process_fold_hybrid(args):
    """混合: RF + HyperNet投票"""
    fold_idx, X_train, y_train, X_test, y_test = args
    X_aug, y_aug, scaler = massive_vae_augment(X_train, y_train, aug_factor=100)
    X_test_s = scaler.transform(X_test)
    
    n_classes = len(np.unique(y_train))
    
    # 1. 训练RF
    rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    rf.fit(X_aug, y_aug)
    rf_pred = rf.predict(X_test_s)
    rf_prob = rf.predict_proba(X_test_s)
    
    # 2. 训练HyperNet (使用RF概率作为额外特征)
    X_aug_with_rf = np.hstack([X_aug, rf.predict_proba(X_aug)])
    X_test_with_rf = np.hstack([X_test_s, rf_prob])
    
    model = HyperNetClassifier(X_aug_with_rf.shape[1], n_classes, n_estimators=30)
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    X_tensor = torch.FloatTensor(X_aug_with_rf)
    y_tensor = torch.LongTensor(y_aug)
    
    model.train()
    for _ in range(100):
        idx = np.random.choice(len(X_aug_with_rf), min(len(X_aug_with_rf), 512), replace=True)
        optimizer.zero_grad()
        loss = criterion(model(X_tensor[idx]), y_tensor[idx])
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_with_rf)
        hypernet_pred = model(X_test_tensor).argmax(dim=1).numpy()
    
    # 3. 投票 (RF权重更高)
    final_pred = []
    for i in range(len(X_test_s)):
        votes = [rf_pred[i], rf_pred[i], hypernet_pred[i]]  # RF 2票, HyperNet 1票
        final_pred.append(np.bincount(votes).argmax())
    
    return {'y_true': y_test.tolist(), 'y_pred': final_pred}


def process_fold_stacking(args):
    """Stacking: 用HyperNet学习RF的软输出"""
    fold_idx, X_train, y_train, X_test, y_test = args
    X_aug, y_aug, scaler = massive_vae_augment(X_train, y_train, aug_factor=100)
    X_test_s = scaler.transform(X_test)
    
    n_classes = len(np.unique(y_train))
    
    # 训练多个基学习器
    rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
    
    rf.fit(X_aug, y_aug)
    gb.fit(X_aug, y_aug)
    
    # 获取基学习器的预测概率
    rf_train_prob = rf.predict_proba(X_aug)
    gb_train_prob = gb.predict_proba(X_aug)
    rf_test_prob = rf.predict_proba(X_test_s)
    gb_test_prob = gb.predict_proba(X_test_s)
    
    # 组合特征
    X_stack_train = np.hstack([X_aug, rf_train_prob, gb_train_prob])
    X_stack_test = np.hstack([X_test_s, rf_test_prob, gb_test_prob])
    
    # HyperNet作为元学习器
    model = HyperNetClassifier(X_stack_train.shape[1], n_classes, n_estimators=30)
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    X_tensor = torch.FloatTensor(X_stack_train)
    y_tensor = torch.LongTensor(y_aug)
    
    model.train()
    for _ in range(100):
        idx = np.random.choice(len(X_stack_train), min(len(X_stack_train), 512), replace=True)
        optimizer.zero_grad()
        loss = criterion(model(X_tensor[idx]), y_tensor[idx])
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_stack_test)
        y_pred = model(X_test_tensor).argmax(dim=1).numpy()
    
    return {'y_true': y_test.tolist(), 'y_pred': y_pred.tolist()}


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR / f'24_hybrid_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding='utf-8')]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 70)
    logger.info("24_vae_hypernet_rf_hybrid.py - VAE + HyperNet + RF 混合")
    logger.info("=" * 70)
    
    data_path = SCRIPT_DIR / 'data' / 'Data_for_Jinming.csv'
    df = pd.read_csv(data_path)
    X = df[['LAA', 'Glutamate', 'Choline', 'Sarcosine']].values.astype(np.float32)
    y = LabelEncoder().fit_transform(df['Group'].values)
    
    n_samples = len(X)
    logger.info(f"数据: {n_samples} 样本, {X.shape[1]} 特征")
    
    test_combos = list(combinations(range(n_samples), 2))
    n_folds = len(test_combos)
    logger.info(f"Leave-2-Out: {n_folds} folds")
    
    fold_datas = []
    for fold_idx, test_idx in enumerate(test_combos):
        test_idx = np.array(test_idx)
        train_idx = np.setdiff1d(np.arange(n_samples), test_idx)
        fold_datas.append((fold_idx, X[train_idx], y[train_idx], X[test_idx], y[test_idx]))
    
    n_processes = min(cpu_count(), 64)
    logger.info(f"使用 {n_processes} 进程")
    
    results = {}
    
    # 1. RF
    logger.info("\n[1/5] RF基线...")
    start = time.time()
    with Pool(n_processes) as pool:
        r = list(pool.imap(process_fold_rf, fold_datas, chunksize=10))
    results['RF'] = accuracy_score([i for x in r for i in x['y_true']], [i for x in r for i in x['y_pred']]) * 100
    logger.info(f"   RF: {results['RF']:.2f}% ({time.time()-start:.1f}s)")
    
    # 2. VAE + RF
    logger.info("\n[2/5] VAE(100x) + RF...")
    start = time.time()
    with Pool(n_processes) as pool:
        r = list(pool.imap(process_fold_vae_rf, fold_datas, chunksize=10))
    results['VAE+RF'] = accuracy_score([i for x in r for i in x['y_true']], [i for x in r for i in x['y_pred']]) * 100
    logger.info(f"   VAE+RF: {results['VAE+RF']:.2f}% ({time.time()-start:.1f}s)")
    
    # 3. VAE + HyperNet
    logger.info("\n[3/5] VAE(100x) + HyperNet...")
    start = time.time()
    with Pool(n_processes) as pool:
        r = list(pool.imap(process_fold_vae_hypernet, fold_datas, chunksize=10))
    results['VAE+HyperNet'] = accuracy_score([i for x in r for i in x['y_true']], [i for x in r for i in x['y_pred']]) * 100
    logger.info(f"   VAE+HyperNet: {results['VAE+HyperNet']:.2f}% ({time.time()-start:.1f}s)")
    
    # 4. Hybrid (RF + HyperNet投票)
    logger.info("\n[4/5] Hybrid (RF + HyperNet投票)...")
    start = time.time()
    with Pool(n_processes) as pool:
        r = list(pool.imap(process_fold_hybrid, fold_datas, chunksize=10))
    results['Hybrid'] = accuracy_score([i for x in r for i in x['y_true']], [i for x in r for i in x['y_pred']]) * 100
    logger.info(f"   Hybrid: {results['Hybrid']:.2f}% ({time.time()-start:.1f}s)")
    
    # 5. Stacking
    logger.info("\n[5/5] Stacking (RF+GB → HyperNet)...")
    start = time.time()
    with Pool(n_processes) as pool:
        r = list(pool.imap(process_fold_stacking, fold_datas, chunksize=10))
    results['Stacking'] = accuracy_score([i for x in r for i in x['y_true']], [i for x in r for i in x['y_pred']]) * 100
    logger.info(f"   Stacking: {results['Stacking']:.2f}% ({time.time()-start:.1f}s)")
    
    logger.info("\n" + "=" * 70)
    logger.info("[结果对比]")
    logger.info("=" * 70)
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for i, (name, acc) in enumerate(sorted_results):
        marker = "🏆" if i == 0 else f"{i+1:2d}"
        logger.info(f"{marker}. {name:15s}: {acc:.2f}%")
    logger.info("=" * 70)
    
    with open(OUTPUT_DIR / f'24_hybrid_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
