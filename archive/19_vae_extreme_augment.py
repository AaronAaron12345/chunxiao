#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
19_vae_extreme_augment.py - 极端数据增强 + 简单神经网络
========================================================
尝试更激进的方法：
1. 100倍数据增强
2. SMOTE + VAE组合
3. 最简单的单层网络
4. 多次运行取最优

运行: python 19_vae_extreme_augment.py
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
        self.fc_mu = nn.Linear(32, 8)
        self.fc_var = nn.Linear(32, 8)
        self.decoder = nn.Sequential(
            nn.Linear(8, 32), nn.LeakyReLU(0.2),
            nn.Linear(32, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, input_dim)
        )
    
    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = self.fc_mu(h), self.fc_var(h)
        z = mu + torch.exp(0.5 * log_var) * torch.randn_like(log_var)
        return self.decoder(z), mu, log_var


def smote_oversample(X, y, k=3):
    """SMOTE过采样"""
    X_new, y_new = [X], [y]
    
    for cls in np.unique(y):
        X_cls = X[y == cls]
        if len(X_cls) < 2:
            continue
        
        # 对每个样本，生成k个新样本
        for i in range(len(X_cls)):
            # 找最近的邻居
            dists = np.linalg.norm(X_cls - X_cls[i], axis=1)
            dists[i] = np.inf
            nearest = np.argsort(dists)[:min(k, len(X_cls)-1)]
            
            for j in nearest:
                # 在两点之间随机插值
                alpha = np.random.random()
                new_sample = X_cls[i] + alpha * (X_cls[j] - X_cls[i])
                X_new.append(new_sample.reshape(1, -1))
                y_new.append(np.array([cls]))
    
    return np.vstack(X_new), np.hstack(y_new)


def extreme_vae_augment(X_train, y_train, aug_factor=50):
    """极端VAE增强 - 生成大量样本"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X_train.shape[1]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # 先用SMOTE扩展
    X_smote, y_smote = smote_oversample(X_scaled, y_train, k=3)
    
    X_aug_list = [X_scaled]
    y_aug_list = [y_train]
    
    for cls in np.unique(y_train):
        X_cls = X_smote[y_smote == cls]
        
        vae = VAE(input_dim).to(device)
        optimizer = torch.optim.Adam(vae.parameters(), lr=0.01)
        X_tensor = torch.FloatTensor(X_cls).to(device)
        
        vae.train()
        for _ in range(60):
            optimizer.zero_grad()
            recon, mu, log_var = vae(X_tensor)
            loss = nn.MSELoss()(recon, X_tensor) + 0.005 * (-0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp()))
            loss.backward()
            optimizer.step()
        
        vae.eval()
        with torch.no_grad():
            # 1. 重建插值 - 更多点
            recon = vae(X_tensor)[0].cpu().numpy()
            for alpha in np.linspace(0.05, 0.95, 15):
                X_aug_list.append(alpha * X_cls + (1 - alpha) * recon)
                y_aug_list.append(np.full(len(X_cls), cls))
            
            # 2. 潜在空间采样 - 更多样本
            mu, log_var = vae.fc_mu(vae.encoder(X_tensor)), vae.fc_var(vae.encoder(X_tensor))
            for _ in range(aug_factor // 20):
                z = mu + torch.exp(0.5 * log_var) * torch.randn_like(log_var) * 0.3
                new_samples = vae.decoder(z).cpu().numpy()
                X_aug_list.append(new_samples)
                y_aug_list.append(np.full(len(X_cls), cls))
            
            # 3. 噪声增强
            for noise_level in [0.05, 0.1, 0.15]:
                noisy = X_cls + np.random.randn(*X_cls.shape) * noise_level
                X_aug_list.append(noisy)
                y_aug_list.append(np.full(len(X_cls), cls))
    
    return np.vstack(X_aug_list), np.hstack(y_aug_list), scaler


class LinearClassifier(nn.Module):
    """最简单的线性分类器"""
    def __init__(self, input_dim, n_classes):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, n_classes)
    
    def forward(self, x):
        return self.linear(x)


class TwoLayerNet(nn.Module):
    """两层网络"""
    def __init__(self, input_dim, n_classes):
        super(TwoLayerNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, n_classes)
        )
    
    def forward(self, x):
        return self.net(x)


def train_and_predict(model, X_train, y_train, X_test, epochs=100, lr=0.05):
    """训练并预测"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    X_tensor = torch.FloatTensor(X_train).to(device)
    y_tensor = torch.LongTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X_tensor), y_tensor)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        return model(X_test_tensor).argmax(dim=1).cpu().numpy()


def process_fold_rf(args):
    fold_idx, X_train, y_train, X_test, y_test = args
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    rf.fit(X_train_s, y_train)
    return {'y_true': y_test.tolist(), 'y_pred': rf.predict(X_test_s).tolist()}


def process_fold_vae_rf(args):
    fold_idx, X_train, y_train, X_test, y_test = args
    X_aug, y_aug, scaler = extreme_vae_augment(X_train, y_train, aug_factor=50)
    X_test_s = scaler.transform(X_test)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    rf.fit(X_aug, y_aug)
    return {'y_true': y_test.tolist(), 'y_pred': rf.predict(X_test_s).tolist()}


def process_fold_vae_linear(args):
    """VAE + 线性分类器"""
    fold_idx, X_train, y_train, X_test, y_test = args
    X_aug, y_aug, scaler = extreme_vae_augment(X_train, y_train, aug_factor=50)
    X_test_s = scaler.transform(X_test)
    
    n_classes = len(np.unique(y_train))
    model = LinearClassifier(X_train.shape[1], n_classes)
    y_pred = train_and_predict(model, X_aug, y_aug, X_test_s, epochs=100, lr=0.05)
    
    return {'y_true': y_test.tolist(), 'y_pred': y_pred.tolist()}


def process_fold_vae_twolayer(args):
    """VAE + 两层网络"""
    fold_idx, X_train, y_train, X_test, y_test = args
    X_aug, y_aug, scaler = extreme_vae_augment(X_train, y_train, aug_factor=50)
    X_test_s = scaler.transform(X_test)
    
    n_classes = len(np.unique(y_train))
    model = TwoLayerNet(X_train.shape[1], n_classes)
    y_pred = train_and_predict(model, X_aug, y_aug, X_test_s, epochs=100, lr=0.03)
    
    return {'y_true': y_test.tolist(), 'y_pred': y_pred.tolist()}


def process_fold_vae_ensemble_vote(args):
    """VAE + 多模型投票"""
    fold_idx, X_train, y_train, X_test, y_test = args
    X_aug, y_aug, scaler = extreme_vae_augment(X_train, y_train, aug_factor=50)
    X_test_s = scaler.transform(X_test)
    
    n_classes = len(np.unique(y_train))
    input_dim = X_train.shape[1]
    
    predictions = []
    
    # 多次运行线性模型
    for seed in range(5):
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = LinearClassifier(input_dim, n_classes)
        pred = train_and_predict(model, X_aug, y_aug, X_test_s, epochs=80, lr=0.03 + seed*0.01)
        predictions.append(pred)
    
    # 多次运行两层网络
    for seed in range(5):
        np.random.seed(seed + 100)
        torch.manual_seed(seed + 100)
        model = TwoLayerNet(input_dim, n_classes)
        pred = train_and_predict(model, X_aug, y_aug, X_test_s, epochs=80, lr=0.02 + seed*0.005)
        predictions.append(pred)
    
    # 投票
    predictions = np.array(predictions)
    y_pred = []
    for i in range(len(X_test_s)):
        votes = predictions[:, i]
        y_pred.append(np.bincount(votes.astype(int)).argmax())
    
    return {'y_true': y_test.tolist(), 'y_pred': y_pred}


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR / f'19_extreme_{timestamp}.log'
    
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
    logger.info("19_vae_extreme_augment.py - 极端数据增强实验")
    logger.info("=" * 70)
    
    data_path = SCRIPT_DIR / 'data' / 'Data_for_Jinming.csv'
    df = pd.read_csv(data_path)
    X = df[['LAA', 'Glutamate', 'Choline', 'Sarcosine']].values.astype(np.float32)
    y = LabelEncoder().fit_transform(df['Group'].values)
    
    n_samples = len(X)
    logger.info(f"数据: {n_samples} 样本")
    
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
    
    # 2. VAE+RF
    logger.info("\n[2/5] VAE(极端)+RF...")
    start = time.time()
    with Pool(n_processes) as pool:
        r = list(pool.imap(process_fold_vae_rf, fold_datas, chunksize=10))
    results['VAE+RF'] = accuracy_score([i for x in r for i in x['y_true']], [i for x in r for i in x['y_pred']]) * 100
    logger.info(f"   VAE+RF: {results['VAE+RF']:.2f}% ({time.time()-start:.1f}s)")
    
    # 3. VAE+线性
    logger.info("\n[3/5] VAE(极端)+线性分类器...")
    start = time.time()
    with Pool(n_processes) as pool:
        r = list(pool.imap(process_fold_vae_linear, fold_datas, chunksize=10))
    results['VAE+Linear'] = accuracy_score([i for x in r for i in x['y_true']], [i for x in r for i in x['y_pred']]) * 100
    logger.info(f"   VAE+Linear: {results['VAE+Linear']:.2f}% ({time.time()-start:.1f}s)")
    
    # 4. VAE+两层
    logger.info("\n[4/5] VAE(极端)+两层网络...")
    start = time.time()
    with Pool(n_processes) as pool:
        r = list(pool.imap(process_fold_vae_twolayer, fold_datas, chunksize=10))
    results['VAE+TwoLayer'] = accuracy_score([i for x in r for i in x['y_true']], [i for x in r for i in x['y_pred']]) * 100
    logger.info(f"   VAE+TwoLayer: {results['VAE+TwoLayer']:.2f}% ({time.time()-start:.1f}s)")
    
    # 5. VAE+多模型投票
    logger.info("\n[5/5] VAE(极端)+多模型投票...")
    start = time.time()
    with Pool(n_processes) as pool:
        r = list(pool.imap(process_fold_vae_ensemble_vote, fold_datas, chunksize=10))
    results['VAE+Vote'] = accuracy_score([i for x in r for i in x['y_true']], [i for x in r for i in x['y_pred']]) * 100
    logger.info(f"   VAE+Vote: {results['VAE+Vote']:.2f}% ({time.time()-start:.1f}s)")
    
    logger.info("\n" + "=" * 70)
    logger.info("[结果对比]")
    logger.info("=" * 70)
    for name, acc in results.items():
        marker = "🏆" if acc == max(results.values()) else "  "
        logger.info(f"{marker} {name:15s}: {acc:.2f}%")
    logger.info("=" * 70)
    
    result_file = OUTPUT_DIR / f'19_extreme_{timestamp}.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"保存: {result_file}")


if __name__ == '__main__':
    main()
