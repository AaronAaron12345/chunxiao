#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
17_vae_hypernet_improved.py - 改进版VAE+HyperNet，目标达到RF水平
==================================================================
核心改进：
1. 更简单的网络（防止过拟合）
2. 更大的数据增强倍数
3. 早停机制
4. 多次运行取最优（类似RF的随机性）

运行: python 17_vae_hypernet_improved.py
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
    """VAE - 更大的增强能力"""
    def __init__(self, input_dim, hidden_dim=128, latent_dim=16):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LeakyReLU(0.2)
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_var = nn.Linear(hidden_dim // 2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = self.fc_mu(h), self.fc_var(h)
        std = torch.exp(0.5 * log_var)
        z = mu + std * torch.randn_like(std)
        return self.decoder(z), mu, log_var
    
    def sample(self, mu, log_var, n=1):
        """从潜在空间采样"""
        std = torch.exp(0.5 * log_var)
        samples = []
        for _ in range(n):
            z = mu + std * torch.randn_like(std)
            samples.append(self.decoder(z))
        return torch.cat(samples, dim=0)


def vae_augment_v2(X_train, y_train, aug_factor=10):
    """
    改进的VAE增强 - 生成更多样本
    aug_factor: 增强倍数，最终数据量 = 原始数据 * aug_factor
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X_train.shape[1]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    X_aug_list = [X_scaled]
    y_aug_list = [y_train]
    
    for cls in np.unique(y_train):
        X_cls = X_scaled[y_train == cls]
        if len(X_cls) < 2:
            continue
        
        vae = VAE(input_dim, hidden_dim=128, latent_dim=16).to(device)
        optimizer = torch.optim.Adam(vae.parameters(), lr=0.003, weight_decay=1e-5)
        X_tensor = torch.FloatTensor(X_cls).to(device)
        
        # 训练VAE
        vae.train()
        for _ in range(50):
            optimizer.zero_grad()
            recon, mu, log_var = vae(X_tensor)
            recon_loss = nn.MSELoss()(recon, X_tensor)
            kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
            loss = recon_loss + 0.005 * kl_loss
            loss.backward()
            optimizer.step()
        
        vae.eval()
        with torch.no_grad():
            mu, log_var = vae.encoder(X_tensor), vae.fc_var(vae.encoder(X_tensor))
            mu = vae.fc_mu(vae.encoder(X_tensor))
            log_var = vae.fc_var(vae.encoder(X_tensor))
            
            # 1. 重建插值
            recon = vae(X_tensor)[0].cpu().numpy()
            for alpha in np.linspace(0.1, 0.9, 8):
                aug = alpha * X_cls + (1 - alpha) * recon
                X_aug_list.append(aug)
                y_aug_list.append(np.full(len(X_cls), cls))
            
            # 2. 潜在空间采样
            n_samples_per_point = aug_factor // 10
            for _ in range(n_samples_per_point):
                z_noise = mu + torch.exp(0.5 * log_var) * torch.randn_like(log_var) * 0.5
                new_samples = vae.decoder(z_noise).cpu().numpy()
                X_aug_list.append(new_samples)
                y_aug_list.append(np.full(len(X_cls), cls))
    
    X_aug = np.vstack(X_aug_list)
    y_aug = np.hstack(y_aug_list)
    
    return X_aug, y_aug, scaler


class TinyClassifier(nn.Module):
    """超小型分类器 - 防止过拟合"""
    def __init__(self, input_dim, n_classes):
        super(TinyClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, n_classes)
        )
    
    def forward(self, x):
        return self.net(x)


class TinyEnsemble:
    """超小型集成 - 多个小网络投票"""
    def __init__(self, input_dim, n_classes, n_estimators=10):
        self.n_estimators = n_estimators
        self.models = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for _ in range(n_estimators):
            model = TinyClassifier(input_dim, n_classes).to(self.device)
            self.models.append(model)
    
    def fit(self, X, y, epochs=30):
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        criterion = nn.CrossEntropyLoss()
        
        for i, model in enumerate(self.models):
            # Bootstrap采样
            np.random.seed(i * 100 + int(time.time()) % 1000)
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_boot = X_tensor[indices]
            y_boot = y_tensor[indices]
            
            optimizer = optim.Adam(model.parameters(), lr=0.02, weight_decay=0.01)
            model.train()
            
            for _ in range(epochs):
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
                probs = torch.softmax(model(X_tensor), dim=1)
                all_probs.append(probs)
        
        avg_probs = torch.stack(all_probs).mean(dim=0)
        return avg_probs.argmax(dim=1).cpu().numpy()


def process_fold_rf(args):
    """RF基线"""
    fold_idx, X_train, y_train, X_test, y_test = args
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    rf.fit(X_train_s, y_train)
    y_pred = rf.predict(X_test_s)
    
    return {'y_true': y_test.tolist(), 'y_pred': y_pred.tolist()}


def process_fold_vae_rf(args):
    """VAE + RF"""
    fold_idx, X_train, y_train, X_test, y_test = args
    
    X_aug, y_aug, scaler = vae_augment_v2(X_train, y_train, aug_factor=10)
    X_test_s = scaler.transform(X_test)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    rf.fit(X_aug, y_aug)
    y_pred = rf.predict(X_test_s)
    
    return {'y_true': y_test.tolist(), 'y_pred': y_pred.tolist()}


def process_fold_vae_ensemble(args):
    """VAE + 超小型神经网络集成"""
    fold_idx, X_train, y_train, X_test, y_test = args
    
    X_aug, y_aug, scaler = vae_augment_v2(X_train, y_train, aug_factor=10)
    X_test_s = scaler.transform(X_test)
    
    n_classes = len(np.unique(y_train))
    ensemble = TinyEnsemble(X_train.shape[1], n_classes, n_estimators=10)
    ensemble.fit(X_aug, y_aug, epochs=30)
    y_pred = ensemble.predict(X_test_s)
    
    return {'y_true': y_test.tolist(), 'y_pred': y_pred.tolist()}


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR / f'17_improved_{timestamp}.log'
    
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
    logger.info("17_vae_hypernet_improved.py - 改进版：目标达到RF水平")
    logger.info("=" * 70)
    
    data_path = SCRIPT_DIR / 'data' / 'Data_for_Jinming.csv'
    df = pd.read_csv(data_path)
    feature_cols = ['LAA', 'Glutamate', 'Choline', 'Sarcosine']
    X = df[feature_cols].values.astype(np.float32)
    y_raw = df['Group'].values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    n_samples = len(X)
    logger.info(f"数据: {n_samples} 样本, {X.shape[1]} 特征")
    
    all_indices = np.arange(n_samples)
    test_combos = list(combinations(all_indices, 2))
    n_folds = len(test_combos)
    logger.info(f"Leave-2-Out: {n_folds} folds")
    
    fold_datas = []
    for fold_idx, test_idx in enumerate(test_combos):
        test_idx = np.array(test_idx)
        train_idx = np.setdiff1d(all_indices, test_idx)
        fold_datas.append((fold_idx, X[train_idx], y[train_idx], X[test_idx], y[test_idx]))
    
    n_processes = min(cpu_count(), 64)
    logger.info(f"使用 {n_processes} 进程")
    
    # 1. RF
    logger.info("\n[1/3] RF基线...")
    start = time.time()
    with Pool(n_processes) as pool:
        results = list(pool.imap(process_fold_rf, fold_datas, chunksize=10))
    all_true = [i for r in results for i in r['y_true']]
    all_pred = [i for r in results for i in r['y_pred']]
    rf_acc = accuracy_score(all_true, all_pred) * 100
    logger.info(f"   RF: {rf_acc:.2f}% ({time.time()-start:.1f}s)")
    
    # 2. VAE+RF
    logger.info("\n[2/3] VAE+RF...")
    start = time.time()
    with Pool(n_processes) as pool:
        results = list(pool.imap(process_fold_vae_rf, fold_datas, chunksize=10))
    all_true = [i for r in results for i in r['y_true']]
    all_pred = [i for r in results for i in r['y_pred']]
    vae_rf_acc = accuracy_score(all_true, all_pred) * 100
    logger.info(f"   VAE+RF: {vae_rf_acc:.2f}% ({time.time()-start:.1f}s)")
    
    # 3. VAE+小网络集成
    logger.info("\n[3/3] VAE+TinyEnsemble...")
    start = time.time()
    with Pool(n_processes) as pool:
        results = list(pool.imap(process_fold_vae_ensemble, fold_datas, chunksize=10))
    all_true = [i for r in results for i in r['y_true']]
    all_pred = [i for r in results for i in r['y_pred']]
    vae_nn_acc = accuracy_score(all_true, all_pred) * 100
    logger.info(f"   VAE+TinyEnsemble: {vae_nn_acc:.2f}% ({time.time()-start:.1f}s)")
    
    logger.info("\n" + "=" * 70)
    logger.info("[结果对比]")
    logger.info("=" * 70)
    logger.info(f"  RF:              {rf_acc:.2f}%")
    logger.info(f"  VAE+RF:          {vae_rf_acc:.2f}%")
    logger.info(f"  VAE+TinyEnsemble:{vae_nn_acc:.2f}%")
    logger.info("=" * 70)
    
    result_file = OUTPUT_DIR / f'17_improved_{timestamp}.json'
    with open(result_file, 'w') as f:
        json.dump({
            'RF': rf_acc, 'VAE+RF': vae_rf_acc, 'VAE+TinyEnsemble': vae_nn_acc
        }, f, indent=2)
    logger.info(f"保存: {result_file}")


if __name__ == '__main__':
    main()
