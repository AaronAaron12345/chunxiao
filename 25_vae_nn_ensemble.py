#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
25_vae_nn_ensemble.py - VAE + 神经网络集成 (模仿RF)
===================================================
核心思路：
1. 使用论文风格的VAE增强
2. 训练多个不同初始化的小型神经网络（像RF的多棵树）
3. Bagging采样训练每个网络
4. 投票预测

目标：让神经网络方法达到80%

运行: python 25_vae_nn_ensemble.py
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
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).parent
LOG_DIR = SCRIPT_DIR / 'logs'
OUTPUT_DIR = SCRIPT_DIR / 'output'
LOG_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


class SimpleVAE(nn.Module):
    """简单VAE"""
    def __init__(self, input_dim, latent_dim=4):
        super(SimpleVAE, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU())
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_var = nn.Linear(32, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, input_dim)
        )
    
    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = self.fc_mu(h), self.fc_var(h)
        z = mu + torch.exp(0.5 * log_var) * torch.randn_like(log_var)
        return self.decoder(z), mu, log_var


def vae_augment(X_train, y_train, num_interpolation=5):
    """VAE增强"""
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_train).astype(np.float32)
    
    input_dim = X_scaled.shape[1]
    aug_data, aug_labels = [], []
    
    for cls in np.unique(y_train):
        X_cls = X_scaled[y_train == cls]
        
        vae = SimpleVAE(input_dim, latent_dim=4)
        optimizer = torch.optim.Adam(vae.parameters(), lr=0.01)
        X_tensor = torch.FloatTensor(X_cls)
        
        for _ in range(50):
            optimizer.zero_grad()
            recon, mu, log_var = vae(X_tensor)
            loss = F.mse_loss(recon, X_tensor) + 0.01 * (-0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp()))
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            recon = vae(X_tensor)[0].numpy()
            for orig, dec in zip(X_cls, recon):
                for alpha in np.linspace(0.1, 0.9, num_interpolation):
                    aug_data.append(alpha * orig + (1-alpha) * dec)
                    aug_labels.append(cls)
    
    return np.array(aug_data, dtype=np.float32), np.array(aug_labels), scaler


class TinyNet(nn.Module):
    """极简神经网络 - 模仿决策树"""
    def __init__(self, input_dim, n_classes, hidden=8):
        super(TinyNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes)
        )
    
    def forward(self, x):
        return self.net(x)


class NNEnsemble:
    """神经网络集成 - 模仿随机森林"""
    def __init__(self, input_dim, n_classes, n_estimators=20, hidden=8):
        self.n_estimators = n_estimators
        self.models = []
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.hidden = hidden
    
    def fit(self, X, y, epochs=100, lr=0.05):
        """训练集成 - 每个模型用bootstrapped样本"""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        n_samples = len(X)
        
        for i in range(self.n_estimators):
            torch.manual_seed(i * 42)
            np.random.seed(i * 42)
            
            # Bootstrap采样
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X_tensor[indices]
            y_boot = y_tensor[indices]
            
            # 创建并训练模型
            model = TinyNet(self.input_dim, self.n_classes, self.hidden)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
            
            model.train()
            for _ in range(epochs):
                optimizer.zero_grad()
                loss = F.cross_entropy(model(X_boot), y_boot)
                loss.backward()
                optimizer.step()
            
            model.eval()
            self.models.append(model)
    
    def predict(self, X):
        """投票预测"""
        X_tensor = torch.FloatTensor(X)
        all_preds = []
        
        for model in self.models:
            with torch.no_grad():
                preds = model(X_tensor).argmax(dim=1).numpy()
                all_preds.append(preds)
        
        all_preds = np.array(all_preds)  # (n_estimators, n_samples)
        
        # 投票
        final_preds = []
        for i in range(len(X)):
            votes = all_preds[:, i]
            final_preds.append(np.bincount(votes, minlength=self.n_classes).argmax())
        
        return np.array(final_preds)
    
    def predict_proba_avg(self, X):
        """平均概率预测"""
        X_tensor = torch.FloatTensor(X)
        all_probs = []
        
        for model in self.models:
            with torch.no_grad():
                probs = F.softmax(model(X_tensor), dim=1).numpy()
                all_probs.append(probs)
        
        avg_probs = np.mean(all_probs, axis=0)
        return avg_probs.argmax(axis=1)


def process_fold_rf(args):
    fold_idx, X_train, y_train, X_test, y_test = args
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    rf.fit(X_train_s, y_train)
    return {'y_true': y_test.tolist(), 'y_pred': rf.predict(X_test_s).tolist()}


def process_fold_vae_rf(args):
    fold_idx, X_train, y_train, X_test, y_test = args
    
    X_aug, y_aug, scaler = vae_augment(X_train, y_train, num_interpolation=5)
    X_test_s = scaler.transform(X_test)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    rf.fit(X_aug, y_aug)
    return {'y_true': y_test.tolist(), 'y_pred': rf.predict(X_test_s).tolist()}


def process_fold_vae_nn_ensemble_vote(args):
    """VAE + NN集成 (投票)"""
    fold_idx, X_train, y_train, X_test, y_test = args
    n_classes = len(np.unique(y_train))
    
    X_aug, y_aug, scaler = vae_augment(X_train, y_train, num_interpolation=5)
    X_test_s = scaler.transform(X_test)
    
    ensemble = NNEnsemble(X_aug.shape[1], n_classes, n_estimators=30, hidden=8)
    ensemble.fit(X_aug, y_aug, epochs=80, lr=0.05)
    y_pred = ensemble.predict(X_test_s)
    
    return {'y_true': y_test.tolist(), 'y_pred': y_pred.tolist()}


def process_fold_vae_nn_ensemble_avg(args):
    """VAE + NN集成 (平均概率)"""
    fold_idx, X_train, y_train, X_test, y_test = args
    n_classes = len(np.unique(y_train))
    
    X_aug, y_aug, scaler = vae_augment(X_train, y_train, num_interpolation=5)
    X_test_s = scaler.transform(X_test)
    
    ensemble = NNEnsemble(X_aug.shape[1], n_classes, n_estimators=30, hidden=8)
    ensemble.fit(X_aug, y_aug, epochs=80, lr=0.05)
    y_pred = ensemble.predict_proba_avg(X_test_s)
    
    return {'y_true': y_test.tolist(), 'y_pred': y_pred.tolist()}


def process_fold_nn_ensemble_no_vae(args):
    """NN集成 (无VAE)"""
    fold_idx, X_train, y_train, X_test, y_test = args
    n_classes = len(np.unique(y_train))
    
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)
    
    ensemble = NNEnsemble(X_train_s.shape[1], n_classes, n_estimators=30, hidden=8)
    ensemble.fit(X_train_s, y_train, epochs=80, lr=0.05)
    y_pred = ensemble.predict(X_test_s)
    
    return {'y_true': y_test.tolist(), 'y_pred': y_pred.tolist()}


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR / f'25_nn_ensemble_{timestamp}.log'
    
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
    logger.info("25_vae_nn_ensemble.py - 神经网络集成 (模仿RF)")
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
    logger.info("\n[1/5] RF...")
    start = time.time()
    with Pool(n_processes) as pool:
        r = list(pool.imap(process_fold_rf, fold_datas, chunksize=10))
    results['RF'] = accuracy_score([i for x in r for i in x['y_true']], [i for x in r for i in x['y_pred']]) * 100
    logger.info(f"   RF: {results['RF']:.2f}% ({time.time()-start:.1f}s)")
    
    # 2. VAE+RF
    logger.info("\n[2/5] VAE+RF...")
    start = time.time()
    with Pool(n_processes) as pool:
        r = list(pool.imap(process_fold_vae_rf, fold_datas, chunksize=10))
    results['VAE+RF'] = accuracy_score([i for x in r for i in x['y_true']], [i for x in r for i in x['y_pred']]) * 100
    logger.info(f"   VAE+RF: {results['VAE+RF']:.2f}% ({time.time()-start:.1f}s)")
    
    # 3. NN集成 (无VAE)
    logger.info("\n[3/5] NN集成 (无VAE)...")
    start = time.time()
    with Pool(n_processes) as pool:
        r = list(pool.imap(process_fold_nn_ensemble_no_vae, fold_datas, chunksize=10))
    results['NNEnsemble'] = accuracy_score([i for x in r for i in x['y_true']], [i for x in r for i in x['y_pred']]) * 100
    logger.info(f"   NNEnsemble: {results['NNEnsemble']:.2f}% ({time.time()-start:.1f}s)")
    
    # 4. VAE + NN集成 (投票)
    logger.info("\n[4/5] VAE+NNEnsemble (投票)...")
    start = time.time()
    with Pool(n_processes) as pool:
        r = list(pool.imap(process_fold_vae_nn_ensemble_vote, fold_datas, chunksize=10))
    results['VAE+NNEnsemble(Vote)'] = accuracy_score([i for x in r for i in x['y_true']], [i for x in r for i in x['y_pred']]) * 100
    logger.info(f"   VAE+NNEnsemble(Vote): {results['VAE+NNEnsemble(Vote)']:.2f}% ({time.time()-start:.1f}s)")
    
    # 5. VAE + NN集成 (平均)
    logger.info("\n[5/5] VAE+NNEnsemble (平均)...")
    start = time.time()
    with Pool(n_processes) as pool:
        r = list(pool.imap(process_fold_vae_nn_ensemble_avg, fold_datas, chunksize=10))
    results['VAE+NNEnsemble(Avg)'] = accuracy_score([i for x in r for i in x['y_true']], [i for x in r for i in x['y_pred']]) * 100
    logger.info(f"   VAE+NNEnsemble(Avg): {results['VAE+NNEnsemble(Avg)']:.2f}% ({time.time()-start:.1f}s)")
    
    # 结果
    logger.info("\n" + "=" * 70)
    logger.info("[结果对比]")
    logger.info("=" * 70)
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        marker = "🏆" if acc == max(results.values()) else "  "
        logger.info(f"{marker} {name:25s}: {acc:.2f}%")
    logger.info("=" * 70)
    
    result_file = OUTPUT_DIR / f'25_nn_ensemble_{timestamp}.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
