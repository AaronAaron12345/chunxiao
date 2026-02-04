#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
23_vae_hypernet_simple.py - 简化版VAE+HyperNet
===============================================
问题分析：之前HyperNet太复杂，容易过拟合
新思路：
1. 最简单的HyperNet：直接生成线性分类器权重
2. 用大量VAE增强数据
3. 不用蒸馏，直接从数据学习
4. 强正则化 + 早停

运行: python 23_vae_hypernet_simple.py
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
    def __init__(self, input_dim, latent_dim=4):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(16, latent_dim)
        self.fc_var = nn.Linear(16, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, input_dim)
        )
    
    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = self.fc_mu(h), self.fc_var(h)
        z = mu + torch.exp(0.5 * log_var) * torch.randn_like(log_var)
        return self.decoder(z), mu, log_var


class SimpleHyperNet(nn.Module):
    """
    最简单的HyperNet：
    - 输入: 一批样本的均值特征
    - 输出: 线性分类器的权重
    """
    def __init__(self, input_dim, n_classes):
        super(SimpleHyperNet, self).__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        
        # HyperNet非常简单
        self.hypernet = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        
        # 生成权重
        self.weight_gen = nn.Linear(16, input_dim * n_classes)
        self.bias_gen = nn.Linear(16, n_classes)
        
        # 初始化
        nn.init.xavier_uniform_(self.weight_gen.weight, gain=0.1)
        nn.init.xavier_uniform_(self.bias_gen.weight, gain=0.1)
    
    def forward(self, x, context):
        """
        x: (batch, input_dim) 要分类的样本
        context: (input_dim,) 上下文（训练数据均值）
        """
        # 用上下文生成权重
        h = self.hypernet(context)
        W = self.weight_gen(h).view(self.n_classes, self.input_dim)
        b = self.bias_gen(h)
        
        # 线性分类
        logits = F.linear(x, W, b)
        return logits


def vae_augment(X_train, y_train, aug_factor=20):
    """VAE数据增强"""
    input_dim = X_train.shape[1]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    X_aug_list = [X_scaled]
    y_aug_list = [y_train]
    
    for cls in np.unique(y_train):
        X_cls = X_scaled[y_train == cls]
        if len(X_cls) < 2:
            continue
        
        vae = VAE(input_dim)
        optimizer = torch.optim.Adam(vae.parameters(), lr=0.02)
        X_tensor = torch.FloatTensor(X_cls)
        
        for _ in range(50):
            optimizer.zero_grad()
            recon, mu, log_var = vae(X_tensor)
            loss = nn.MSELoss()(recon, X_tensor) + 0.01 * (-0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp()))
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            recon = vae(X_tensor)[0].numpy()
            for alpha in np.linspace(0.2, 0.8, aug_factor):
                X_aug_list.append(alpha * X_cls + (1 - alpha) * recon)
                y_aug_list.append(np.full(len(X_cls), cls))
    
    return np.vstack(X_aug_list), np.hstack(y_aug_list), scaler


def train_simple_hypernet(X_aug, y_aug, X_test, n_classes):
    """训练简单HyperNet"""
    input_dim = X_aug.shape[1]
    
    # 上下文 = 训练数据均值
    context = torch.FloatTensor(X_aug.mean(axis=0))
    
    model = SimpleHyperNet(input_dim, n_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.05, weight_decay=0.1)
    
    X_tensor = torch.FloatTensor(X_aug)
    y_tensor = torch.LongTensor(y_aug)
    
    best_acc = 0
    best_state = None
    patience = 0
    
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        logits = model(X_tensor, context)
        loss = F.cross_entropy(logits, y_tensor)
        loss.backward()
        optimizer.step()
        
        # 验证
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == y_tensor).float().mean().item()
            if acc > best_acc:
                best_acc = acc
                best_state = model.state_dict().copy()
                patience = 0
            else:
                patience += 1
            
            if patience > 30:
                break
    
    # 预测
    if best_state:
        model.load_state_dict(best_state)
    
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        logits = model(X_test_tensor, context)
        return logits.argmax(dim=1).numpy()


class DirectClassifier(nn.Module):
    """直接的神经网络分类器（对比）"""
    def __init__(self, input_dim, n_classes):
        super(DirectClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(8, n_classes)
        )
    
    def forward(self, x):
        return self.net(x)


def train_direct_classifier(X_aug, y_aug, X_test, n_classes):
    """训练直接分类器"""
    input_dim = X_aug.shape[1]
    
    model = DirectClassifier(input_dim, n_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.05, weight_decay=0.1)
    
    X_tensor = torch.FloatTensor(X_aug)
    y_tensor = torch.LongTensor(y_aug)
    
    best_acc = 0
    best_state = None
    
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(X_tensor), y_tensor)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            acc = (model(X_tensor).argmax(dim=1) == y_tensor).float().mean().item()
            if acc > best_acc:
                best_acc = acc
                best_state = model.state_dict().copy()
    
    if best_state:
        model.load_state_dict(best_state)
    
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X_test)).argmax(dim=1).numpy()


def process_fold_rf(args):
    fold_idx, X_train, y_train, X_test, y_test = args
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    rf.fit(X_train_s, y_train)
    return {'y_true': y_test.tolist(), 'y_pred': rf.predict(X_test_s).tolist()}


def process_fold_vae_hypernet(args):
    fold_idx, X_train, y_train, X_test, y_test = args
    n_classes = len(np.unique(y_train))
    
    X_aug, y_aug, scaler = vae_augment(X_train, y_train, aug_factor=20)
    X_test_s = scaler.transform(X_test)
    
    y_pred = train_simple_hypernet(X_aug, y_aug, X_test_s, n_classes)
    return {'y_true': y_test.tolist(), 'y_pred': y_pred.tolist()}


def process_fold_vae_direct(args):
    fold_idx, X_train, y_train, X_test, y_test = args
    n_classes = len(np.unique(y_train))
    
    X_aug, y_aug, scaler = vae_augment(X_train, y_train, aug_factor=20)
    X_test_s = scaler.transform(X_test)
    
    y_pred = train_direct_classifier(X_aug, y_aug, X_test_s, n_classes)
    return {'y_true': y_test.tolist(), 'y_pred': y_pred.tolist()}


def process_fold_vae_rf(args):
    fold_idx, X_train, y_train, X_test, y_test = args
    
    X_aug, y_aug, scaler = vae_augment(X_train, y_train, aug_factor=20)
    X_test_s = scaler.transform(X_test)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    rf.fit(X_aug, y_aug)
    return {'y_true': y_test.tolist(), 'y_pred': rf.predict(X_test_s).tolist()}


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR / f'23_simple_hypernet_{timestamp}.log'
    
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
    logger.info("23_vae_hypernet_simple.py - 简化版HyperNet")
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
    logger.info("\n[1/4] RF...")
    start = time.time()
    with Pool(n_processes) as pool:
        r = list(pool.imap(process_fold_rf, fold_datas, chunksize=10))
    results['RF'] = accuracy_score([i for x in r for i in x['y_true']], [i for x in r for i in x['y_pred']]) * 100
    logger.info(f"   RF: {results['RF']:.2f}% ({time.time()-start:.1f}s)")
    
    # 2. VAE+RF
    logger.info("\n[2/4] VAE+RF...")
    start = time.time()
    with Pool(n_processes) as pool:
        r = list(pool.imap(process_fold_vae_rf, fold_datas, chunksize=10))
    results['VAE+RF'] = accuracy_score([i for x in r for i in x['y_true']], [i for x in r for i in x['y_pred']]) * 100
    logger.info(f"   VAE+RF: {results['VAE+RF']:.2f}% ({time.time()-start:.1f}s)")
    
    # 3. VAE+DirectNN
    logger.info("\n[3/4] VAE+DirectNN...")
    start = time.time()
    with Pool(n_processes) as pool:
        r = list(pool.imap(process_fold_vae_direct, fold_datas, chunksize=10))
    results['VAE+DirectNN'] = accuracy_score([i for x in r for i in x['y_true']], [i for x in r for i in x['y_pred']]) * 100
    logger.info(f"   VAE+DirectNN: {results['VAE+DirectNN']:.2f}% ({time.time()-start:.1f}s)")
    
    # 4. VAE+HyperNet
    logger.info("\n[4/4] VAE+SimpleHyperNet...")
    start = time.time()
    with Pool(n_processes) as pool:
        r = list(pool.imap(process_fold_vae_hypernet, fold_datas, chunksize=10))
    results['VAE+HyperNet'] = accuracy_score([i for x in r for i in x['y_true']], [i for x in r for i in x['y_pred']]) * 100
    logger.info(f"   VAE+HyperNet: {results['VAE+HyperNet']:.2f}% ({time.time()-start:.1f}s)")
    
    # 结果
    logger.info("\n" + "=" * 70)
    logger.info("[结果对比]")
    logger.info("=" * 70)
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        marker = "🏆" if acc == max(results.values()) else "  "
        logger.info(f"{marker} {name:15s}: {acc:.2f}%")
    logger.info("=" * 70)
    
    result_file = OUTPUT_DIR / f'23_simple_hypernet_{timestamp}.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
