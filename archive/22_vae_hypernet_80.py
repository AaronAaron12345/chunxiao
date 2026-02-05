#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
22_vae_hypernet_80.py - VAE+HyperNet 目标80%准确率
===================================================
核心思路：
1. VAE 大量数据增强
2. HyperNet 生成分类器权重（模仿RF的多树思想）
3. 多个HyperNet集成投票
4. RF软标签蒸馏指导
5. 更强的正则化防止过拟合

运行: python 22_vae_hypernet_80.py
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
    """VAE用于数据增强"""
    def __init__(self, input_dim, latent_dim=8):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.LeakyReLU(0.2), nn.BatchNorm1d(32),
            nn.Linear(32, 16), nn.LeakyReLU(0.2)
        )
        self.fc_mu = nn.Linear(16, latent_dim)
        self.fc_var = nn.Linear(16, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.LeakyReLU(0.2),
            nn.Linear(16, 32), nn.LeakyReLU(0.2),
            nn.Linear(32, input_dim)
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, log_var, noise_scale=1.0):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std) * noise_scale
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x, noise_scale=1.0):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var, noise_scale)
        return self.decode(z), mu, log_var


class HyperNetwork(nn.Module):
    """
    HyperNetwork: 根据输入样本特征生成分类器权重
    模仿RF的思想：每棵树根据数据特征做决策
    """
    def __init__(self, input_dim, n_classes, hidden_dim=32):
        super(HyperNetwork, self).__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        
        # HyperNet: 输入特征 -> 生成分类器权重
        self.hypernet = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
        )
        
        # 生成分类器的权重和偏置
        self.weight_gen = nn.Linear(hidden_dim, input_dim * n_classes)
        self.bias_gen = nn.Linear(hidden_dim, n_classes)
    
    def forward(self, x):
        """
        x: (batch, input_dim)
        返回: (batch, n_classes) 分类logits
        """
        batch_size = x.shape[0]
        
        # 生成权重
        h = self.hypernet(x)
        weights = self.weight_gen(h).view(batch_size, self.n_classes, self.input_dim)
        biases = self.bias_gen(h)
        
        # 用生成的权重对输入分类
        # (batch, n_classes, input_dim) @ (batch, input_dim, 1) -> (batch, n_classes, 1)
        logits = torch.bmm(weights, x.unsqueeze(2)).squeeze(2) + biases
        
        return logits


class HyperNetEnsemble(nn.Module):
    """多个HyperNet集成"""
    def __init__(self, input_dim, n_classes, n_nets=5, hidden_dim=32):
        super(HyperNetEnsemble, self).__init__()
        self.nets = nn.ModuleList([
            HyperNetwork(input_dim, n_classes, hidden_dim) 
            for _ in range(n_nets)
        ])
    
    def forward(self, x):
        # 多个网络的logits平均
        logits_list = [net(x) for net in self.nets]
        return torch.stack(logits_list).mean(dim=0)
    
    def predict_vote(self, x):
        """投票预测"""
        preds = [net(x).argmax(dim=1) for net in self.nets]
        preds = torch.stack(preds, dim=1)  # (batch, n_nets)
        
        # 投票
        batch_size = x.shape[0]
        n_classes = self.nets[0].n_classes
        voted = []
        for i in range(batch_size):
            counts = torch.bincount(preds[i], minlength=n_classes)
            voted.append(counts.argmax().item())
        return voted


def vae_augment(X_train, y_train, aug_factor=30):
    """VAE数据增强"""
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
        
        vae = VAE(input_dim).to(device)
        optimizer = torch.optim.Adam(vae.parameters(), lr=0.01)
        X_tensor = torch.FloatTensor(X_cls).to(device)
        
        # 训练VAE
        vae.train()
        for _ in range(80):
            optimizer.zero_grad()
            recon, mu, log_var = vae(X_tensor)
            recon_loss = nn.MSELoss()(recon, X_tensor)
            kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
            loss = recon_loss + 0.01 * kl_loss
            loss.backward()
            optimizer.step()
        
        # 生成增强数据
        vae.eval()
        with torch.no_grad():
            mu, log_var = vae.encode(X_tensor)
            
            # 1. 插值增强
            recon = vae.decode(mu).cpu().numpy()
            for alpha in np.linspace(0.1, 0.9, aug_factor // 3):
                X_aug_list.append(alpha * X_cls + (1 - alpha) * recon)
                y_aug_list.append(np.full(len(X_cls), cls))
            
            # 2. 潜在空间采样（小噪声）
            for _ in range(aug_factor // 3):
                z = vae.reparameterize(mu, log_var, noise_scale=0.3)
                new_samples = vae.decode(z).cpu().numpy()
                X_aug_list.append(new_samples)
                y_aug_list.append(np.full(len(X_cls), cls))
            
            # 3. 噪声增强
            for noise in [0.05, 0.1, 0.15]:
                noisy = X_cls + np.random.randn(*X_cls.shape) * noise
                X_aug_list.append(noisy)
                y_aug_list.append(np.full(len(X_cls), cls))
    
    return np.vstack(X_aug_list), np.hstack(y_aug_list), scaler


def train_hypernet_with_rf_guidance(X_aug, y_aug, X_test, n_classes, n_runs=3):
    """
    训练HyperNet，用RF软标签指导
    多次运行取最优
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X_aug.shape[1]
    
    # 先训练RF获取软标签
    rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    rf.fit(X_aug, y_aug)
    rf_probs = rf.predict_proba(X_aug)  # 软标签
    
    best_preds = None
    best_confidence = -1
    
    for run in range(n_runs):
        torch.manual_seed(run * 100)
        np.random.seed(run * 100)
        
        # 创建HyperNet集成
        model = HyperNetEnsemble(input_dim, n_classes, n_nets=5, hidden_dim=32).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)
        
        X_tensor = torch.FloatTensor(X_aug).to(device)
        y_tensor = torch.LongTensor(y_aug).to(device)
        rf_probs_tensor = torch.FloatTensor(rf_probs).to(device)
        
        # 训练
        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            
            logits = model(X_tensor)
            
            # 硬标签损失
            ce_loss = F.cross_entropy(logits, y_tensor)
            
            # RF软标签蒸馏损失（KL散度）
            log_probs = F.log_softmax(logits, dim=1)
            kd_loss = F.kl_div(log_probs, rf_probs_tensor, reduction='batchmean')
            
            # 组合损失
            loss = ce_loss + 0.5 * kd_loss
            
            loss.backward()
            optimizer.step()
        
        # 预测
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            logits = model(X_test_tensor)
            probs = F.softmax(logits, dim=1)
            confidence = probs.max(dim=1)[0].mean().item()
            preds = model.predict_vote(X_test_tensor)
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_preds = preds
    
    return best_preds


def process_fold_rf(args):
    """RF基线"""
    fold_idx, X_train, y_train, X_test, y_test = args
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    rf.fit(X_train_s, y_train)
    return {'y_true': y_test.tolist(), 'y_pred': rf.predict(X_test_s).tolist()}


def process_fold_vae_hypernet(args):
    """VAE + HyperNet集成"""
    fold_idx, X_train, y_train, X_test, y_test = args
    
    n_classes = len(np.unique(y_train))
    
    # VAE增强
    X_aug, y_aug, scaler = vae_augment(X_train, y_train, aug_factor=30)
    X_test_s = scaler.transform(X_test)
    
    # HyperNet训练和预测
    y_pred = train_hypernet_with_rf_guidance(X_aug, y_aug, X_test_s, n_classes, n_runs=3)
    
    return {'y_true': y_test.tolist(), 'y_pred': y_pred}


def process_fold_vae_rf(args):
    """VAE + RF (对比)"""
    fold_idx, X_train, y_train, X_test, y_test = args
    
    X_aug, y_aug, scaler = vae_augment(X_train, y_train, aug_factor=30)
    X_test_s = scaler.transform(X_test)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    rf.fit(X_aug, y_aug)
    
    return {'y_true': y_test.tolist(), 'y_pred': rf.predict(X_test_s).tolist()}


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR / f'22_vae_hypernet_80_{timestamp}.log'
    
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
    logger.info("22_vae_hypernet_80.py - VAE+HyperNet 目标80%")
    logger.info("=" * 70)
    
    data_path = SCRIPT_DIR / 'data' / 'Data_for_Jinming.csv'
    df = pd.read_csv(data_path)
    X = df[['LAA', 'Glutamate', 'Choline', 'Sarcosine']].values.astype(np.float32)
    y = LabelEncoder().fit_transform(df['Group'].values)
    
    n_samples = len(X)
    n_classes = len(np.unique(y))
    logger.info(f"数据: {n_samples} 样本, {n_classes} 类别")
    
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
    
    # 1. RF基线
    logger.info("\n[1/3] RF基线...")
    start = time.time()
    with Pool(n_processes) as pool:
        r = list(pool.imap(process_fold_rf, fold_datas, chunksize=10))
    results['RF'] = accuracy_score([i for x in r for i in x['y_true']], [i for x in r for i in x['y_pred']]) * 100
    logger.info(f"   RF: {results['RF']:.2f}% ({time.time()-start:.1f}s)")
    
    # 2. VAE+RF
    logger.info("\n[2/3] VAE+RF...")
    start = time.time()
    with Pool(n_processes) as pool:
        r = list(pool.imap(process_fold_vae_rf, fold_datas, chunksize=10))
    results['VAE+RF'] = accuracy_score([i for x in r for i in x['y_true']], [i for x in r for i in x['y_pred']]) * 100
    logger.info(f"   VAE+RF: {results['VAE+RF']:.2f}% ({time.time()-start:.1f}s)")
    
    # 3. VAE+HyperNet
    logger.info("\n[3/3] VAE+HyperNet集成 (RF蒸馏)...")
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
    
    result_file = OUTPUT_DIR / f'22_vae_hypernet_80_{timestamp}.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"保存: {result_file}")


if __name__ == '__main__':
    main()
