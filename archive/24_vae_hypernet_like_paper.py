#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
24_vae_hypernet_like_paper.py - 完全模仿论文的VAE + 改进HyperNet
================================================================
关键改进：
1. 使用论文中相同的VAE结构（intermediate_dim=512, latent_dim=2）
2. 使用线性插值增强（原论文方法）
3. HyperNet模仿RF：输入数据特征 -> 输出多个"树"的预测 -> 投票
4. 更大的增强倍数

运行: python 24_vae_hypernet_like_paper.py
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


class PaperStyleVAE(nn.Module):
    """模仿论文的VAE结构：intermediate_dim=512, latent_dim=2"""
    def __init__(self, input_dim, intermediate_dim=512, latent_dim=2):
        super(PaperStyleVAE, self).__init__()
        
        # Encoder
        self.encoder_hidden = nn.Linear(input_dim, intermediate_dim)
        self.fc_mu = nn.Linear(intermediate_dim, latent_dim)
        self.fc_var = nn.Linear(intermediate_dim, latent_dim)
        
        # Decoder  
        self.decoder_hidden = nn.Linear(latent_dim, intermediate_dim)
        self.decoder_out = nn.Linear(intermediate_dim, input_dim)
    
    def encode(self, x):
        h = F.relu(self.encoder_hidden(x))
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = F.relu(self.decoder_hidden(z))
        return torch.sigmoid(self.decoder_out(h))
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


def linear_interpolate(point_a, point_b, num_points=5):
    """论文中的线性插值方法"""
    return np.linspace(point_a, point_b, num=num_points + 2)[1:-1]


def paper_style_vae_augment(X_train, y_train, num_interpolation_points=5):
    """完全模仿论文的VAE数据增强"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # MinMaxScaler (论文使用)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_train).astype(np.float32)
    
    input_dim = X_scaled.shape[1]
    augmented_data = []
    augmented_labels = []
    
    for cls in np.unique(y_train):
        X_cls = X_scaled[y_train == cls]
        
        # 创建VAE
        vae = PaperStyleVAE(input_dim, intermediate_dim=512, latent_dim=2).to(device)
        optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
        X_tensor = torch.FloatTensor(X_cls).to(device)
        
        # 训练VAE (论文epochs=50)
        vae.train()
        for epoch in range(50):
            optimizer.zero_grad()
            recon, mu, log_var = vae(X_tensor)
            
            # 确保值在(0,1)范围内，避免BCE错误
            recon_clamped = torch.clamp(recon, 1e-6, 1-1e-6)
            target_clamped = torch.clamp(X_tensor, 1e-6, 1-1e-6)
            
            # BCE重建损失 + KL损失 (论文使用)
            recon_loss = F.binary_cross_entropy(recon_clamped, target_clamped, reduction='sum') / len(X_cls)
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / len(X_cls)
            loss = recon_loss + kl_loss
            
            loss.backward()
            optimizer.step()
        
        # 生成增强数据
        vae.eval()
        with torch.no_grad():
            mu, _ = vae.encode(X_tensor)
            decoded = vae.decode(mu).cpu().numpy()
            
            # 线性插值 (论文方法)
            for original, dec in zip(X_cls, decoded):
                interpolated = linear_interpolate(original, dec, num_interpolation_points)
                augmented_data.extend(interpolated)
                augmented_labels.extend([cls] * len(interpolated))
    
    return np.array(augmented_data), np.array(augmented_labels), scaler


class TreeLikeHyperNet(nn.Module):
    """
    模仿RF的HyperNet：
    - 每个"树"是一个小型权重生成器
    - 最终通过投票或平均得到结果
    """
    def __init__(self, input_dim, n_classes, n_trees=10, tree_hidden=16):
        super(TreeLikeHyperNet, self).__init__()
        self.n_trees = n_trees
        self.n_classes = n_classes
        self.input_dim = input_dim
        
        # 每棵"树"的权重生成器
        self.trees = nn.ModuleList()
        for _ in range(n_trees):
            tree = nn.Sequential(
                nn.Linear(input_dim, tree_hidden),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(tree_hidden, input_dim * n_classes + n_classes)  # 权重+偏置
            )
            self.trees.append(tree)
    
    def forward(self, x):
        """返回所有树的平均logits"""
        batch_size = x.shape[0]
        all_logits = []
        
        for tree in self.trees:
            params = tree(x)
            # 分离权重和偏置
            W = params[:, :self.input_dim * self.n_classes].view(batch_size, self.n_classes, self.input_dim)
            b = params[:, self.input_dim * self.n_classes:]
            
            # 计算logits: (batch, n_classes, input_dim) @ (batch, input_dim, 1)
            logits = torch.bmm(W, x.unsqueeze(2)).squeeze(2) + b
            all_logits.append(logits)
        
        # 平均所有树的logits
        return torch.stack(all_logits).mean(dim=0)
    
    def predict_vote(self, x):
        """投票预测"""
        batch_size = x.shape[0]
        all_preds = []
        
        for tree in self.trees:
            params = tree(x)
            W = params[:, :self.input_dim * self.n_classes].view(batch_size, self.n_classes, self.input_dim)
            b = params[:, self.input_dim * self.n_classes:]
            logits = torch.bmm(W, x.unsqueeze(2)).squeeze(2) + b
            preds = logits.argmax(dim=1)
            all_preds.append(preds)
        
        # 投票
        all_preds = torch.stack(all_preds, dim=1)  # (batch, n_trees)
        voted = []
        for i in range(batch_size):
            votes = all_preds[i].cpu().numpy()
            voted.append(np.bincount(votes, minlength=self.n_classes).argmax())
        return voted


def train_hypernet(X_train, y_train, X_test, n_classes):
    """训练HyperNet"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X_train.shape[1]
    
    # 多次尝试，取最好结果
    best_preds = None
    best_train_acc = 0
    
    for seed in range(3):
        torch.manual_seed(seed * 42)
        np.random.seed(seed * 42)
        
        model = TreeLikeHyperNet(input_dim, n_classes, n_trees=20, tree_hidden=16).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.02, weight_decay=0.05)
        
        X_tensor = torch.FloatTensor(X_train).to(device)
        y_tensor = torch.LongTensor(y_train).to(device)
        
        # 训练
        model.train()
        for epoch in range(150):
            optimizer.zero_grad()
            logits = model(X_tensor)
            loss = F.cross_entropy(logits, y_tensor)
            loss.backward()
            optimizer.step()
        
        # 评估训练准确率
        model.eval()
        with torch.no_grad():
            train_preds = model(X_tensor).argmax(dim=1)
            train_acc = (train_preds == y_tensor).float().mean().item()
            
            if train_acc > best_train_acc:
                best_train_acc = train_acc
                X_test_tensor = torch.FloatTensor(X_test).to(device)
                best_preds = model.predict_vote(X_test_tensor)
    
    return best_preds


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
    
    X_aug, y_aug, scaler = paper_style_vae_augment(X_train, y_train, num_interpolation_points=5)
    X_test_s = scaler.transform(X_test)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    rf.fit(X_aug, y_aug)
    return {'y_true': y_test.tolist(), 'y_pred': rf.predict(X_test_s).tolist()}


def process_fold_vae_hypernet(args):
    fold_idx, X_train, y_train, X_test, y_test = args
    n_classes = len(np.unique(y_train))
    
    X_aug, y_aug, scaler = paper_style_vae_augment(X_train, y_train, num_interpolation_points=5)
    X_test_s = scaler.transform(X_test)
    
    y_pred = train_hypernet(X_aug, y_aug, X_test_s, n_classes)
    return {'y_true': y_test.tolist(), 'y_pred': y_pred}


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR / f'24_paper_style_{timestamp}.log'
    
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
    logger.info("24_vae_hypernet_like_paper.py - 论文风格VAE + TreeLikeHyperNet")
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
    logger.info("\n[1/3] RF...")
    start = time.time()
    with Pool(n_processes) as pool:
        r = list(pool.imap(process_fold_rf, fold_datas, chunksize=10))
    results['RF'] = accuracy_score([i for x in r for i in x['y_true']], [i for x in r for i in x['y_pred']]) * 100
    logger.info(f"   RF: {results['RF']:.2f}% ({time.time()-start:.1f}s)")
    
    # 2. VAE+RF (论文方法)
    logger.info("\n[2/3] VAE+RF (论文方法)...")
    start = time.time()
    with Pool(n_processes) as pool:
        r = list(pool.imap(process_fold_vae_rf, fold_datas, chunksize=10))
    results['VAE+RF'] = accuracy_score([i for x in r for i in x['y_true']], [i for x in r for i in x['y_pred']]) * 100
    logger.info(f"   VAE+RF: {results['VAE+RF']:.2f}% ({time.time()-start:.1f}s)")
    
    # 3. VAE+HyperNet
    logger.info("\n[3/3] VAE+TreeLikeHyperNet...")
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
    
    result_file = OUTPUT_DIR / f'24_paper_style_{timestamp}.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
