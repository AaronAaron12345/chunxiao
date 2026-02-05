#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
28_neural_decision_tree.py - 神经决策树
=======================================
思路：用神经网络模拟决策树的分层决策过程
每个节点是一个软分割，而不是硬分割

核心：
1. VAE数据增强
2. 神经决策树（可微分的树结构）
3. 多棵树集成（类似RF）

运行: python 28_neural_decision_tree.py
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


class SoftDecisionNode(nn.Module):
    """软决策节点 - 可微分的分割"""
    def __init__(self, input_dim):
        super(SoftDecisionNode, self).__init__()
        # 学习分割方向和阈值
        self.split_weight = nn.Parameter(torch.randn(input_dim))
        self.split_bias = nn.Parameter(torch.zeros(1))
        self.temperature = 1.0
    
    def forward(self, x):
        # 计算软分割概率 (0=左, 1=右)
        logit = (x @ self.split_weight + self.split_bias) / self.temperature
        return torch.sigmoid(logit)


class NeuralDecisionTree(nn.Module):
    """神经决策树 - 深度3的软决策树"""
    def __init__(self, input_dim, n_classes, depth=3):
        super(NeuralDecisionTree, self).__init__()
        self.depth = depth
        self.n_classes = n_classes
        self.n_leaves = 2 ** depth
        
        # 创建所有内部节点
        n_internal = 2 ** depth - 1
        self.nodes = nn.ModuleList([SoftDecisionNode(input_dim) for _ in range(n_internal)])
        
        # 叶子节点的类别分布
        self.leaf_distributions = nn.Parameter(torch.randn(self.n_leaves, n_classes))
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # 计算到达每个叶子的概率
        leaf_probs = torch.ones(batch_size, 1, device=x.device)
        
        # 层级遍历计算路径概率
        node_idx = 0
        current_probs = [leaf_probs]
        
        for level in range(self.depth):
            new_probs = []
            for prob in current_probs:
                if node_idx < len(self.nodes):
                    split_prob = self.nodes[node_idx](x).unsqueeze(1)  # P(右)
                    left_prob = prob * (1 - split_prob)
                    right_prob = prob * split_prob
                    new_probs.extend([left_prob, right_prob])
                    node_idx += 1
            current_probs = new_probs
        
        # 合并所有叶子概率
        leaf_probs = torch.cat(current_probs, dim=1)  # (batch, n_leaves)
        
        # 加权叶子分布得到最终预测
        leaf_dist = F.softmax(self.leaf_distributions, dim=1)  # (n_leaves, n_classes)
        output = leaf_probs @ leaf_dist  # (batch, n_classes)
        
        return output


class NeuralForest(nn.Module):
    """神经森林 - 多棵神经决策树的集成"""
    def __init__(self, input_dim, n_classes, n_trees=20, depth=3):
        super(NeuralForest, self).__init__()
        self.trees = nn.ModuleList([
            NeuralDecisionTree(input_dim, n_classes, depth) 
            for _ in range(n_trees)
        ])
        self.n_trees = n_trees
    
    def forward(self, x):
        # 平均所有树的预测
        outputs = [tree(x) for tree in self.trees]
        return torch.stack(outputs, dim=0).mean(dim=0)


def vae_augment(vae, X_train, y_train, aug_factor=30, device='cuda'):
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
            
            for alpha in np.linspace(0.1, 0.9, aug_factor // 3):
                X_aug.append(alpha * X_cls_np + (1 - alpha) * recon)
                y_aug.append(np.full(mask.sum(), cls))
            
            for _ in range(aug_factor // 3):
                z = mu + torch.exp(0.5 * log_var) * torch.randn_like(log_var) * 0.3
                X_aug.append(vae.decoder(z).cpu().numpy())
                y_aug.append(np.full(mask.sum(), cls))
    
    return np.vstack(X_aug), np.hstack(y_aug)


def train_neural_forest(model, X, y, epochs=200, lr=0.01, device='cuda'):
    """训练神经森林"""
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    X_t = torch.FloatTensor(X).to(device)
    y_t = torch.LongTensor(y).to(device)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 每棵树随机采样一部分数据（bootstrap）
        outputs = []
        for tree in model.trees:
            # Bootstrap采样
            idx = torch.randint(0, len(X_t), (int(len(X_t) * 0.8),), device=device)
            out = tree(X_t[idx])
            target = y_t[idx]
            outputs.append((out, target))
        
        # 计算损失
        loss = 0
        for out, target in outputs:
            loss += F.cross_entropy(out, target, label_smoothing=0.1)
        loss /= len(outputs)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR / f'28_neural_tree_{timestamp}.log'
    
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
    logger.info("28_neural_decision_tree.py - 神经决策树")
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
    logger.info("\n[1/4] RF 基线...")
    rf_preds, rf_trues = [], []
    for train_idx, test_idx in skf.split(X_scaled, y):
        rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
        rf.fit(X_scaled[train_idx], y[train_idx])
        rf_preds.extend(rf.predict(X_scaled[test_idx]).tolist())
        rf_trues.extend(y[test_idx].tolist())
    results['RF'] = accuracy_score(rf_trues, rf_preds) * 100
    logger.info(f"   RF: {results['RF']:.2f}%")
    
    # ==================== VAE+RF ====================
    logger.info("\n[2/4] VAE + RF...")
    vae_rf_preds, vae_rf_trues = [], []
    for train_idx, test_idx in skf.split(X_scaled, y):
        X_train, y_train = X_scaled[train_idx], y[train_idx]
        X_test, y_test = X_scaled[test_idx], y[test_idx]
        
        vae = VAE(X_train.shape[1]).to(device)
        optimizer = optim.Adam(vae.parameters(), lr=0.01)
        X_t = torch.FloatTensor(X_train).to(device)
        for _ in range(60):
            optimizer.zero_grad()
            recon, mu, log_var = vae(X_t)
            loss = nn.MSELoss()(recon, X_t) + 0.01 * (-0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp()))
            loss.backward()
            optimizer.step()
        
        X_aug, y_aug = vae_augment(vae, X_train, y_train, aug_factor=30, device=device)
        
        rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
        rf.fit(X_aug, y_aug)
        vae_rf_preds.extend(rf.predict(X_test).tolist())
        vae_rf_trues.extend(y_test.tolist())
    
    results['VAE+RF'] = accuracy_score(vae_rf_trues, vae_rf_preds) * 100
    logger.info(f"   VAE+RF: {results['VAE+RF']:.2f}%")
    
    # ==================== 神经森林（无VAE） ====================
    logger.info("\n[3/4] Neural Forest (无VAE)...")
    nf_preds, nf_trues = [], []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
        X_train, y_train = X_scaled[train_idx], y[train_idx]
        X_test, y_test = X_scaled[test_idx], y[test_idx]
        
        model = NeuralForest(X_train.shape[1], len(np.unique(y_train)), n_trees=20, depth=3).to(device)
        train_neural_forest(model, X_train, y_train, epochs=200, lr=0.01, device=device)
        
        model.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test).to(device)
            pred = model(X_test_t).argmax(dim=1).cpu().numpy()
        
        nf_preds.extend(pred.tolist())
        nf_trues.extend(y_test.tolist())
        logger.info(f"   Fold {fold_idx+1}/5: {accuracy_score(y_test, pred)*100:.2f}%")
    
    results['NeuralForest'] = accuracy_score(nf_trues, nf_preds) * 100
    logger.info(f"   NeuralForest: {results['NeuralForest']:.2f}%")
    
    # ==================== VAE + 神经森林 ====================
    logger.info("\n[4/4] VAE + Neural Forest...")
    vae_nf_preds, vae_nf_trues = [], []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
        X_train, y_train = X_scaled[train_idx], y[train_idx]
        X_test, y_test = X_scaled[test_idx], y[test_idx]
        
        # VAE
        vae = VAE(X_train.shape[1]).to(device)
        optimizer = optim.Adam(vae.parameters(), lr=0.01)
        X_t = torch.FloatTensor(X_train).to(device)
        for _ in range(60):
            optimizer.zero_grad()
            recon, mu, log_var = vae(X_t)
            loss = nn.MSELoss()(recon, X_t) + 0.01 * (-0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp()))
            loss.backward()
            optimizer.step()
        
        X_aug, y_aug = vae_augment(vae, X_train, y_train, aug_factor=50, device=device)
        
        # 神经森林
        model = NeuralForest(X_train.shape[1], len(np.unique(y_train)), n_trees=30, depth=3).to(device)
        train_neural_forest(model, X_aug, y_aug, epochs=300, lr=0.01, device=device)
        
        model.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test).to(device)
            pred = model(X_test_t).argmax(dim=1).cpu().numpy()
        
        vae_nf_preds.extend(pred.tolist())
        vae_nf_trues.extend(y_test.tolist())
        logger.info(f"   Fold {fold_idx+1}/5: {accuracy_score(y_test, pred)*100:.2f}%")
    
    results['VAE+NeuralForest'] = accuracy_score(vae_nf_trues, vae_nf_preds) * 100
    logger.info(f"   VAE+NeuralForest: {results['VAE+NeuralForest']:.2f}%")
    
    # ==================== 汇总 ====================
    logger.info("\n" + "=" * 70)
    logger.info("[结果对比]")
    logger.info("=" * 70)
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        marker = "🏆" if acc == max(results.values()) else "  "
        logger.info(f"{marker} {name:20s}: {acc:.2f}%")
    logger.info("=" * 70)
    
    result_file = OUTPUT_DIR / f'28_neural_tree_{timestamp}.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"保存: {result_file}")


if __name__ == '__main__':
    main()
