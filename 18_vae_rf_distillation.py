#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
18_vae_rf_distillation.py - VAE增强 + RF知识蒸馏到神经网络
============================================================
核心思路：
1. VAE数据增强
2. RF在增强数据上训练（作为教师）
3. 神经网络学习RF的软标签（知识蒸馏）
4. 最终用神经网络预测

这样神经网络学到的是RF的决策边界，而不是直接学原始标签！

运行: python 18_vae_rf_distillation.py
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


def vae_augment(X_train, y_train):
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
        optimizer = torch.optim.Adam(vae.parameters(), lr=0.005)
        X_tensor = torch.FloatTensor(X_cls).to(device)
        
        vae.train()
        for _ in range(40):
            optimizer.zero_grad()
            recon, mu, log_var = vae(X_tensor)
            loss = nn.MSELoss()(recon, X_tensor) + 0.01 * (-0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp()))
            loss.backward()
            optimizer.step()
        
        vae.eval()
        with torch.no_grad():
            recon = vae(X_tensor)[0].cpu().numpy()
        
        for alpha in np.linspace(0.15, 0.85, 7):
            X_aug_list.append(alpha * X_cls + (1 - alpha) * recon)
            y_aug_list.append(np.full(len(X_cls), cls))
    
    return np.vstack(X_aug_list), np.hstack(y_aug_list), scaler


class StudentNet(nn.Module):
    """学生网络 - 学习RF的决策"""
    def __init__(self, input_dim, n_classes):
        super(StudentNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, n_classes)
        )
    
    def forward(self, x):
        return self.net(x)


def distillation_loss(student_logits, teacher_probs, true_labels, temperature=3.0, alpha=0.7):
    """
    知识蒸馏损失
    alpha: 软标签权重
    temperature: 温度参数，越高分布越平滑
    """
    soft_loss = nn.KLDivLoss(reduction='batchmean')(
        nn.functional.log_softmax(student_logits / temperature, dim=1),
        teacher_probs
    ) * (temperature ** 2)
    
    hard_loss = nn.CrossEntropyLoss()(student_logits, true_labels)
    
    return alpha * soft_loss + (1 - alpha) * hard_loss


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
    """VAE+RF"""
    fold_idx, X_train, y_train, X_test, y_test = args
    X_aug, y_aug, scaler = vae_augment(X_train, y_train)
    X_test_s = scaler.transform(X_test)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    rf.fit(X_aug, y_aug)
    y_pred = rf.predict(X_test_s)
    
    return {'y_true': y_test.tolist(), 'y_pred': y_pred.tolist()}


def process_fold_distillation(args):
    """VAE + RF知识蒸馏到神经网络"""
    fold_idx, X_train, y_train, X_test, y_test = args
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. VAE增强
    X_aug, y_aug, scaler = vae_augment(X_train, y_train)
    X_test_s = scaler.transform(X_test)
    
    # 2. 训练RF教师
    rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    rf.fit(X_aug, y_aug)
    
    # 获取RF的软标签（概率）
    teacher_probs = rf.predict_proba(X_aug)
    teacher_probs = torch.FloatTensor(teacher_probs).to(device)
    
    # 3. 训练学生网络（知识蒸馏）
    n_classes = len(np.unique(y_train))
    student = StudentNet(X_train.shape[1], n_classes).to(device)
    optimizer = optim.Adam(student.parameters(), lr=0.01, weight_decay=0.001)
    
    X_tensor = torch.FloatTensor(X_aug).to(device)
    y_tensor = torch.LongTensor(y_aug).to(device)
    
    student.train()
    for epoch in range(50):
        optimizer.zero_grad()
        student_logits = student(X_tensor)
        loss = distillation_loss(student_logits, teacher_probs, y_tensor, temperature=3.0, alpha=0.7)
        loss.backward()
        optimizer.step()
    
    # 4. 用学生网络预测
    student.eval()
    X_test_tensor = torch.FloatTensor(X_test_s).to(device)
    with torch.no_grad():
        y_pred = student(X_test_tensor).argmax(dim=1).cpu().numpy()
    
    return {'y_true': y_test.tolist(), 'y_pred': y_pred.tolist()}


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR / f'18_distillation_{timestamp}.log'
    
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
    logger.info("18_vae_rf_distillation.py - VAE + RF知识蒸馏到神经网络")
    logger.info("=" * 70)
    logger.info("思路: 神经网络学习RF的软决策边界，而不是硬标签")
    
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
    
    # 1. RF
    logger.info("\n[1/3] RF基线...")
    start = time.time()
    with Pool(n_processes) as pool:
        results = list(pool.imap(process_fold_rf, fold_datas, chunksize=10))
    rf_acc = accuracy_score(
        [i for r in results for i in r['y_true']], 
        [i for r in results for i in r['y_pred']]
    ) * 100
    logger.info(f"   RF: {rf_acc:.2f}% ({time.time()-start:.1f}s)")
    
    # 2. VAE+RF
    logger.info("\n[2/3] VAE+RF...")
    start = time.time()
    with Pool(n_processes) as pool:
        results = list(pool.imap(process_fold_vae_rf, fold_datas, chunksize=10))
    vae_rf_acc = accuracy_score(
        [i for r in results for i in r['y_true']], 
        [i for r in results for i in r['y_pred']]
    ) * 100
    logger.info(f"   VAE+RF: {vae_rf_acc:.2f}% ({time.time()-start:.1f}s)")
    
    # 3. 知识蒸馏
    logger.info("\n[3/3] VAE+RF→NN(蒸馏)...")
    start = time.time()
    with Pool(n_processes) as pool:
        results = list(pool.imap(process_fold_distillation, fold_datas, chunksize=10))
    distill_acc = accuracy_score(
        [i for r in results for i in r['y_true']], 
        [i for r in results for i in r['y_pred']]
    ) * 100
    logger.info(f"   知识蒸馏: {distill_acc:.2f}% ({time.time()-start:.1f}s)")
    
    logger.info("\n" + "=" * 70)
    logger.info("[结果对比]")
    logger.info("=" * 70)
    logger.info(f"  RF:           {rf_acc:.2f}%")
    logger.info(f"  VAE+RF:       {vae_rf_acc:.2f}%")
    logger.info(f"  VAE+RF→NN:    {distill_acc:.2f}% (知识蒸馏)")
    logger.info("=" * 70)
    
    result_file = OUTPUT_DIR / f'18_distillation_{timestamp}.json'
    with open(result_file, 'w') as f:
        json.dump({'RF': rf_acc, 'VAE+RF': vae_rf_acc, 'Distillation': distill_acc}, f, indent=2)
    logger.info(f"保存: {result_file}")


if __name__ == '__main__':
    main()
