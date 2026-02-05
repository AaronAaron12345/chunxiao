#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
20_vae_xgboost_catboost.py - VAE + 集成学习方法对比
====================================================
尝试其他强力分类器：
1. XGBoost
2. LightGBM  
3. CatBoost
4. 多模型融合

运行: python 20_vae_xgboost_catboost.py
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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    BaggingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
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


def vae_augment(X_train, y_train, aug_factor=20):
    """VAE增强"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X_train.shape[1]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    X_aug_list = [X_scaled]
    y_aug_list = [y_train]
    
    for cls in np.unique(y_train):
        X_cls = X_scaled[y_train == cls]
        
        vae = VAE(input_dim).to(device)
        optimizer = torch.optim.Adam(vae.parameters(), lr=0.01)
        X_tensor = torch.FloatTensor(X_cls).to(device)
        
        vae.train()
        for _ in range(50):
            optimizer.zero_grad()
            recon, mu, log_var = vae(X_tensor)
            loss = nn.MSELoss()(recon, X_tensor) + 0.005 * (-0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp()))
            loss.backward()
            optimizer.step()
        
        vae.eval()
        with torch.no_grad():
            # 重建插值
            recon = vae(X_tensor)[0].cpu().numpy()
            for alpha in np.linspace(0.1, 0.9, aug_factor // 4):
                X_aug_list.append(alpha * X_cls + (1 - alpha) * recon)
                y_aug_list.append(np.full(len(X_cls), cls))
            
            # 潜在采样
            mu, log_var = vae.fc_mu(vae.encoder(X_tensor)), vae.fc_var(vae.encoder(X_tensor))
            for _ in range(aug_factor // 4):
                z = mu + torch.exp(0.5 * log_var) * torch.randn_like(log_var) * 0.3
                new = vae.decoder(z).cpu().numpy()
                X_aug_list.append(new)
                y_aug_list.append(np.full(len(X_cls), cls))
    
    return np.vstack(X_aug_list), np.hstack(y_aug_list), scaler


def make_classifier(name):
    """创建分类器"""
    classifiers = {
        'RF': RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42),
        'GB': GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=50, learning_rate=0.5, random_state=42),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=100, max_depth=4, random_state=42),
        'Bagging': BaggingClassifier(n_estimators=50, max_samples=0.8, random_state=42),
        'LR': LogisticRegression(C=1.0, max_iter=500, random_state=42),
        'SVM': SVC(C=1.0, kernel='rbf', random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=3),
        'NaiveBayes': GaussianNB(),
        'DecisionTree': DecisionTreeClassifier(max_depth=4, random_state=42),
    }
    return classifiers[name]


def process_fold(args):
    """处理单个fold - 多分类器"""
    fold_idx, X_train, y_train, X_test, y_test, clf_name, use_vae = args
    
    if use_vae:
        X_aug, y_aug, scaler = vae_augment(X_train, y_train, aug_factor=20)
        X_test_s = scaler.transform(X_test)
    else:
        scaler = StandardScaler()
        X_aug = scaler.fit_transform(X_train)
        y_aug = y_train
        X_test_s = scaler.transform(X_test)
    
    clf = make_classifier(clf_name)
    clf.fit(X_aug, y_aug)
    y_pred = clf.predict(X_test_s)
    
    return {'y_true': y_test.tolist(), 'y_pred': y_pred.tolist()}


def process_fold_voting(args):
    """多模型投票"""
    fold_idx, X_train, y_train, X_test, y_test, use_vae = args
    
    if use_vae:
        X_aug, y_aug, scaler = vae_augment(X_train, y_train, aug_factor=20)
        X_test_s = scaler.transform(X_test)
    else:
        scaler = StandardScaler()
        X_aug = scaler.fit_transform(X_train)
        y_aug = y_train
        X_test_s = scaler.transform(X_test)
    
    # 训练多个分类器
    classifiers = ['RF', 'GB', 'ExtraTrees', 'LR', 'SVM']
    predictions = []
    
    for clf_name in classifiers:
        clf = make_classifier(clf_name)
        clf.fit(X_aug, y_aug)
        predictions.append(clf.predict(X_test_s))
    
    # 投票
    predictions = np.array(predictions)
    y_pred = []
    for i in range(len(X_test_s)):
        votes = predictions[:, i].astype(int)
        y_pred.append(np.bincount(votes).argmax())
    
    return {'y_true': y_test.tolist(), 'y_pred': y_pred}


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR / f'20_multi_clf_{timestamp}.log'
    
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
    logger.info("20_vae_xgboost_catboost.py - 多分类器对比实验")
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
    classifiers_to_test = ['RF', 'GB', 'AdaBoost', 'ExtraTrees', 'Bagging', 'LR', 'SVM', 'KNN', 'NaiveBayes', 'DecisionTree']
    
    # 测试每个分类器（无VAE）
    logger.info("\n" + "=" * 70)
    logger.info("[无VAE增强]")
    logger.info("=" * 70)
    
    for clf_name in classifiers_to_test:
        start = time.time()
        fold_args = [(fd[0], fd[1], fd[2], fd[3], fd[4], clf_name, False) for fd in fold_datas]
        with Pool(n_processes) as pool:
            r = list(pool.imap(process_fold, fold_args, chunksize=10))
        acc = accuracy_score([i for x in r for i in x['y_true']], [i for x in r for i in x['y_pred']]) * 100
        results[clf_name] = acc
        logger.info(f"   {clf_name:12s}: {acc:.2f}% ({time.time()-start:.1f}s)")
    
    # 投票（无VAE）
    start = time.time()
    fold_args = [(fd[0], fd[1], fd[2], fd[3], fd[4], False) for fd in fold_datas]
    with Pool(n_processes) as pool:
        r = list(pool.imap(process_fold_voting, fold_args, chunksize=10))
    acc = accuracy_score([i for x in r for i in x['y_true']], [i for x in r for i in x['y_pred']]) * 100
    results['Voting'] = acc
    logger.info(f"   {'Voting':12s}: {acc:.2f}% ({time.time()-start:.1f}s)")
    
    # 测试每个分类器（有VAE）
    logger.info("\n" + "=" * 70)
    logger.info("[VAE增强]")
    logger.info("=" * 70)
    
    for clf_name in classifiers_to_test:
        start = time.time()
        fold_args = [(fd[0], fd[1], fd[2], fd[3], fd[4], clf_name, True) for fd in fold_datas]
        with Pool(n_processes) as pool:
            r = list(pool.imap(process_fold, fold_args, chunksize=10))
        acc = accuracy_score([i for x in r for i in x['y_true']], [i for x in r for i in x['y_pred']]) * 100
        results[f'VAE+{clf_name}'] = acc
        logger.info(f"   VAE+{clf_name:10s}: {acc:.2f}% ({time.time()-start:.1f}s)")
    
    # 投票（有VAE）
    start = time.time()
    fold_args = [(fd[0], fd[1], fd[2], fd[3], fd[4], True) for fd in fold_datas]
    with Pool(n_processes) as pool:
        r = list(pool.imap(process_fold_voting, fold_args, chunksize=10))
    acc = accuracy_score([i for x in r for i in x['y_true']], [i for x in r for i in x['y_pred']]) * 100
    results['VAE+Voting'] = acc
    logger.info(f"   VAE+{'Voting':10s}: {acc:.2f}% ({time.time()-start:.1f}s)")
    
    # 总结
    logger.info("\n" + "=" * 70)
    logger.info("[Top 10 结果]")
    logger.info("=" * 70)
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)[:10]
    for i, (name, acc) in enumerate(sorted_results):
        marker = "🏆" if i == 0 else f"{i+1:2d}"
        logger.info(f"{marker}. {name:20s}: {acc:.2f}%")
    logger.info("=" * 70)
    
    result_file = OUTPUT_DIR / f'20_multi_clf_{timestamp}.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"保存: {result_file}")


if __name__ == '__main__':
    main()
