#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
21_local_test_simple.py - 本地简化测试版本
==========================================
在本地快速测试各种分类器性能
使用更少的fold来加速测试

运行: python 21_local_test_simple.py
"""

import warnings
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from itertools import combinations

warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).parent


class VAE(nn.Module):
    def __init__(self, input_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU()
        )
        self.fc_mu = nn.Linear(16, 8)
        self.fc_var = nn.Linear(16, 8)
        self.decoder = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, input_dim)
        )
    
    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = self.fc_mu(h), self.fc_var(h)
        z = mu + torch.exp(0.5 * log_var) * torch.randn_like(log_var)
        return self.decoder(z), mu, log_var


def vae_augment_simple(X_train, y_train, aug_factor=10):
    """简化VAE增强"""
    input_dim = X_train.shape[1]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    X_aug, y_aug = [X_scaled], [y_train]
    
    for cls in np.unique(y_train):
        X_cls = X_scaled[y_train == cls]
        
        vae = VAE(input_dim)
        optimizer = torch.optim.Adam(vae.parameters(), lr=0.01)
        X_tensor = torch.FloatTensor(X_cls)
        
        for _ in range(30):
            optimizer.zero_grad()
            recon, mu, log_var = vae(X_tensor)
            loss = nn.MSELoss()(recon, X_tensor) + 0.01 * (-0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp()))
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            recon = vae(X_tensor)[0].numpy()
            for a in np.linspace(0.2, 0.8, aug_factor // 2):
                X_aug.append(a * X_cls + (1-a) * recon)
                y_aug.append(np.full(len(X_cls), cls))
    
    return np.vstack(X_aug), np.hstack(y_aug), scaler


def test_classifier(X, y, clf_name, clf, use_vae=False):
    """测试分类器"""
    test_combos = list(combinations(range(len(X)), 2))
    
    y_true_all, y_pred_all = [], []
    
    for test_idx in test_combos:
        test_idx = np.array(test_idx)
        train_idx = np.setdiff1d(np.arange(len(X)), test_idx)
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        if use_vae:
            X_aug, y_aug, scaler = vae_augment_simple(X_train, y_train, aug_factor=10)
            X_test_s = scaler.transform(X_test)
        else:
            scaler = StandardScaler()
            X_aug = scaler.fit_transform(X_train)
            y_aug = y_train
            X_test_s = scaler.transform(X_test)
        
        clf.fit(X_aug, y_aug)
        y_pred = clf.predict(X_test_s)
        
        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())
    
    return accuracy_score(y_true_all, y_pred_all) * 100


def main():
    print("=" * 60)
    print("21_local_test_simple.py - 本地分类器测试")
    print("=" * 60)
    
    data_path = SCRIPT_DIR / 'data' / 'Data_for_Jinming.csv'
    df = pd.read_csv(data_path)
    X = df[['LAA', 'Glutamate', 'Choline', 'Sarcosine']].values.astype(np.float32)
    y = LabelEncoder().fit_transform(df['Group'].values)
    
    print(f"数据: {len(X)} 样本, {X.shape[1]} 特征")
    print(f"Leave-2-Out: {len(list(combinations(range(len(X)), 2)))} folds")
    
    classifiers = {
        'RF': lambda: RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42),
        'GB': lambda: GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42),
        'ExtraTrees': lambda: ExtraTreesClassifier(n_estimators=100, max_depth=4, random_state=42),
        'LR': lambda: LogisticRegression(C=1.0, max_iter=500, random_state=42),
        'SVM': lambda: SVC(C=1.0, kernel='rbf', random_state=42),
    }
    
    results = {}
    
    print("\n[无VAE增强]")
    for name, clf_fn in classifiers.items():
        start = time.time()
        acc = test_classifier(X, y, name, clf_fn(), use_vae=False)
        results[name] = acc
        print(f"  {name:12s}: {acc:.2f}% ({time.time()-start:.1f}s)")
    
    print("\n[VAE增强]")
    for name, clf_fn in classifiers.items():
        start = time.time()
        acc = test_classifier(X, y, name, clf_fn(), use_vae=True)
        results[f'VAE+{name}'] = acc
        print(f"  VAE+{name:10s}: {acc:.2f}% ({time.time()-start:.1f}s)")
    
    print("\n" + "=" * 60)
    print("[结果排名]")
    print("=" * 60)
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for i, (name, acc) in enumerate(sorted_results):
        marker = "🏆" if i == 0 else f"{i+1:2d}"
        print(f"{marker}. {name:20s}: {acc:.2f}%")


if __name__ == '__main__':
    main()
