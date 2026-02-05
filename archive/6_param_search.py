#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6_param_search.py - 参数搜索版本
=================================
通过网格搜索找到最佳参数组合

运行: python 6_param_search.py
"""

import os
import sys
import json
import time
import warnings
import multiprocessing as mp
from datetime import datetime
from itertools import combinations, product
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')


# ==================== VAE ====================
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=2):
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


def vae_augment(X, y, vae_epochs, num_interp):
    """VAE增强"""
    X_aug, y_aug = [X.copy()], [y.copy()]
    
    for cls in np.unique(y):
        X_cls = X[y == cls]
        if len(X_cls) < 2:
            continue
        
        vae = VAE(X_cls.shape[1], 64, 2)
        opt = torch.optim.Adam(vae.parameters(), lr=0.001)
        X_t = torch.FloatTensor(X_cls)
        
        for _ in range(vae_epochs):
            opt.zero_grad()
            recon, mu, log_var = vae(X_t)
            loss = nn.MSELoss()(recon, X_t) + 0.5 * (-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())) / len(X_cls)
            loss.backward()
            opt.step()
        
        vae.eval()
        with torch.no_grad():
            recon = vae(X_t)[0].numpy()
        
        for alpha in np.linspace(0.1, 0.9, num_interp):
            X_aug.append(alpha * X_cls + (1 - alpha) * recon)
            y_aug.append(np.full(len(X_cls), cls))
    
    return np.vstack(X_aug), np.hstack(y_aug)


def eval_config(config, X, y):
    """评估单个配置"""
    n = len(X)
    all_indices = np.arange(n)
    test_combos = list(combinations(all_indices, 2))
    
    correct = 0
    total = 0
    
    for test_idx in test_combos:
        test_idx = np.array(test_idx)
        train_idx = np.setdiff1d(all_indices, test_idx)
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        X_aug, y_aug = vae_augment(X_train_s, y_train, config['vae_epochs'], config['num_interp'])
        
        clf = RandomForestClassifier(
            n_estimators=config['n_est'],
            max_depth=config['max_depth'],
            random_state=42, n_jobs=1
        )
        clf.fit(X_aug, y_aug)
        
        y_pred = clf.predict(X_test_s)
        correct += np.sum(y_pred == y_test)
        total += len(y_test)
    
    return correct / total


def search_worker(args):
    """搜索工作进程"""
    config, X, y = args
    try:
        acc = eval_config(config, X, y)
        return {**config, 'accuracy': acc}
    except:
        return {**config, 'accuracy': 0.0}


def main():
    print("=" * 60)
    print("参数搜索 - 找最佳配置")
    print("=" * 60)
    
    # 加载数据
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'data', 'Data_for_Jinming.csv')
    df = pd.read_csv(data_path)
    
    feature_cols = ['LAA', 'Glutamate', 'Choline', 'Sarcosine']
    X = df[feature_cols].values.astype(np.float32)
    y = LabelEncoder().fit_transform(df['Group'].values)
    
    print(f"数据: {len(X)} 样本, {X.shape[1]} 特征")
    
    # 参数网格
    param_grid = {
        'vae_epochs': [50, 100, 150],
        'num_interp': [5, 8, 10, 15],
        'n_est': [100, 150, 200, 250],
        'max_depth': [3, 5, 7, 10, None]
    }
    
    # 生成所有组合
    keys = list(param_grid.keys())
    configs = [dict(zip(keys, v)) for v in product(*param_grid.values())]
    
    print(f"搜索空间: {len(configs)} 个配置")
    print(f"使用 {min(32, mp.cpu_count())} 个进程")
    
    # 并行搜索
    start = time.time()
    
    args_list = [(c, X, y) for c in configs]
    
    results = []
    with mp.Pool(min(32, mp.cpu_count())) as pool:
        for i, res in enumerate(pool.imap_unordered(search_worker, args_list)):
            results.append(res)
            if (i + 1) % 10 == 0:
                best_so_far = max(results, key=lambda x: x['accuracy'])
                print(f"[{i+1}/{len(configs)}] 当前最佳: {best_so_far['accuracy']*100:.2f}%")
    
    elapsed = time.time() - start
    
    # 排序结果
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print("\n" + "=" * 60)
    print("Top 10 配置:")
    print("=" * 60)
    for i, r in enumerate(results[:10]):
        print(f"{i+1}. 准确率: {r['accuracy']*100:.2f}% | "
              f"VAE epochs: {r['vae_epochs']}, interp: {r['num_interp']}, "
              f"n_est: {r['n_est']}, max_depth: {r['max_depth']}")
    
    print(f"\n总用时: {elapsed:.1f}秒")
    
    # 保存结果
    output_dir = os.path.join(script_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(os.path.join(output_dir, f'param_search_{timestamp}.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n最佳配置:")
    best = results[0]
    print(f"  VAE epochs: {best['vae_epochs']}")
    print(f"  插值点数: {best['num_interp']}")
    print(f"  RF n_estimators: {best['n_est']}")
    print(f"  RF max_depth: {best['max_depth']}")
    print(f"  准确率: {best['accuracy']*100:.2f}%")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
