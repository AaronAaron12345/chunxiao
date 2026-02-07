#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
8_bayes_per_fold.py - 每个fold独立贝叶斯优化
==============================================
为每个fold独立寻找最佳RF参数，可能达到更高准确率

计算量: 325 folds × 内部CV × 贝叶斯迭代 = 较大
但对于26样本数据集是可行的

运行: python 8_bayes_per_fold.py

日志输出:
- logs/8_bayes_per_fold_YYYYMMDD_HHMMSS.log
- output/8_bayes_per_fold_YYYYMMDD_HHMMSS.json
"""

import os
import sys
import json
import time
import logging
import warnings
import multiprocessing as mp
from datetime import datetime
from itertools import combinations
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

# 获取脚本目录
SCRIPT_DIR = Path(__file__).parent
LOG_DIR = SCRIPT_DIR / 'logs'
OUTPUT_DIR = SCRIPT_DIR / 'output'
LOG_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ==================== VAE 数据增强 ====================
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


def vae_augment(X, y, vae_epochs=100, num_interp=5):
    """VAE数据增强 - 返回原始数据+增强数据"""
    X_aug_list = [X.copy()]
    y_aug_list = [y.copy()]
    
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
        
        # 线性插值生成增强数据
        for alpha in np.linspace(0.1, 0.9, num_interp):
            X_aug_list.append(alpha * X_cls + (1 - alpha) * recon)
            y_aug_list.append(np.full(len(X_cls), cls))
    
    return np.vstack(X_aug_list), np.hstack(y_aug_list)


def bayesian_optimize_rf(X_train, y_train, n_iter=20):
    """
    在单个fold内部用贝叶斯优化寻找最佳RF参数
    使用3折交叉验证评估（因为训练集只有24个样本）
    """
    from skopt import BayesSearchCV
    from skopt.space import Integer, Categorical
    
    # 参数搜索空间
    search_space = {
        'n_estimators': Integer(50, 300),
        'max_depth': Integer(2, 15),
        'min_samples_split': Integer(2, 10),
        'min_samples_leaf': Integer(1, 5),
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=1)
    
    # 贝叶斯优化 - 内部用3折CV
    opt = BayesSearchCV(
        rf, search_space,
        n_iter=n_iter,
        cv=3,  # 内部3折验证
        scoring='accuracy',
        random_state=42,
        n_jobs=1,
        verbose=0
    )
    
    opt.fit(X_train, y_train)
    
    return opt.best_estimator_, opt.best_params_, opt.best_score_


def process_fold_with_bayes(fold_info, X, y, config):
    """处理单个fold - 内部做贝叶斯优化"""
    fold_idx, (train_idx, test_idx) = fold_info
    
    try:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # VAE增强
        X_aug, y_aug = vae_augment(
            X_train_scaled, y_train,
            vae_epochs=config['vae_epochs'],
            num_interp=config['num_interp']
        )
        
        # 贝叶斯优化找最佳RF参数
        best_clf, best_params, cv_score = bayesian_optimize_rf(
            X_aug, y_aug, 
            n_iter=config['bayes_n_iter']
        )
        
        # 预测
        y_pred = best_clf.predict(X_test_scaled)
        y_prob = best_clf.predict_proba(X_test_scaled)
        
        return {
            'fold_idx': fold_idx,
            'accuracy': accuracy_score(y_test, y_pred),
            'y_true': y_test.tolist(),
            'y_pred': y_pred.tolist(),
            'y_prob': y_prob.tolist(),
            'best_params': best_params,
            'cv_score': cv_score
        }
    except Exception as e:
        return {'fold_idx': fold_idx, 'accuracy': 0.0, 'error': str(e)}


def main():
    # 设置日志
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR / f'8_bayes_per_fold_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("8_bayes_per_fold.py - 每个fold独立贝叶斯优化")
    logger.info(f"日志文件: {log_file}")
    logger.info("=" * 60)
    
    # 配置
    config = {
        'vae_epochs': 100,
        'num_interp': 5,
        'bayes_n_iter': 20,  # 每个fold做20次贝叶斯迭代
        'leave_p_out': 2
    }
    
    logger.info(f"配置: {config}")
    logger.info(f"  - 每个fold内部: 贝叶斯优化 {config['bayes_n_iter']} 次迭代")
    logger.info(f"  - VAE epochs: {config['vae_epochs']}")
    
    # 加载数据
    data_path = SCRIPT_DIR / 'data' / 'Data_for_Jinming.csv'
    
    df = pd.read_csv(data_path)
    feature_cols = ['LAA', 'Glutamate', 'Choline', 'Sarcosine']
    X = df[feature_cols].values.astype(np.float32)
    y = LabelEncoder().fit_transform(df['Group'].values)
    
    n_samples = len(X)
    logger.info(f"数据: {n_samples} 样本, {X.shape[1]} 特征")
    
    # 生成所有fold
    all_indices = np.arange(n_samples)
    test_combos = list(combinations(all_indices, config['leave_p_out']))
    n_folds = len(test_combos)
    
    logger.info(f"Leave-{config['leave_p_out']}-Out: {n_folds} 个 folds")
    
    fold_infos = []
    for fold_idx, test_idx in enumerate(test_combos):
        test_idx = np.array(test_idx)
        train_idx = np.setdiff1d(all_indices, test_idx)
        fold_infos.append((fold_idx, (train_idx, test_idx)))
    
    # 确定进程数
    n_processes = min(mp.cpu_count(), n_folds, 32)
    logger.info(f"使用 {n_processes} 个进程并行处理")
    logger.info("⚠️  注意: 每个fold都做贝叶斯优化，预计需要较长时间...")
    
    start_time = time.time()
    
    # 并行处理
    process_func = partial(process_fold_with_bayes, X=X, y=y, config=config)
    
    results = []
    with mp.Pool(processes=n_processes) as pool:
        for i, res in enumerate(pool.imap_unordered(process_func, fold_infos)):
            results.append(res)
            if (i + 1) % 20 == 0 or i + 1 == n_folds:
                # 计算当前准确率
                accs = [r['accuracy'] for r in results if 'error' not in r]
                curr_acc = np.mean(accs) * 100 if accs else 0
                elapsed = time.time() - start_time
                eta = elapsed / (i + 1) * (n_folds - i - 1)
                logger.info(f"[{i+1}/{n_folds}] 当前准确率: {curr_acc:.2f}% | "
                           f"已用: {elapsed:.0f}s | 预计剩余: {eta:.0f}s")
    
    elapsed_time = time.time() - start_time
    
    # 统计结果
    valid_results = [r for r in results if 'error' not in r]
    accuracies = [r['accuracy'] for r in valid_results]
    
    mean_acc = np.mean(accuracies) * 100
    std_acc = np.std(accuracies) * 100
    
    # 汇总预测计算整体准确率
    all_y_true, all_y_pred = [], []
    for r in valid_results:
        all_y_true.extend(r['y_true'])
        all_y_pred.extend(r['y_pred'])
    
    overall_acc = accuracy_score(all_y_true, all_y_pred) * 100
    
    # 统计最佳参数分布
    param_counts = {}
    for r in valid_results:
        if 'best_params' in r:
            for k, v in r['best_params'].items():
                if k not in param_counts:
                    param_counts[k] = []
                param_counts[k].append(v)
    
    logger.info("=" * 60)
    logger.info("[结果] Leave-2-Out + 每fold贝叶斯优化")
    logger.info("=" * 60)
    logger.info(f"  平均准确率: {mean_acc:.2f}% ± {std_acc:.2f}%")
    logger.info(f"  整体准确率: {overall_acc:.2f}%")
    logger.info(f"  成功folds: {len(valid_results)}/{n_folds}")
    logger.info(f"  总用时: {elapsed_time:.1f}秒")
    
    logger.info("[最佳参数分布统计]")
    for k, vals in param_counts.items():
        logger.info(f"  {k}: mean={np.mean(vals):.1f}, "
                   f"min={np.min(vals)}, max={np.max(vals)}")
    
    # 保存结果到JSON
    result_file = OUTPUT_DIR / f'8_bayes_per_fold_{timestamp}.json'
    
    result_data = {
        'experiment': '8_bayes_per_fold',
        'timestamp': datetime.now().isoformat(),
        'mean_accuracy': mean_acc / 100,
        'std_accuracy': std_acc / 100,
        'overall_accuracy': overall_acc / 100,
        'elapsed_time': elapsed_time,
        'config': config,
        'n_samples': n_samples,
        'n_folds': n_folds,
        'param_stats': {k: {'mean': float(np.mean(v)), 'min': int(np.min(v)), 'max': int(np.max(v))} 
                      for k, v in param_counts.items()}
    }
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"结果已保存: {result_file}")
    logger.info(f"日志已保存: {log_file}")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
