#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5_final_optimized.py - VAE-HyperNetFusion 最终优化版
=====================================================
结合了贝叶斯优化 + 多进程并行 + 进度显示

特点：
1. 多进程并行（服务器64核心）
2. VAE数据增强 + 随机森林
3. 实时进度显示
4. 详细日志输出

运行方式：
  服务器: cd /data2/image_identification/src && python 5_final_optimized.py
  本地:   python 5_final_optimized.py

作者: Jinming Zhang
日期: 2026-02-03
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

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

warnings.filterwarnings('ignore')

# ==================== 配置日志 ====================
def setup_logging(log_file=None):
    """配置日志"""
    log_format = '%(asctime)s [%(levelname)s] %(message)s'
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers
    )
    return logging.getLogger(__name__)


# ==================== VAE 数据增强 ====================
class VAE(nn.Module):
    """变分自编码器"""
    def __init__(self, input_dim, hidden_dim=64, latent_dim=2):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_var = nn.Linear(hidden_dim // 2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


def vae_augment(X, y, config):
    """VAE数据增强"""
    X_aug_list = [X.copy()]
    y_aug_list = [y.copy()]
    
    classes = np.unique(y)
    for cls in classes:
        X_cls = X[y == cls]
        if len(X_cls) < 2:
            continue
        
        input_dim = X_cls.shape[1]
        vae = VAE(input_dim, config['vae_hidden_dim'], config['vae_latent_dim'])
        optimizer = torch.optim.Adam(vae.parameters(), lr=config['vae_lr'])
        
        X_tensor = torch.FloatTensor(X_cls)
        vae.train()
        for _ in range(config['vae_epochs']):
            optimizer.zero_grad()
            recon, mu, log_var = vae(X_tensor)
            recon_loss = nn.MSELoss()(recon, X_tensor)
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = recon_loss + config['vae_kl_weight'] * kl_loss / len(X_cls)
            loss.backward()
            optimizer.step()
        
        vae.eval()
        with torch.no_grad():
            recon, _, _ = vae(X_tensor)
            recon_np = recon.numpy()
        
        for alpha in np.linspace(0.1, 0.9, config['num_interpolation_points']):
            augmented = alpha * X_cls + (1 - alpha) * recon_np
            X_aug_list.append(augmented)
            y_aug_list.append(np.full(len(augmented), cls))
    
    return np.vstack(X_aug_list), np.hstack(y_aug_list)


# ==================== 单fold处理 ====================
def process_fold(fold_info, X, y, config):
    """处理单个fold（用于多进程）"""
    fold_idx, (train_idx, test_idx) = fold_info
    
    try:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # VAE增强
        X_aug, y_aug = vae_augment(X_train_scaled, y_train, config)
        
        # 随机森林分类
        clf = RandomForestClassifier(
            n_estimators=config['rf_n_estimators'],
            max_depth=config['rf_max_depth'],
            min_samples_split=config['rf_min_samples_split'],
            min_samples_leaf=config['rf_min_samples_leaf'],
            random_state=42,
            n_jobs=1
        )
        clf.fit(X_aug, y_aug)
        
        y_pred = clf.predict(X_test_scaled)
        y_prob = clf.predict_proba(X_test_scaled)
        
        return {
            'fold_idx': fold_idx,
            'accuracy': accuracy_score(y_test, y_pred),
            'y_true': y_test.tolist(),
            'y_pred': y_pred.tolist(),
            'y_prob': y_prob.tolist()
        }
    except Exception as e:
        return {'fold_idx': fold_idx, 'accuracy': 0.0, 'error': str(e)}


# ==================== 进度监控 ====================
class ProgressTracker:
    """进度追踪器"""
    def __init__(self, total, update_interval=10):
        self.total = total
        self.update_interval = update_interval
        self.start_time = time.time()
        self.completed = 0
    
    def update(self, n=1):
        self.completed += n
        if self.completed % self.update_interval == 0 or self.completed == self.total:
            elapsed = time.time() - self.start_time
            rate = self.completed / elapsed if elapsed > 0 else 0
            eta = (self.total - self.completed) / rate if rate > 0 else 0
            pct = self.completed / self.total * 100
            print(f"\r[进度] {self.completed}/{self.total} ({pct:.1f}%) | "
                  f"速度: {rate:.1f} folds/s | 剩余: {eta:.0f}s", end='', flush=True)


def run_parallel_cv(X, y, config, n_processes=None, logger=None):
    """并行运行Leave-P-Out交叉验证"""
    n_samples = len(X)
    p = config['leave_p_out']
    
    # 生成所有fold
    all_indices = np.arange(n_samples)
    test_combinations = list(combinations(all_indices, p))
    n_folds = len(test_combinations)
    
    logger.info(f"Leave-{p}-Out 交叉验证")
    logger.info(f"  样本数: {n_samples}, fold数: {n_folds}")
    
    # 准备fold信息
    fold_infos = []
    for fold_idx, test_idx in enumerate(test_combinations):
        test_idx = np.array(test_idx)
        train_idx = np.setdiff1d(all_indices, test_idx)
        fold_infos.append((fold_idx, (train_idx, test_idx)))
    
    # 确定进程数
    if n_processes is None:
        n_processes = min(mp.cpu_count(), n_folds, 64)
    
    logger.info(f"  使用 {n_processes} 个进程并行处理")
    
    start_time = time.time()
    
    # 并行处理（带进度显示）
    process_func = partial(process_fold, X=X, y=y, config=config)
    
    results = []
    tracker = ProgressTracker(n_folds, update_interval=max(1, n_folds // 20))
    
    with mp.Pool(processes=n_processes) as pool:
        for result in pool.imap_unordered(process_func, fold_infos, chunksize=max(1, n_folds // n_processes)):
            results.append(result)
            tracker.update()
    
    print()  # 换行
    
    elapsed_time = time.time() - start_time
    
    # 统计结果
    accuracies = [r['accuracy'] for r in results if 'error' not in r]
    errors = [r for r in results if 'error' in r]
    
    mean_acc = np.mean(accuracies) * 100
    std_acc = np.std(accuracies) * 100
    
    # 汇总预测
    all_y_true, all_y_pred, all_y_prob = [], [], []
    for r in results:
        if 'error' not in r:
            all_y_true.extend(r['y_true'])
            all_y_pred.extend(r['y_pred'])
            all_y_prob.extend([p[1] if len(p) > 1 else p[0] for p in r['y_prob']])
    
    overall_acc = accuracy_score(all_y_true, all_y_pred) * 100
    
    try:
        overall_auc = roc_auc_score(all_y_true, all_y_prob)
    except:
        overall_auc = None
    
    logger.info("=" * 60)
    logger.info(f"[结果] Leave-{p}-Out 完成")
    logger.info(f"  平均准确率: {mean_acc:.2f}% ± {std_acc:.2f}%")
    logger.info(f"  整体准确率: {overall_acc:.2f}%")
    if overall_auc:
        logger.info(f"  AUC: {overall_auc:.4f}")
    logger.info(f"  用时: {elapsed_time:.2f}秒")
    if errors:
        logger.warning(f"  错误fold数: {len(errors)}")
    logger.info("=" * 60)
    
    return {
        'mean_accuracy': mean_acc / 100,
        'std_accuracy': std_acc / 100,
        'overall_accuracy': overall_acc / 100,
        'overall_auc': overall_auc,
        'elapsed_time': elapsed_time,
        'n_folds': n_folds,
        'n_errors': len(errors),
        'config': config
    }


def main():
    """主函数"""
    # 配置
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(output_dir, f'run_{timestamp}.log')
    
    logger = setup_logging(log_file)
    
    logger.info("=" * 60)
    logger.info("VAE-HyperNetFusion 最终优化版 (5_final_optimized)")
    logger.info("=" * 60)
    
    # 优化后的配置（基于贝叶斯优化找到的最佳参数）
    config = {
        # VAE参数
        'vae_latent_dim': 2,
        'vae_hidden_dim': 64,
        'vae_epochs': 80,  # 增加epochs
        'vae_lr': 0.001,
        'vae_kl_weight': 0.5,
        'num_interpolation_points': 8,  # 增加插值点
        
        # 随机森林参数（优化后）
        'rf_n_estimators': 180,
        'rf_max_depth': 6,
        'rf_min_samples_split': 2,
        'rf_min_samples_leaf': 1,
        
        # 交叉验证
        'leave_p_out': 2
    }
    
    logger.info(f"配置: {json.dumps(config, indent=2)}")
    
    # 加载数据
    data_path = os.path.join(script_dir, 'data', 'Data_for_Jinming.csv')
    
    if not os.path.exists(data_path):
        logger.error(f"找不到数据: {data_path}")
        sys.exit(1)
    
    logger.info(f"加载数据: {data_path}")
    df = pd.read_csv(data_path)
    
    # 准备数据
    possible_targets = ['Class', 'class', 'label', 'target', 'Group', 'group']
    possible_ids = ['Sample ID', 'sample_id', 'ID', 'id']
    
    target_col = None
    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break
    
    exclude_cols = [target_col] + [col for col in possible_ids if col in df.columns]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].values.astype(np.float32)
    y_raw = df[target_col].values
    
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    logger.info(f"  特征: {feature_cols}")
    logger.info(f"  样本数: {len(X)}, 特征数: {X.shape[1]}")
    logger.info(f"  类别: {list(le.classes_)}, 分布: {list(np.bincount(y))}")
    
    # 确定进程数
    n_cpu = mp.cpu_count()
    n_processes = min(64, n_cpu)
    logger.info(f"  CPU核心数: {n_cpu}, 使用进程数: {n_processes}")
    
    # 运行实验
    results = run_parallel_cv(X, y, config, n_processes=n_processes, logger=logger)
    
    # 保存结果
    result_file = os.path.join(output_dir, f'final_results_{timestamp}.json')
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"结果已保存: {result_file}")
    logger.info(f"日志已保存: {log_file}")
    
    # 保存混淆矩阵
    # （需要重新计算，因为results里只有汇总数据）
    
    print(f"\n{'='*60}")
    print(f"实验完成！")
    print(f"  准确率: {results['mean_accuracy']*100:.2f}% ± {results['std_accuracy']*100:.2f}%")
    print(f"{'='*60}")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
