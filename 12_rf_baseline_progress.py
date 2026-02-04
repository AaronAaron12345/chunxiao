#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12_rf_baseline_progress.py - RF基线对比（带实时进度条）
========================================================
使用 VAE + RandomForest 作为基线，对比 HyperNetFusion

特点：
1. 实时进度条显示
2. 多进程CPU并行（RF在CPU上更高效）
3. 每10个fold打印一次进度

运行: python 12_rf_baseline_progress.py
"""

import os
import sys
import json
import time
import logging
import warnings
import multiprocessing as mp
from multiprocessing import Manager
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
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

# 路径配置
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
    """VAE数据增强"""
    X_aug_list = [X.copy()]
    y_aug_list = [y.copy()]
    
    for cls in np.unique(y):
        X_cls = X[y == cls]
        if len(X_cls) < 2:
            continue
        
        vae = VAE(X_cls.shape[1], 64, 2)
        optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
        X_tensor = torch.FloatTensor(X_cls)
        
        vae.train()
        for _ in range(vae_epochs):
            optimizer.zero_grad()
            recon, mu, log_var = vae(X_tensor)
            recon_loss = nn.MSELoss()(recon, X_tensor)
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = recon_loss + 0.001 * kl_loss / len(X_cls)
            loss.backward()
            optimizer.step()
        
        vae.eval()
        with torch.no_grad():
            recon = vae(X_tensor)[0].numpy()
        
        for alpha in np.linspace(0.1, 0.9, num_interp):
            X_aug_list.append(alpha * X_cls + (1 - alpha) * recon)
            y_aug_list.append(np.full(len(X_cls), cls))
    
    return np.vstack(X_aug_list), np.hstack(y_aug_list)


# ==================== 单fold处理 (带进度更新) ====================
def process_fold_rf(args):
    """处理单个fold"""
    fold_info, X, y, config, counter, lock = args
    fold_idx, (train_idx, test_idx) = fold_info
    
    try:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # VAE增强
        X_aug, y_aug = vae_augment(X_train_scaled, y_train, config['vae_epochs'], config['num_interp'])
        
        # 随机森林
        clf = RandomForestClassifier(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            random_state=42,
            n_jobs=1
        )
        clf.fit(X_aug, y_aug)
        
        y_pred = clf.predict(X_test_scaled)
        
        result = {
            'fold_idx': fold_idx,
            'accuracy': accuracy_score(y_test, y_pred),
            'y_true': y_test.tolist(),
            'y_pred': y_pred.tolist()
        }
        
    except Exception as e:
        result = {'fold_idx': fold_idx, 'accuracy': 0.0, 'error': str(e)}
    
    # 更新进度计数器
    with lock:
        counter.value += 1
    
    return result


def progress_monitor(counter, total, start_time, results_queue, stop_event):
    """进度监控进程"""
    last_count = 0
    accuracies = []
    
    while not stop_event.is_set():
        current = counter.value
        if current != last_count:
            elapsed = time.time() - start_time
            rate = current / elapsed if elapsed > 0 else 0
            eta = (total - current) / rate if rate > 0 else 0
            pct = current / total * 100
            
            # 打印进度条
            bar_len = 30
            filled = int(bar_len * current / total)
            bar = '█' * filled + '░' * (bar_len - filled)
            
            print(f'\r[{bar}] {current}/{total} ({pct:.1f}%) | '
                  f'速度: {rate:.1f}/s | 剩余: {eta:.0f}s', end='', flush=True)
            
            last_count = current
        
        if current >= total:
            break
        
        time.sleep(0.5)
    
    print()  # 换行


def main():
    # 设置日志
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR / f'12_rf_baseline_{timestamp}.log'
    
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
    logger.info("12_rf_baseline_progress.py - RF基线对比（带进度条）")
    logger.info("=" * 60)
    
    # 配置
    config = {
        'vae_epochs': 100,
        'num_interp': 5,
        'n_estimators': 150,
        'max_depth': 5
    }
    
    logger.info(f"配置: {config}")
    
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
    leave_p_out = 2
    test_combos = list(combinations(all_indices, leave_p_out))
    n_folds = len(test_combos)
    
    logger.info(f"Leave-{leave_p_out}-Out: {n_folds} 个 folds")
    
    fold_infos = []
    for fold_idx, test_idx in enumerate(test_combos):
        test_idx = np.array(test_idx)
        train_idx = np.setdiff1d(all_indices, test_idx)
        fold_infos.append((fold_idx, (train_idx, test_idx)))
    
    # 多进程设置
    n_processes = min(mp.cpu_count(), 64)
    logger.info(f"使用 {n_processes} 个CPU进程")
    
    # 创建共享计数器
    manager = Manager()
    counter = manager.Value('i', 0)
    lock = manager.Lock()
    stop_event = manager.Event()
    results_queue = manager.Queue()
    
    start_time = time.time()
    
    # 启动进度监控进程
    monitor = mp.Process(target=progress_monitor, args=(counter, n_folds, start_time, results_queue, stop_event))
    monitor.start()
    
    # 准备参数
    args_list = [(fold_info, X, y, config, counter, lock) for fold_info in fold_infos]
    
    # 并行处理
    logger.info("开始处理...")
    print()  # 为进度条留空行
    
    with mp.Pool(processes=n_processes) as pool:
        results = pool.map(process_fold_rf, args_list)
    
    # 停止进度监控
    stop_event.set()
    monitor.join(timeout=2)
    
    elapsed_time = time.time() - start_time
    
    # 统计结果
    valid_results = [r for r in results if 'error' not in r]
    error_results = [r for r in results if 'error' in r]
    accuracies = [r['accuracy'] for r in valid_results]
    
    mean_acc = np.mean(accuracies) * 100
    std_acc = np.std(accuracies) * 100
    
    # 汇总预测
    all_y_true, all_y_pred = [], []
    for r in valid_results:
        all_y_true.extend(r['y_true'])
        all_y_pred.extend(r['y_pred'])
    
    overall_acc = accuracy_score(all_y_true, all_y_pred) * 100
    
    print()
    logger.info("=" * 60)
    logger.info("[结果] VAE + RandomForest 基线")
    logger.info("=" * 60)
    logger.info(f"  平均准确率: {mean_acc:.2f}% ± {std_acc:.2f}%")
    logger.info(f"  整体准确率: {overall_acc:.2f}%")
    logger.info(f"  成功folds: {len(valid_results)}/{n_folds}")
    logger.info(f"  失败folds: {len(error_results)}")
    logger.info(f"  总用时: {elapsed_time:.1f}秒")
    logger.info(f"  速度: {n_folds / elapsed_time:.1f} folds/秒")
    
    # 保存结果
    result_file = OUTPUT_DIR / f'12_rf_baseline_{timestamp}.json'
    
    result_data = {
        'experiment': '12_rf_baseline (VAE + RF)',
        'method': 'VAE数据增强 + RandomForest (CPU多进程)',
        'timestamp': datetime.now().isoformat(),
        'mean_accuracy': mean_acc / 100,
        'std_accuracy': std_acc / 100,
        'overall_accuracy': overall_acc / 100,
        'elapsed_time': elapsed_time,
        'folds_per_second': n_folds / elapsed_time,
        'n_samples': n_samples,
        'n_folds': n_folds,
        'n_processes': n_processes,
        'config': config,
        'successful_folds': len(valid_results),
        'failed_folds': len(error_results)
    }
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"结果已保存: {result_file}")
    logger.info(f"日志已保存: {log_file}")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
