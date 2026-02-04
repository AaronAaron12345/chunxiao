#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12_rf_baseline_progress.py - 多GPU并行RF基线（带实时进度条）
============================================================
使用 VAE + GPU神经网络分类器 作为基线
（RF不支持GPU，用神经网络替代以实现GPU加速）

特点：
1. 6块GPU并行处理
2. 批处理多个fold提高GPU利用率
3. 实时进度条显示

运行: python 12_rf_baseline_progress.py
"""

import os
import sys
import json
import time
import logging
import warnings
import threading
from datetime import datetime
from itertools import combinations
from pathlib import Path
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

# 路径配置
SCRIPT_DIR = Path(__file__).parent
LOG_DIR = SCRIPT_DIR / 'logs'
OUTPUT_DIR = SCRIPT_DIR / 'output'
LOG_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# GPU配置
GPU_IDS = [0, 1, 2, 3, 4, 5]


# ==================== 模型定义 ====================
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


class MLPClassifier(nn.Module):
    """GPU上的MLP分类器（替代RF）"""
    def __init__(self, input_dim, hidden_dim=64, n_classes=2):
        super(MLPClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_classes)
        )
    
    def forward(self, x):
        return self.net(x)


class GPUWorker:
    """GPU工作线程"""
    def __init__(self, gpu_id, config):
        self.gpu_id = gpu_id
        self.device = torch.device(f'cuda:{gpu_id}')
        self.config = config
        self.results = []
        self.processed_count = 0
        self.lock = threading.Lock()
    
    def vae_augment(self, X, y):
        """GPU上的VAE增强"""
        X_aug_list = [torch.FloatTensor(X).to(self.device)]
        y_aug_list = [torch.LongTensor(y).to(self.device)]
        
        for cls in np.unique(y):
            X_cls = X[y == cls]
            if len(X_cls) < 2:
                continue
            
            vae = VAE(X_cls.shape[1], 64, 2).to(self.device)
            optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
            X_tensor = torch.FloatTensor(X_cls).to(self.device)
            
            vae.train()
            for _ in range(self.config['vae_epochs']):
                optimizer.zero_grad()
                recon, mu, log_var = vae(X_tensor)
                recon_loss = nn.MSELoss()(recon, X_tensor)
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = recon_loss + 0.001 * kl_loss / len(X_cls)
                loss.backward()
                optimizer.step()
            
            vae.eval()
            with torch.no_grad():
                recon = vae(X_tensor)[0]
            
            for alpha in np.linspace(0.1, 0.9, self.config['num_interp']):
                aug_data = alpha * X_tensor + (1 - alpha) * recon
                X_aug_list.append(aug_data)
                y_aug_list.append(torch.full((len(X_cls),), cls, dtype=torch.long, device=self.device))
        
        return torch.cat(X_aug_list), torch.cat(y_aug_list)
    
    def process_fold(self, fold_data):
        """处理单个fold"""
        fold_idx, X_train, y_train, X_test, y_test = fold_data
        
        try:
            # VAE增强
            X_aug, y_aug = self.vae_augment(X_train, y_train)
            
            # MLP分类器
            n_classes = len(np.unique(y_train))
            model = MLPClassifier(X_train.shape[1], self.config['hidden_dim'], n_classes).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
            criterion = nn.CrossEntropyLoss()
            
            # 训练
            model.train()
            for _ in range(self.config['epochs']):
                optimizer.zero_grad()
                outputs = model(X_aug)
                loss = criterion(outputs, y_aug)
                loss.backward()
                optimizer.step()
            
            # 评估
            model.eval()
            X_test_t = torch.FloatTensor(X_test).to(self.device)
            with torch.no_grad():
                outputs = model(X_test_t)
                y_pred = outputs.argmax(dim=1).cpu().numpy()
            
            acc = accuracy_score(y_test, y_pred)
            result = {
                'fold_idx': fold_idx,
                'accuracy': acc,
                'y_true': y_test.tolist(),
                'y_pred': y_pred.tolist(),
                'gpu_id': self.gpu_id
            }
        except Exception as e:
            result = {'fold_idx': fold_idx, 'accuracy': 0.0, 'error': str(e), 'gpu_id': self.gpu_id}
        
        with self.lock:
            self.results.append(result)
            self.processed_count += 1
        
        return result
    
    def process_batch(self, fold_batch):
        """批处理多个fold"""
        for fold_data in fold_batch:
            self.process_fold(fold_data)


def progress_monitor(workers, total, start_time, stop_event):
    """进度监控线程"""
    while not stop_event.is_set():
        current = sum(w.processed_count for w in workers)
        elapsed = time.time() - start_time
        
        if elapsed > 0:
            rate = current / elapsed
            eta = (total - current) / rate if rate > 0 else 0
            pct = current / total * 100
            
            bar_len = 40
            filled = int(bar_len * current / total)
            bar = '█' * filled + '░' * (bar_len - filled)
            
            gpu_counts = [f"G{w.gpu_id}:{w.processed_count}" for w in workers]
            gpu_str = ' '.join(gpu_counts)
            
            print(f'\r[{bar}] {current}/{total} ({pct:.1f}%) | '
                  f'{rate:.1f}/s | ETA: {eta:.0f}s | {gpu_str}', end='', flush=True)
        
        if current >= total:
            break
        
        time.sleep(0.3)
    print()


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
    logger.info("12_rf_baseline_progress.py - 多GPU并行基线（MLP替代RF）")
    logger.info("=" * 60)
    
    # 检查GPU
    if not torch.cuda.is_available():
        logger.error("CUDA不可用!")
        return
    
    n_gpus = torch.cuda.device_count()
    available_gpus = [i for i in GPU_IDS if i < n_gpus]
    logger.info(f"可用GPU: {available_gpus}")
    
    for gpu_id in available_gpus:
        logger.info(f"  GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    
    # 配置
    config = {
        'vae_epochs': 100,
        'num_interp': 5,
        'hidden_dim': 128,
        'lr': 0.001,
        'weight_decay': 0.0001,
        'epochs': 200
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
    
    # 准备fold数据（预先标准化）
    fold_datas = []
    for fold_idx, test_idx in enumerate(test_combos):
        test_idx = np.array(test_idx)
        train_idx = np.setdiff1d(all_indices, test_idx)
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        fold_datas.append((fold_idx, X_train_scaled, y_train, X_test_scaled, y_test))
    
    # 分配fold到各个GPU
    gpu_fold_batches = {gpu_id: [] for gpu_id in available_gpus}
    for i, fold_data in enumerate(fold_datas):
        gpu_id = available_gpus[i % len(available_gpus)]
        gpu_fold_batches[gpu_id].append(fold_data)
    
    logger.info(f"分配: {', '.join([f'GPU{g}={len(b)}' for g,b in gpu_fold_batches.items()])}")
    
    # 创建GPU工作器
    workers = [GPUWorker(gpu_id, config) for gpu_id in available_gpus]
    
    start_time = time.time()
    stop_event = threading.Event()
    
    # 启动进度监控
    monitor = threading.Thread(target=progress_monitor, args=(workers, n_folds, start_time, stop_event))
    monitor.start()
    
    # 多线程并行处理各GPU
    logger.info("开始多GPU并行处理...")
    print()
    
    with ThreadPoolExecutor(max_workers=len(available_gpus)) as executor:
        futures = []
        for worker in workers:
            batch = gpu_fold_batches[worker.gpu_id]
            futures.append(executor.submit(worker.process_batch, batch))
        
        for f in futures:
            f.result()
    
    stop_event.set()
    monitor.join(timeout=2)
    
    elapsed_time = time.time() - start_time
    
    # 收集结果
    all_results = []
    for worker in workers:
        all_results.extend(worker.results)
    
    valid_results = [r for r in all_results if 'error' not in r]
    error_results = [r for r in all_results if 'error' in r]
    accuracies = [r['accuracy'] for r in valid_results]
    
    mean_acc = np.mean(accuracies) * 100
    std_acc = np.std(accuracies) * 100
    
    all_y_true, all_y_pred = [], []
    for r in valid_results:
        all_y_true.extend(r['y_true'])
        all_y_pred.extend(r['y_pred'])
    
    overall_acc = accuracy_score(all_y_true, all_y_pred) * 100
    
    print()
    logger.info("=" * 60)
    logger.info("[结果] VAE + MLP 多GPU并行基线")
    logger.info("=" * 60)
    logger.info(f"  平均准确率: {mean_acc:.2f}% ± {std_acc:.2f}%")
    logger.info(f"  整体准确率: {overall_acc:.2f}%")
    logger.info(f"  成功folds: {len(valid_results)}/{n_folds}")
    logger.info(f"  失败folds: {len(error_results)}")
    logger.info(f"  总用时: {elapsed_time:.1f}秒")
    logger.info(f"  速度: {n_folds / elapsed_time:.1f} folds/秒")
    logger.info(f"  GPU分布: {dict((w.gpu_id, w.processed_count) for w in workers)}")
    
    # 保存结果
    result_file = OUTPUT_DIR / f'12_rf_baseline_{timestamp}.json'
    
    result_data = {
        'experiment': '12_rf_baseline (VAE + MLP 多GPU)',
        'method': 'VAE数据增强 + MLP分类器 (多GPU并行)',
        'timestamp': datetime.now().isoformat(),
        'mean_accuracy': mean_acc / 100,
        'std_accuracy': std_acc / 100,
        'overall_accuracy': overall_acc / 100,
        'elapsed_time': elapsed_time,
        'folds_per_second': n_folds / elapsed_time,
        'n_samples': n_samples,
        'n_folds': n_folds,
        'n_gpus': len(available_gpus),
        'config': config,
        'successful_folds': len(valid_results),
        'failed_folds': len(error_results)
    }
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"结果已保存: {result_file}")
    logger.info(f"日志已保存: {log_file}")


if __name__ == '__main__':
    main()
