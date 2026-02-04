#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
13_hypernet_gpu_optimized.py - 优化的GPU版HyperNetFusion（带进度条）
====================================================================
改进点：
1. 实时进度条显示
2. 单GPU处理（避免多进程开销）
3. 更好的超参数配置
4. 详细日志

运行: CUDA_VISIBLE_DEVICES=0 python 13_hypernet_gpu_optimized.py
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


class HyperNetwork(nn.Module):
    """超网络：从训练数据统计量生成目标网络权重"""
    def __init__(self, input_dim, hidden_dim, target_hidden, n_classes):
        super(HyperNetwork, self).__init__()
        # 输入：特征均值、标准差、协方差等统计量
        stat_dim = input_dim * 2 + input_dim * input_dim
        
        self.net = nn.Sequential(
            nn.Linear(stat_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU()
        )
        
        # 生成 TargetNetwork 的权重
        # Layer 1: input_dim -> target_hidden
        self.gen_w1 = nn.Linear(hidden_dim // 2, input_dim * target_hidden)
        self.gen_b1 = nn.Linear(hidden_dim // 2, target_hidden)
        # Layer 2: target_hidden -> n_classes
        self.gen_w2 = nn.Linear(hidden_dim // 2, target_hidden * n_classes)
        self.gen_b2 = nn.Linear(hidden_dim // 2, n_classes)
        
        self.input_dim = input_dim
        self.target_hidden = target_hidden
        self.n_classes = n_classes
    
    def forward(self, stats):
        h = self.net(stats)
        w1 = self.gen_w1(h).view(-1, self.input_dim, self.target_hidden)
        b1 = self.gen_b1(h).view(-1, self.target_hidden)
        w2 = self.gen_w2(h).view(-1, self.target_hidden, self.n_classes)
        b2 = self.gen_b2(h).view(-1, self.n_classes)
        return w1, b1, w2, b2


class TargetNetwork(nn.Module):
    """目标网络：使用超网络生成的权重进行分类"""
    @staticmethod
    def forward_with_weights(x, w1, b1, w2, b2):
        # x: (batch_size, input_dim)
        # w1: (batch_size, input_dim, hidden) or (input_dim, hidden)
        if w1.dim() == 3:
            # Batch mode
            h = torch.bmm(x.unsqueeze(1), w1).squeeze(1) + b1
            h = torch.relu(h)
            out = torch.bmm(h.unsqueeze(1), w2).squeeze(1) + b2
        else:
            # Single mode
            h = torch.mm(x, w1) + b1
            h = torch.relu(h)
            out = torch.mm(h, w2) + b2
        return out


def compute_stats(X, device):
    """计算数据统计量作为超网络输入"""
    X_t = torch.FloatTensor(X).to(device)
    mean = X_t.mean(dim=0)
    std = X_t.std(dim=0) + 1e-6
    
    # 计算协方差矩阵（归一化）
    X_centered = X_t - mean
    cov = torch.mm(X_centered.T, X_centered) / (len(X_t) - 1)
    cov_flat = cov.flatten()
    
    # 合并所有统计量
    stats = torch.cat([mean, std, cov_flat])
    return stats.unsqueeze(0)


def vae_augment_gpu(X, y, vae_epochs, num_interp, device):
    """GPU上的VAE数据增强"""
    X_aug_list = [torch.FloatTensor(X).to(device)]
    y_aug_list = [torch.LongTensor(y).to(device)]
    
    for cls in np.unique(y):
        X_cls = X[y == cls]
        if len(X_cls) < 2:
            continue
        
        vae = VAE(X_cls.shape[1], 64, 2).to(device)
        optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
        X_tensor = torch.FloatTensor(X_cls).to(device)
        
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
            recon = vae(X_tensor)[0]
        
        for alpha in np.linspace(0.1, 0.9, num_interp):
            aug_data = alpha * X_tensor + (1 - alpha) * recon
            X_aug_list.append(aug_data)
            y_aug_list.append(torch.full((len(X_cls),), cls, dtype=torch.long, device=device))
    
    return torch.cat(X_aug_list), torch.cat(y_aug_list)


def train_hypernet_fold(X_train, y_train, X_test, y_test, config, device):
    """训练单个fold的HyperNetFusion"""
    input_dim = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    
    # VAE增强
    X_aug, y_aug = vae_augment_gpu(X_train, y_train, config['vae_epochs'], config['num_interp'], device)
    
    # 创建超网络
    hypernet = HyperNetwork(
        input_dim=input_dim,
        hidden_dim=config['hyper_hidden'],
        target_hidden=config['target_hidden'],
        n_classes=n_classes
    ).to(device)
    
    optimizer = optim.Adam(hypernet.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    
    # 计算训练数据统计量
    stats = compute_stats(X_train, device)
    
    # 训练超网络
    hypernet.train()
    for epoch in range(config['epochs']):
        optimizer.zero_grad()
        
        # 生成目标网络权重
        w1, b1, w2, b2 = hypernet(stats)
        
        # 对增强后的数据前向传播
        outputs = TargetNetwork.forward_with_weights(X_aug, w1[0], b1[0], w2[0], b2[0])
        loss = criterion(outputs, y_aug)
        
        loss.backward()
        optimizer.step()
    
    # 评估
    hypernet.eval()
    with torch.no_grad():
        w1, b1, w2, b2 = hypernet(stats)
        X_test_t = torch.FloatTensor(X_test).to(device)
        outputs = TargetNetwork.forward_with_weights(X_test_t, w1[0], b1[0], w2[0], b2[0])
        y_pred = outputs.argmax(dim=1).cpu().numpy()
    
    return y_pred


def print_progress_bar(current, total, elapsed, prefix='', length=40):
    """打印进度条"""
    percent = current / total
    filled = int(length * percent)
    bar = '█' * filled + '░' * (length - filled)
    
    rate = current / elapsed if elapsed > 0 else 0
    eta = (total - current) / rate if rate > 0 else 0
    
    print(f'\r{prefix} [{bar}] {current}/{total} ({percent*100:.1f}%) | '
          f'{rate:.1f}/s | ETA: {eta:.0f}s', end='', flush=True)


def main():
    # 设置日志
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR / f'13_hypernet_gpu_{timestamp}.log'
    
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
    logger.info("13_hypernet_gpu_optimized.py - GPU版HyperNetFusion")
    logger.info("=" * 60)
    
    # GPU设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("警告：未检测到GPU，使用CPU")
    
    # 配置 - 优化后的参数
    config = {
        'hyper_hidden': 128,      # 增大超网络
        'target_hidden': 64,      # 增大目标网络
        'lr': 0.001,              # 降低学习率
        'weight_decay': 0.0005,
        'epochs': 300,            # 增加训练轮数
        'vae_epochs': 150,        # 增加VAE训练
        'num_interp': 7           # 增加插值数量
    }
    
    logger.info(f"配置: {config}")
    
    # 加载数据
    data_path = SCRIPT_DIR / 'data' / 'Data_for_Jinming.csv'
    df = pd.read_csv(data_path)
    feature_cols = ['LAA', 'Glutamate', 'Choline', 'Sarcosine']
    X = df[feature_cols].values.astype(np.float32)
    y_raw = df['Group'].values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    n_samples = len(X)
    logger.info(f"数据: {n_samples} 样本, {X.shape[1]} 特征")
    logger.info(f"类别: {le.classes_}")
    
    # 生成所有fold
    all_indices = np.arange(n_samples)
    leave_p_out = 2
    test_combos = list(combinations(all_indices, leave_p_out))
    n_folds = len(test_combos)
    
    logger.info(f"Leave-{leave_p_out}-Out: {n_folds} 个 folds")
    
    # 开始处理
    results = []
    all_y_true = []
    all_y_pred = []
    
    start_time = time.time()
    logger.info("开始处理...")
    print()  # 进度条换行
    
    for fold_idx, test_idx in enumerate(test_combos):
        test_idx = np.array(test_idx)
        train_idx = np.setdiff1d(all_indices, test_idx)
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        try:
            y_pred = train_hypernet_fold(X_train_scaled, y_train, X_test_scaled, y_test, config, device)
            acc = accuracy_score(y_test, y_pred)
            
            results.append({'fold_idx': fold_idx, 'accuracy': acc})
            all_y_true.extend(y_test.tolist())
            all_y_pred.extend(y_pred.tolist())
            
        except Exception as e:
            results.append({'fold_idx': fold_idx, 'accuracy': 0.0, 'error': str(e)})
        
        # 更新进度条
        elapsed = time.time() - start_time
        print_progress_bar(fold_idx + 1, n_folds, elapsed, prefix='进度')
        
        # 每50个fold打印一次当前准确率
        if (fold_idx + 1) % 50 == 0:
            current_acc = accuracy_score(all_y_true, all_y_pred) * 100
            print(f' | 当前准确率: {current_acc:.2f}%', end='')
    
    print()  # 进度条结束换行
    
    elapsed_time = time.time() - start_time
    
    # 统计结果
    valid_results = [r for r in results if 'error' not in r]
    error_results = [r for r in results if 'error' in r]
    accuracies = [r['accuracy'] for r in valid_results]
    
    mean_acc = np.mean(accuracies) * 100
    std_acc = np.std(accuracies) * 100
    overall_acc = accuracy_score(all_y_true, all_y_pred) * 100
    
    print()
    logger.info("=" * 60)
    logger.info("[结果] VAE-HyperNetFusion GPU优化版")
    logger.info("=" * 60)
    logger.info(f"  平均准确率: {mean_acc:.2f}% ± {std_acc:.2f}%")
    logger.info(f"  整体准确率: {overall_acc:.2f}%")
    logger.info(f"  成功folds: {len(valid_results)}/{n_folds}")
    logger.info(f"  失败folds: {len(error_results)}")
    logger.info(f"  总用时: {elapsed_time:.1f}秒")
    logger.info(f"  速度: {n_folds / elapsed_time:.1f} folds/秒")
    
    # 保存结果
    result_file = OUTPUT_DIR / f'13_hypernet_gpu_{timestamp}.json'
    
    result_data = {
        'experiment': '13_hypernet_gpu_optimized',
        'method': 'VAE-HyperNetFusion (GPU单进程优化)',
        'timestamp': datetime.now().isoformat(),
        'mean_accuracy': mean_acc / 100,
        'std_accuracy': std_acc / 100,
        'overall_accuracy': overall_acc / 100,
        'elapsed_time': elapsed_time,
        'folds_per_second': n_folds / elapsed_time,
        'n_samples': n_samples,
        'n_folds': n_folds,
        'config': config,
        'device': str(device),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
        'successful_folds': len(valid_results),
        'failed_folds': len(error_results)
    }
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"结果已保存: {result_file}")
    logger.info(f"日志已保存: {log_file}")


if __name__ == '__main__':
    main()
