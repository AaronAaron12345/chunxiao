#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11_hypernet_multi_gpu.py - VAE-HyperNetFusion 多GPU并行版
==========================================================
使用6块GPU (0-5) 并行处理，每块GPU处理一部分folds

运行: python 11_hypernet_multi_gpu.py

日志输出:
- logs/11_hypernet_multi_gpu_YYYYMMDD_HHMMSS.log
- output/11_hypernet_multi_gpu_YYYYMMDD_HHMMSS.json
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

# 使用的GPU列表 (0-5, 避开6,7被VLLM占用)
GPU_IDS = [0, 1, 2, 3, 4, 5]


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


def vae_augment_gpu(X, y, device, vae_epochs=100, num_interp=5):
    X_aug_list = [X.copy()]
    y_aug_list = [y.copy()]
    
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
            recon = vae(X_tensor)[0].cpu().numpy()
        
        for alpha in np.linspace(0.1, 0.9, num_interp):
            X_aug_list.append(alpha * X_cls + (1 - alpha) * recon)
            y_aug_list.append(np.full(len(X_cls), cls))
    
    return np.vstack(X_aug_list), np.hstack(y_aug_list)


# ==================== HyperNetFusion ====================
class TargetNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TargetNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1_weight = None
        self.fc1_bias = None
        self.fc2_weight = None
        self.fc2_bias = None
    
    def set_weights(self, weights):
        idx = 0
        size = self.input_dim * self.hidden_dim
        self.fc1_weight = weights[idx:idx+size].view(self.hidden_dim, self.input_dim)
        idx += size
        self.fc1_bias = weights[idx:idx+self.hidden_dim]
        idx += self.hidden_dim
        size = self.hidden_dim * self.output_dim
        self.fc2_weight = weights[idx:idx+size].view(self.output_dim, self.hidden_dim)
        idx += size
        self.fc2_bias = weights[idx:idx+self.output_dim]
    
    def forward(self, x):
        x = torch.relu(torch.mm(x, self.fc1_weight.t()) + self.fc1_bias)
        x = torch.mm(x, self.fc2_weight.t()) + self.fc2_bias
        return x
    
    def get_weight_size(self):
        return (self.input_dim * self.hidden_dim + self.hidden_dim +
                self.hidden_dim * self.output_dim + self.output_dim)


class HyperNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, target_weight_size):
        super(HyperNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, target_weight_size)
        )
    
    def forward(self, x):
        return self.net(x)


class HyperNetFusion(nn.Module):
    def __init__(self, input_dim, hyper_hidden=64, target_hidden=32, output_dim=2):
        super(HyperNetFusion, self).__init__()
        self.target_net = TargetNetwork(input_dim, target_hidden, output_dim)
        target_weight_size = self.target_net.get_weight_size()
        self.hyper_net = HyperNetwork(input_dim, hyper_hidden, target_weight_size)
    
    def forward(self, x):
        condition = x.mean(dim=0, keepdim=True).expand(x.size(0), -1)
        weights = self.hyper_net(condition)
        self.target_net.set_weights(weights[0])
        return self.target_net(x)


# ==================== GPU Worker ====================
def gpu_worker(args):
    """单个GPU worker，处理分配给它的folds"""
    gpu_id, fold_batch, X, y, config = args
    
    # 设置该进程只看到指定的GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = []
    
    for fold_idx, (train_idx, test_idx) in fold_batch:
        try:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # 标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # VAE增强
            X_aug, y_aug = vae_augment_gpu(
                X_train_scaled, y_train, device,
                vae_epochs=config['vae_epochs'],
                num_interp=config['num_interp']
            )
            
            # 训练HyperNetFusion
            input_dim = X_aug.shape[1]
            n_classes = len(np.unique(y_aug))
            
            model = HyperNetFusion(
                input_dim=input_dim,
                hyper_hidden=config['hyper_hidden'],
                target_hidden=config['target_hidden'],
                output_dim=n_classes
            ).to(device)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
            
            X_tensor = torch.FloatTensor(X_aug).to(device)
            y_tensor = torch.LongTensor(y_aug).to(device)
            
            model.train()
            for _ in range(config['epochs']):
                optimizer.zero_grad()
                outputs = model(X_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()
            
            # 预测
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
                outputs = model(X_test_tensor)
                _, predicted = torch.max(outputs, 1)
                y_pred = predicted.cpu().numpy()
            
            results.append({
                'fold_idx': fold_idx,
                'accuracy': accuracy_score(y_test, y_pred),
                'y_true': y_test.tolist(),
                'y_pred': y_pred.tolist(),
                'gpu_id': gpu_id
            })
            
        except Exception as e:
            results.append({'fold_idx': fold_idx, 'accuracy': 0.0, 'error': str(e), 'gpu_id': gpu_id})
        
        # 清理GPU内存
        torch.cuda.empty_cache()
    
    return results


def main():
    # 设置日志
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR / f'11_hypernet_multi_gpu_{timestamp}.log'
    
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
    logger.info("11_hypernet_multi_gpu.py - 多GPU并行版")
    logger.info("=" * 60)
    
    # 检查GPU
    n_gpus = torch.cuda.device_count()
    logger.info(f"检测到 {n_gpus} 块GPU")
    
    available_gpus = [g for g in GPU_IDS if g < n_gpus]
    logger.info(f"使用 GPU: {available_gpus}")
    
    for gpu_id in available_gpus:
        name = torch.cuda.get_device_name(gpu_id)
        mem = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
        logger.info(f"  GPU {gpu_id}: {name} ({mem:.1f} GB)")
    
    # 配置
    config = {
        'hyper_hidden': 64,
        'target_hidden': 32,
        'lr': 0.005,
        'weight_decay': 0.001,
        'epochs': 150,
        'vae_epochs': 100,
        'num_interp': 5
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
    
    # 将folds分配给各个GPU
    n_gpus_used = len(available_gpus)
    fold_batches = [[] for _ in range(n_gpus_used)]
    for i, fold_info in enumerate(fold_infos):
        fold_batches[i % n_gpus_used].append(fold_info)
    
    logger.info(f"分配: 每块GPU约处理 {n_folds // n_gpus_used} 个folds")
    
    start_time = time.time()
    
    # 多进程并行（每个进程用一块GPU）
    logger.info("开始多GPU并行处理...")
    
    args_list = [
        (available_gpus[i], fold_batches[i], X, y, config)
        for i in range(n_gpus_used)
    ]
    
    # 使用spawn方法启动进程
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=n_gpus_used) as pool:
        all_results = pool.map(gpu_worker, args_list)
    
    # 合并结果
    results = []
    for batch_results in all_results:
        results.extend(batch_results)
    
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
    
    # GPU使用统计
    gpu_counts = {}
    for r in valid_results:
        gid = r.get('gpu_id', 'unknown')
        gpu_counts[gid] = gpu_counts.get(gid, 0) + 1
    
    logger.info("=" * 60)
    logger.info("[结果] VAE-HyperNetFusion 多GPU并行版")
    logger.info("=" * 60)
    logger.info(f"  平均准确率: {mean_acc:.2f}% ± {std_acc:.2f}%")
    logger.info(f"  整体准确率: {overall_acc:.2f}%")
    logger.info(f"  成功folds: {len(valid_results)}/{n_folds}")
    logger.info(f"  失败folds: {len(error_results)}")
    logger.info(f"  总用时: {elapsed_time:.1f}秒")
    logger.info(f"  速度: {n_folds / elapsed_time:.1f} folds/秒")
    logger.info(f"  GPU分布: {gpu_counts}")
    
    # 保存结果
    result_file = OUTPUT_DIR / f'11_hypernet_multi_gpu_{timestamp}.json'
    
    result_data = {
        'experiment': '11_hypernet_multi_gpu (多GPU并行)',
        'method': 'VAE-HyperNetFusion (6块A100并行)',
        'timestamp': datetime.now().isoformat(),
        'gpus_used': available_gpus,
        'mean_accuracy': mean_acc / 100,
        'std_accuracy': std_acc / 100,
        'overall_accuracy': overall_acc / 100,
        'elapsed_time': elapsed_time,
        'folds_per_second': n_folds / elapsed_time,
        'n_samples': n_samples,
        'n_folds': n_folds,
        'config': config,
        'successful_folds': len(valid_results),
        'failed_folds': len(error_results),
        'gpu_distribution': gpu_counts
    }
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"结果已保存: {result_file}")
    logger.info(f"日志已保存: {log_file}")


if __name__ == '__main__':
    main()
