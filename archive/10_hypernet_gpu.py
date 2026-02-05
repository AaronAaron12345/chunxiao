#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_hypernet_gpu.py - VAE-HyperNetFusion GPU加速版
===================================================
使用GPU加速训练，大幅提升速度

服务器GPU: 8 x NVIDIA A100-SXM4-80GB
建议使用: GPU 0-5 (GPU 6,7 被VLLM占用)

运行: CUDA_VISIBLE_DEVICES=0 python 10_hypernet_gpu.py
或多GPU: CUDA_VISIBLE_DEVICES=0,1,2,3 python 10_hypernet_gpu.py

日志输出:
- logs/10_hypernet_gpu_YYYYMMDD_HHMMSS.log
- output/10_hypernet_gpu_YYYYMMDD_HHMMSS.json
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
from torch.utils.data import DataLoader, TensorDataset
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
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==================== VAE 数据增强 (GPU) ====================
class VAE(nn.Module):
    """变分自编码器 - GPU版"""
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
    
    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = self.fc_mu(h), self.fc_var(h)
        std = torch.exp(0.5 * log_var)
        z = mu + std * torch.randn_like(std)
        return self.decoder(z), mu, log_var


def vae_augment_gpu(X, y, device, vae_epochs=100, num_interp=5):
    """VAE数据增强 - GPU加速"""
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


# ==================== HyperNetFusion (GPU) ====================
class TargetNetwork(nn.Module):
    """目标网络 - 权重由HyperNetwork生成"""
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
    """超网络 - 生成目标网络的权重"""
    def __init__(self, input_dim, hidden_dim, target_weight_size):
        super(HyperNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, target_weight_size)
        )
    
    def forward(self, x):
        return self.net(x)


class HyperNetFusion(nn.Module):
    """HyperNetFusion - 你的原创方法 (GPU版)"""
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


def train_hypernet_gpu(X_train, y_train, device, config):
    """训练HyperNetFusion模型 - GPU加速"""
    input_dim = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    
    model = HyperNetFusion(
        input_dim=input_dim,
        hyper_hidden=config['hyper_hidden'],
        target_hidden=config['target_hidden'],
        output_dim=n_classes
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    
    X_tensor = torch.FloatTensor(X_train).to(device)
    y_tensor = torch.LongTensor(y_train).to(device)
    
    model.train()
    for epoch in range(config['epochs']):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
    
    return model


def predict_hypernet_gpu(model, X_test, device):
    """用HyperNetFusion预测 - GPU"""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).to(device)
        outputs = model(X_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.cpu().numpy()


# ==================== 批量处理所有folds (GPU加速) ====================
def process_all_folds_gpu(X, y, fold_infos, device, config, logger):
    """GPU加速处理所有folds"""
    results = []
    total = len(fold_infos)
    
    start_time = time.time()
    
    for i, (fold_idx, (train_idx, test_idx)) in enumerate(fold_infos):
        try:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # 标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # VAE增强 (GPU)
            X_aug, y_aug = vae_augment_gpu(
                X_train_scaled, y_train, device,
                vae_epochs=config['vae_epochs'],
                num_interp=config['num_interp']
            )
            
            # 训练HyperNetFusion (GPU)
            model = train_hypernet_gpu(X_aug, y_aug, device, config)
            
            # 预测
            y_pred = predict_hypernet_gpu(model, X_test_scaled, device)
            
            results.append({
                'fold_idx': fold_idx,
                'accuracy': accuracy_score(y_test, y_pred),
                'y_true': y_test.tolist(),
                'y_pred': y_pred.tolist()
            })
            
        except Exception as e:
            results.append({'fold_idx': fold_idx, 'accuracy': 0.0, 'error': str(e)})
        
        # 进度显示
        if (i + 1) % 50 == 0 or i + 1 == total:
            accs = [r['accuracy'] for r in results if 'error' not in r]
            curr_acc = np.mean(accs) * 100 if accs else 0
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (total - i - 1)
            logger.info(f"[{i+1}/{total}] 准确率: {curr_acc:.2f}% | 已用: {elapsed:.0f}s | 剩余: {eta:.0f}s")
        
        # 清理GPU内存
        if (i + 1) % 100 == 0:
            torch.cuda.empty_cache()
    
    return results


def main():
    # 设置日志
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR / f'10_hypernet_gpu_{timestamp}.log'
    
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
    logger.info("10_hypernet_gpu.py - VAE-HyperNetFusion GPU加速版")
    logger.info("=" * 60)
    
    # GPU信息
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"✅ GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        logger.info(f"CUDA版本: {torch.version.cuda}")
    else:
        logger.warning("⚠️ GPU不可用，使用CPU")
    
    logger.info(f"设备: {DEVICE}")
    logger.info(f"日志文件: {log_file}")
    
    # 配置（经过优化的超参数）
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
    
    start_time = time.time()
    
    # GPU处理所有folds
    results = process_all_folds_gpu(X, y, fold_infos, DEVICE, config, logger)
    
    elapsed_time = time.time() - start_time
    
    # 统计结果
    valid_results = [r for r in results if 'error' not in r]
    error_results = [r for r in results if 'error' in r]
    accuracies = [r['accuracy'] for r in valid_results]
    
    mean_acc = np.mean(accuracies) * 100
    std_acc = np.std(accuracies) * 100
    
    # 汇总预测计算整体准确率
    all_y_true, all_y_pred = [], []
    for r in valid_results:
        all_y_true.extend(r['y_true'])
        all_y_pred.extend(r['y_pred'])
    
    overall_acc = accuracy_score(all_y_true, all_y_pred) * 100
    
    logger.info("=" * 60)
    logger.info("[结果] VAE-HyperNetFusion GPU版")
    logger.info("=" * 60)
    logger.info(f"  平均准确率: {mean_acc:.2f}% ± {std_acc:.2f}%")
    logger.info(f"  整体准确率: {overall_acc:.2f}%")
    logger.info(f"  成功folds: {len(valid_results)}/{n_folds}")
    logger.info(f"  失败folds: {len(error_results)}")
    logger.info(f"  总用时: {elapsed_time:.1f}秒")
    logger.info(f"  速度: {n_folds / elapsed_time:.1f} folds/秒")
    
    # 保存结果
    result_file = OUTPUT_DIR / f'10_hypernet_gpu_{timestamp}.json'
    
    result_data = {
        'experiment': '10_hypernet_gpu (VAE-HyperNetFusion GPU版)',
        'method': 'VAE数据增强 + HyperNetwork (GPU加速)',
        'timestamp': datetime.now().isoformat(),
        'device': str(DEVICE),
        'mean_accuracy': mean_acc / 100,
        'std_accuracy': std_acc / 100,
        'overall_accuracy': overall_acc / 100,
        'elapsed_time': elapsed_time,
        'folds_per_second': n_folds / elapsed_time,
        'n_samples': n_samples,
        'n_folds': n_folds,
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
