#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
9_hypernet_bayes.py - VAE-HyperNetFusion + 每fold贝叶斯优化
============================================================
这是你的原创方法：VAE数据增强 + HyperNetwork生成分类器权重

特点：
1. VAE 数据增强（你的创新点）
2. HyperNetwork 生成 TargetNetwork 的权重（你的创新点）
3. 每个fold做贝叶斯优化寻找最佳超参数

运行: python 9_hypernet_bayes.py

日志输出:
- logs/9_hypernet_bayes_YYYYMMDD_HHMMSS.log
- output/9_hypernet_bayes_YYYYMMDD_HHMMSS.json
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
from skopt import gp_minimize
from skopt.space import Real, Integer

warnings.filterwarnings('ignore')

# 路径配置
SCRIPT_DIR = Path(__file__).parent
LOG_DIR = SCRIPT_DIR / 'logs'
OUTPUT_DIR = SCRIPT_DIR / 'output'
LOG_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ==================== VAE 数据增强 ====================
class VAE(nn.Module):
    """变分自编码器 - 用于数据增强"""
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


# ==================== HyperNetFusion (你的原创方法) ====================
class TargetNetwork(nn.Module):
    """目标网络 - 权重由HyperNetwork生成"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TargetNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 权重占位符（将由HyperNetwork填充）
        self.fc1_weight = None
        self.fc1_bias = None
        self.fc2_weight = None
        self.fc2_bias = None
    
    def set_weights(self, weights):
        """设置由HyperNetwork生成的权重"""
        idx = 0
        # fc1 weights
        size = self.input_dim * self.hidden_dim
        self.fc1_weight = weights[idx:idx+size].view(self.hidden_dim, self.input_dim)
        idx += size
        # fc1 bias
        self.fc1_bias = weights[idx:idx+self.hidden_dim]
        idx += self.hidden_dim
        # fc2 weights
        size = self.hidden_dim * self.output_dim
        self.fc2_weight = weights[idx:idx+size].view(self.output_dim, self.hidden_dim)
        idx += size
        # fc2 bias
        self.fc2_bias = weights[idx:idx+self.output_dim]
    
    def forward(self, x):
        x = torch.relu(torch.mm(x, self.fc1_weight.t()) + self.fc1_bias)
        x = torch.mm(x, self.fc2_weight.t()) + self.fc2_bias
        return x
    
    def get_weight_size(self):
        """计算需要的总权重数量"""
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
    """HyperNetFusion - 你的原创方法"""
    def __init__(self, input_dim, hyper_hidden=64, target_hidden=32, output_dim=2):
        super(HyperNetFusion, self).__init__()
        self.target_net = TargetNetwork(input_dim, target_hidden, output_dim)
        target_weight_size = self.target_net.get_weight_size()
        self.hyper_net = HyperNetwork(input_dim, hyper_hidden, target_weight_size)
    
    def forward(self, x):
        # 用输入的均值作为HyperNetwork的条件输入
        condition = x.mean(dim=0, keepdim=True).expand(x.size(0), -1)
        # HyperNetwork生成权重
        weights = self.hyper_net(condition)
        # 使用生成的权重进行分类
        self.target_net.set_weights(weights[0])
        return self.target_net(x)


def train_hypernet(X_train, y_train, config):
    """训练HyperNetFusion模型"""
    input_dim = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    
    model = HyperNetFusion(
        input_dim=input_dim,
        hyper_hidden=config['hyper_hidden'],
        target_hidden=config['target_hidden'],
        output_dim=n_classes
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.LongTensor(y_train)
    
    model.train()
    for epoch in range(config['epochs']):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
    
    return model


def predict_hypernet(model, X_test):
    """用HyperNetFusion预测"""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test)
        outputs = model(X_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.numpy()


# ==================== 每fold贝叶斯优化 ====================
def evaluate_config(params, X_train, y_train, X_val, y_val):
    """评估一组超参数配置"""
    config = {
        'hyper_hidden': int(params[0]),
        'target_hidden': int(params[1]),
        'lr': params[2],
        'weight_decay': params[3],
        'epochs': int(params[4]),
        'vae_epochs': 100,
        'num_interp': 5
    }
    
    try:
        # VAE增强
        X_aug, y_aug = vae_augment(X_train, y_train, config['vae_epochs'], config['num_interp'])
        
        # 训练HyperNetFusion
        model = train_hypernet(X_aug, y_aug, config)
        
        # 验证集评估
        y_pred = predict_hypernet(model, X_val)
        acc = accuracy_score(y_val, y_pred)
        
        return -acc  # 返回负值因为gp_minimize是最小化
    except:
        return 0.0  # 失败返回最差情况


def bayesian_optimize_hypernet(X_train, y_train, n_iter=15):
    """贝叶斯优化寻找最佳超参数"""
    # 分割出验证集（用于内部评估）
    n = len(X_train)
    indices = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx, val_idx = indices[:split], indices[split:]
    
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    
    # 搜索空间
    space = [
        Integer(32, 128, name='hyper_hidden'),
        Integer(16, 64, name='target_hidden'),
        Real(0.0001, 0.01, prior='log-uniform', name='lr'),
        Real(0.0001, 0.01, prior='log-uniform', name='weight_decay'),
        Integer(50, 200, name='epochs')
    ]
    
    # 贝叶斯优化
    result = gp_minimize(
        lambda p: evaluate_config(p, X_tr, y_tr, X_val, y_val),
        space,
        n_calls=n_iter,
        random_state=42,
        verbose=False
    )
    
    best_config = {
        'hyper_hidden': int(result.x[0]),
        'target_hidden': int(result.x[1]),
        'lr': result.x[2],
        'weight_decay': result.x[3],
        'epochs': int(result.x[4]),
        'vae_epochs': 100,
        'num_interp': 5
    }
    
    return best_config, -result.fun


def process_fold_hypernet(fold_info, X, y, bayes_n_iter):
    """处理单个fold - 使用HyperNetFusion"""
    fold_idx, (train_idx, test_idx) = fold_info
    
    try:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 贝叶斯优化找最佳超参数
        best_config, cv_score = bayesian_optimize_hypernet(X_train_scaled, y_train, bayes_n_iter)
        
        # 用最佳配置训练
        X_aug, y_aug = vae_augment(X_train_scaled, y_train, best_config['vae_epochs'], best_config['num_interp'])
        model = train_hypernet(X_aug, y_aug, best_config)
        
        # 预测
        y_pred = predict_hypernet(model, X_test_scaled)
        
        return {
            'fold_idx': fold_idx,
            'accuracy': accuracy_score(y_test, y_pred),
            'y_true': y_test.tolist(),
            'y_pred': y_pred.tolist(),
            'best_config': best_config,
            'cv_score': cv_score
        }
    except Exception as e:
        return {'fold_idx': fold_idx, 'accuracy': 0.0, 'error': str(e)}


def main():
    # 设置日志
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR / f'9_hypernet_bayes_{timestamp}.log'
    
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
    logger.info("9_hypernet_bayes.py - VAE-HyperNetFusion + 每fold贝叶斯优化")
    logger.info("这是你的原创方法！")
    logger.info(f"日志文件: {log_file}")
    logger.info("=" * 60)
    
    # 配置
    bayes_n_iter = 15  # 每个fold的贝叶斯迭代次数
    leave_p_out = 2
    
    logger.info(f"贝叶斯优化迭代次数/fold: {bayes_n_iter}")
    
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
    test_combos = list(combinations(all_indices, leave_p_out))
    n_folds = len(test_combos)
    
    logger.info(f"Leave-{leave_p_out}-Out: {n_folds} 个 folds")
    
    fold_infos = []
    for fold_idx, test_idx in enumerate(test_combos):
        test_idx = np.array(test_idx)
        train_idx = np.setdiff1d(all_indices, test_idx)
        fold_infos.append((fold_idx, (train_idx, test_idx)))
    
    # 确定进程数
    n_processes = min(mp.cpu_count(), n_folds, 32)
    logger.info(f"使用 {n_processes} 个进程并行处理")
    logger.info("⚠️  HyperNetFusion + 贝叶斯优化，预计需要较长时间...")
    
    start_time = time.time()
    
    # 并行处理
    process_func = partial(process_fold_hypernet, X=X, y=y, bayes_n_iter=bayes_n_iter)
    
    results = []
    with mp.Pool(processes=n_processes) as pool:
        for i, res in enumerate(pool.imap_unordered(process_func, fold_infos)):
            results.append(res)
            if (i + 1) % 20 == 0 or i + 1 == n_folds:
                accs = [r['accuracy'] for r in results if 'error' not in r]
                curr_acc = np.mean(accs) * 100 if accs else 0
                elapsed = time.time() - start_time
                eta = elapsed / (i + 1) * (n_folds - i - 1)
                logger.info(f"[{i+1}/{n_folds}] 当前准确率: {curr_acc:.2f}% | "
                           f"已用: {elapsed:.0f}s | 预计剩余: {eta:.0f}s")
    
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
    logger.info("[结果] VAE-HyperNetFusion + 每fold贝叶斯优化")
    logger.info("=" * 60)
    logger.info(f"  平均准确率: {mean_acc:.2f}% ± {std_acc:.2f}%")
    logger.info(f"  整体准确率: {overall_acc:.2f}%")
    logger.info(f"  成功folds: {len(valid_results)}/{n_folds}")
    logger.info(f"  失败folds: {len(error_results)}")
    logger.info(f"  总用时: {elapsed_time:.1f}秒")
    
    # 保存结果
    result_file = OUTPUT_DIR / f'9_hypernet_bayes_{timestamp}.json'
    
    result_data = {
        'experiment': '9_hypernet_bayes (VAE-HyperNetFusion)',
        'method': '你的原创方法: VAE数据增强 + HyperNetwork生成分类器权重',
        'timestamp': datetime.now().isoformat(),
        'mean_accuracy': mean_acc / 100,
        'std_accuracy': std_acc / 100,
        'overall_accuracy': overall_acc / 100,
        'elapsed_time': elapsed_time,
        'n_samples': n_samples,
        'n_folds': n_folds,
        'bayes_n_iter': bayes_n_iter,
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
