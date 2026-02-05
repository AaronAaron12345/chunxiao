#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
22_vae_hypernet_gpu.py - VAE + HyperNet (GPU并行版本)
=====================================================
核心思路：
1. VAE: 大量数据增强 (100倍+)
2. HyperNet: 模仿RF的结构
   - 多个子网络 (类似多棵树)
   - 每个子网络随机选择特征子集 (类似RF的特征bagging)
   - 投票集成

GPU并行: 多GPU + 批量处理多个fold

运行: python 22_vae_hypernet_gpu.py
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
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).parent
LOG_DIR = SCRIPT_DIR / 'logs'
OUTPUT_DIR = SCRIPT_DIR / 'output'
LOG_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# GPU配置
NUM_GPUS = 6
FOLDS_PER_GPU = 20  # 每个GPU同时处理的fold数


class VAE(nn.Module):
    """变分自编码器 - 用于数据增强"""
    def __init__(self, input_dim, hidden_dim=32, latent_dim=8):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
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


class SubNetwork(nn.Module):
    """子网络 - 类似RF中的一棵树"""
    def __init__(self, input_dim, hidden_dim=16, n_classes=2, dropout=0.3):
        super(SubNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )
    
    def forward(self, x):
        return self.net(x)


class HyperNetEnsemble(nn.Module):
    """
    HyperNet集成 - 模仿Random Forest结构
    
    核心思想：
    1. 多个子网络 (类似多棵决策树)
    2. 每个子网络使用特征子集 (类似RF的特征bagging)
    3. Bootstrap采样训练数据 (类似RF的样本bagging)
    4. 投票集成预测
    """
    def __init__(self, input_dim, n_classes=2, n_estimators=50, 
                 max_features='sqrt', hidden_dim=16, dropout=0.3):
        super(HyperNetEnsemble, self).__init__()
        
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.n_estimators = n_estimators
        
        # 计算每个子网络使用的特征数
        if max_features == 'sqrt':
            self.n_features = max(1, int(np.sqrt(input_dim)))
        elif max_features == 'log2':
            self.n_features = max(1, int(np.log2(input_dim)))
        else:
            self.n_features = input_dim
        
        # 为每个子网络随机选择特征索引
        self.feature_indices = []
        for _ in range(n_estimators):
            indices = np.random.choice(input_dim, self.n_features, replace=False)
            self.feature_indices.append(indices)
        
        # 创建子网络
        self.sub_networks = nn.ModuleList([
            SubNetwork(self.n_features, hidden_dim, n_classes, dropout)
            for _ in range(n_estimators)
        ])
    
    def forward(self, x):
        """前向传播 - 所有子网络的平均输出"""
        outputs = []
        for i, subnet in enumerate(self.sub_networks):
            # 选择该子网络对应的特征
            x_subset = x[:, self.feature_indices[i]]
            outputs.append(subnet(x_subset))
        
        # 平均所有子网络的输出 (软投票)
        return torch.stack(outputs).mean(dim=0)
    
    def predict_with_vote(self, x):
        """硬投票预测"""
        votes = []
        for i, subnet in enumerate(self.sub_networks):
            x_subset = x[:, self.feature_indices[i]]
            pred = subnet(x_subset).argmax(dim=1)
            votes.append(pred)
        
        # 投票
        votes = torch.stack(votes, dim=0)  # [n_estimators, batch_size]
        final_pred = []
        for j in range(x.shape[0]):
            vote_counts = torch.bincount(votes[:, j], minlength=self.n_classes)
            final_pred.append(vote_counts.argmax().item())
        return final_pred


def vae_augment_gpu(X_train, y_train, device, aug_factor=50):
    """
    GPU上的VAE数据增强
    
    增强策略：
    1. 原始-重建插值
    2. 潜在空间采样
    3. 高斯噪声增强
    4. Mixup增强
    """
    input_dim = X_train.shape[1]
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    X_aug_list = [X_scaled]
    y_aug_list = [y_train]
    
    # 对每个类别分别训练VAE
    for cls in np.unique(y_train):
        X_cls = X_scaled[y_train == cls]
        if len(X_cls) < 2:
            continue
        
        # 训练VAE
        vae = VAE(input_dim, hidden_dim=32, latent_dim=8).to(device)
        optimizer = optim.Adam(vae.parameters(), lr=0.005)
        X_tensor = torch.FloatTensor(X_cls).to(device)
        
        vae.train()
        for epoch in range(100):
            optimizer.zero_grad()
            recon, mu, log_var = vae(X_tensor)
            
            # 重建损失 + KL散度
            recon_loss = nn.MSELoss()(recon, X_tensor)
            kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
            loss = recon_loss + 0.01 * kl_loss
            
            loss.backward()
            optimizer.step()
        
        vae.eval()
        with torch.no_grad():
            # 1. 原始-重建插值 (主要增强方式)
            recon = vae(X_tensor)[0].cpu().numpy()
            for alpha in np.linspace(0.1, 0.9, aug_factor // 5):
                interpolated = alpha * X_cls + (1 - alpha) * recon
                X_aug_list.append(interpolated)
                y_aug_list.append(np.full(len(X_cls), cls))
            
            # 2. 潜在空间采样
            mu, log_var = vae.encode(X_tensor)
            for _ in range(aug_factor // 5):
                # 在潜在空间添加噪声
                z = vae.reparameterize(mu, log_var * 0.5)  # 减小方差
                new_samples = vae.decode(z).cpu().numpy()
                X_aug_list.append(new_samples)
                y_aug_list.append(np.full(len(X_cls), cls))
            
            # 3. 高斯噪声增强
            for noise_std in [0.05, 0.1, 0.15, 0.2]:
                noisy = X_cls + np.random.randn(*X_cls.shape) * noise_std
                X_aug_list.append(noisy)
                y_aug_list.append(np.full(len(X_cls), cls))
            
            # 4. Mixup (同类内部)
            if len(X_cls) >= 2:
                for _ in range(aug_factor // 10):
                    idx1, idx2 = np.random.choice(len(X_cls), 2, replace=False)
                    lam = np.random.beta(0.4, 0.4)
                    mixed = lam * X_cls[idx1] + (1 - lam) * X_cls[idx2]
                    X_aug_list.append(mixed.reshape(1, -1))
                    y_aug_list.append(np.array([cls]))
    
    return np.vstack(X_aug_list), np.hstack(y_aug_list), scaler


def train_hypernet(X_train, y_train, device, n_estimators=50, epochs=150):
    """训练HyperNet集成"""
    input_dim = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    
    model = HyperNetEnsemble(
        input_dim=input_dim,
        n_classes=n_classes,
        n_estimators=n_estimators,
        max_features='sqrt' if input_dim > 2 else None,
        hidden_dim=16,
        dropout=0.3
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.CrossEntropyLoss()
    
    X_tensor = torch.FloatTensor(X_train).to(device)
    y_tensor = torch.LongTensor(y_train).to(device)
    
    model.train()
    for epoch in range(epochs):
        # Bootstrap采样 (类似RF)
        bootstrap_idx = np.random.choice(len(X_train), len(X_train), replace=True)
        X_batch = X_tensor[bootstrap_idx]
        y_batch = y_tensor[bootstrap_idx]
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    return model


def process_fold_batch_gpu(args):
    """GPU上批量处理多个fold"""
    gpu_id, fold_batch, X_all, y_all = args
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    results = []
    for fold_idx, train_idx, test_idx in fold_batch:
        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_test, y_test = X_all[test_idx], y_all[test_idx]
        
        # VAE数据增强
        X_aug, y_aug, scaler = vae_augment_gpu(X_train, y_train, device, aug_factor=50)
        X_test_scaled = scaler.transform(X_test)
        
        # 训练HyperNet
        model = train_hypernet(X_aug, y_aug, device, n_estimators=50, epochs=100)
        
        # 预测
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
            y_pred = model.predict_with_vote(X_test_tensor)
        
        results.append({
            'fold_idx': fold_idx,
            'y_true': y_test.tolist(),
            'y_pred': y_pred
        })
    
    return results


def process_fold_rf(args):
    """RF基线"""
    fold_idx, X_train, y_train, X_test, y_test = args
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    rf.fit(X_train_s, y_train)
    y_pred = rf.predict(X_test_s)
    
    return {'y_true': y_test.tolist(), 'y_pred': y_pred.tolist()}


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR / f'22_vae_hypernet_gpu_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 70)
    logger.info("22_vae_hypernet_gpu.py - VAE + HyperNet (GPU并行)")
    logger.info("=" * 70)
    
    # 检测GPU
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        logger.info(f"检测到 {n_gpus} 个GPU")
        for i in range(n_gpus):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.info("未检测到GPU，使用CPU")
        n_gpus = 1
    
    # 加载数据
    data_path = SCRIPT_DIR / 'data' / 'Data_for_Jinming.csv'
    df = pd.read_csv(data_path)
    X = df[['LAA', 'Glutamate', 'Choline', 'Sarcosine']].values.astype(np.float32)
    y = LabelEncoder().fit_transform(df['Group'].values)
    
    n_samples = len(X)
    logger.info(f"数据: {n_samples} 样本, {X.shape[1]} 特征")
    
    # 生成所有fold
    test_combos = list(combinations(range(n_samples), 2))
    n_folds = len(test_combos)
    logger.info(f"Leave-2-Out: {n_folds} folds")
    
    # 准备fold数据
    fold_list = []
    for fold_idx, test_idx in enumerate(test_combos):
        test_idx = np.array(test_idx)
        train_idx = np.setdiff1d(np.arange(n_samples), test_idx)
        fold_list.append((fold_idx, train_idx, test_idx))
    
    results = {}
    
    # ========== 1. RF 基线 ==========
    logger.info("\n[1/2] Random Forest 基线...")
    start = time.time()
    
    rf_fold_args = [
        (i, X[train_idx], y[train_idx], X[test_idx], y[test_idx])
        for i, (fold_idx, train_idx, test_idx) in enumerate(fold_list)
    ]
    
    with ProcessPoolExecutor(max_workers=64) as executor:
        rf_results = list(executor.map(process_fold_rf, rf_fold_args))
    
    y_true_all = [item for r in rf_results for item in r['y_true']]
    y_pred_all = [item for r in rf_results for item in r['y_pred']]
    results['RF'] = accuracy_score(y_true_all, y_pred_all) * 100
    logger.info(f"   RF: {results['RF']:.2f}% ({time.time()-start:.1f}s)")
    
    # ========== 2. VAE + HyperNet (GPU) ==========
    logger.info("\n[2/2] VAE + HyperNet (GPU并行)...")
    start = time.time()
    
    # 分配fold到不同GPU
    use_gpus = min(n_gpus, NUM_GPUS) if torch.cuda.is_available() else 1
    folds_per_batch = FOLDS_PER_GPU
    
    # 将fold分成批次，分配到不同GPU
    gpu_tasks = []
    for gpu_id in range(use_gpus):
        gpu_folds = fold_list[gpu_id::use_gpus]  # 每个GPU处理间隔的fold
        # 再分成小批次
        for i in range(0, len(gpu_folds), folds_per_batch):
            batch = gpu_folds[i:i+folds_per_batch]
            gpu_tasks.append((gpu_id, batch, X, y))
    
    logger.info(f"   使用 {use_gpus} 个GPU, {len(gpu_tasks)} 个批次")
    
    all_hypernet_results = []
    processed = 0
    total_batches = len(gpu_tasks)
    
    # 并行处理
    with ProcessPoolExecutor(max_workers=use_gpus) as executor:
        futures = {executor.submit(process_fold_batch_gpu, task): task for task in gpu_tasks}
        
        for future in as_completed(futures):
            batch_results = future.result()
            all_hypernet_results.extend(batch_results)
            processed += 1
            if processed % 5 == 0 or processed == total_batches:
                logger.info(f"   进度: {processed}/{total_batches} 批次 ({100*processed/total_batches:.1f}%)")
    
    # 计算准确率
    y_true_all = [item for r in all_hypernet_results for item in r['y_true']]
    y_pred_all = [item for r in all_hypernet_results for item in r['y_pred']]
    results['VAE+HyperNet'] = accuracy_score(y_true_all, y_pred_all) * 100
    logger.info(f"   VAE+HyperNet: {results['VAE+HyperNet']:.2f}% ({time.time()-start:.1f}s)")
    
    # ========== 结果 ==========
    logger.info("\n" + "=" * 70)
    logger.info("[最终结果]")
    logger.info("=" * 70)
    for name, acc in results.items():
        marker = "🏆" if acc == max(results.values()) else "  "
        logger.info(f"{marker} {name:20s}: {acc:.2f}%")
    logger.info("=" * 70)
    
    # 保存结果
    result_file = OUTPUT_DIR / f'22_vae_hypernet_gpu_{timestamp}.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"保存: {result_file}")


if __name__ == '__main__':
    main()
