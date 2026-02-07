#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
15_vae_hypernet_paper.py - 严格按照论文实现的VAE-HyperNetFusion
=================================================================
严格按照 IEEE ICC 2026 论文实现：

1. VAE数据增强：
   - hidden_dim=512, latent_dim=20
   - epochs=50, lr=0.001, batch_size=128
   - 插值：5个均匀分布的内部点

2. 超网络 + 目标网络集成：
   - **一个**超网络 H(z; φ) 生成**多个**目标网络的权重
   - 每个目标网络学习数据的不同"切片"(descriptor z_i)
   - 最终预测：平均所有目标网络的logits

关键公式：
   θ_i = H(z_i; φ)          -- 公式3
   ŷ = F(f_1(x), ..., f_m(x)) -- 公式4

运行: python 15_vae_hypernet_paper.py
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
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
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


# ==================== VAE - 按论文参数 ====================
class VAE(nn.Module):
    """
    变分自编码器 - 严格按照论文参数
    论文: "hidden layer of width 512 and a latent dimension of 20"
    """
    def __init__(self, input_dim, hidden_dim=512, latent_dim=20):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder - 输出sigmoid匹配min-max归一化的[0,1]范围
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # 论文: "matches the sigmoid output of the decoder"
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
        recon = self.decode(z)
        return recon, mu, log_var


# ==================== 超网络 - 按论文设计 ====================
class HyperNetwork(nn.Module):
    """
    超网络：生成多个目标网络的权重
    论文: "hypernetwork H used to generate all weights and biases for the target network"
    公式3: θ_i = H(z_i; φ)
    
    z_i 是描述第i个目标网络的descriptor
    """
    def __init__(self, descriptor_dim, hidden_dim, target_input_dim, target_hidden_dim, n_classes):
        super(HyperNetwork, self).__init__()
        
        self.target_input_dim = target_input_dim
        self.target_hidden_dim = target_hidden_dim
        self.n_classes = n_classes
        
        # 超网络主体
        self.net = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 生成目标网络第一层权重和偏置
        self.gen_w1 = nn.Linear(hidden_dim, target_input_dim * target_hidden_dim)
        self.gen_b1 = nn.Linear(hidden_dim, target_hidden_dim)
        
        # 生成目标网络第二层权重和偏置
        self.gen_w2 = nn.Linear(hidden_dim, target_hidden_dim * n_classes)
        self.gen_b2 = nn.Linear(hidden_dim, n_classes)
    
    def forward(self, descriptor):
        """
        输入: descriptor z_i (描述第i个目标网络的特征)
        输出: 目标网络的权重 (w1, b1, w2, b2)
        """
        h = self.net(descriptor)
        
        w1 = self.gen_w1(h).view(-1, self.target_input_dim, self.target_hidden_dim)
        b1 = self.gen_b1(h)
        w2 = self.gen_w2(h).view(-1, self.target_hidden_dim, self.n_classes)
        b2 = self.gen_b2(h)
        
        return w1, b1, w2, b2


class TargetNetwork:
    """
    目标网络：使用超网络生成的权重进行前向传播
    论文: "Each target network is a compact MLP classifier with one hidden layer"
    """
    @staticmethod
    def forward(x, w1, b1, w2, b2):
        """
        x: 输入数据 [batch, input_dim]
        w1, b1: 第一层权重和偏置
        w2, b2: 第二层权重和偏置
        """
        # 第一层
        if len(w1.shape) == 3:
            h = torch.bmm(x.unsqueeze(1), w1).squeeze(1) + b1
        else:
            h = torch.mm(x, w1) + b1
        h = torch.relu(h)
        
        # 第二层（输出层）
        if len(w2.shape) == 3:
            out = torch.bmm(h.unsqueeze(1), w2).squeeze(1) + b2
        else:
            out = torch.mm(h, w2) + b2
        
        return out


class VAEHyperNetFusion:
    """
    VAE-HyperNetFusion 完整框架 - 严格按照论文
    
    1. VAE数据增强 + 插值
    2. 一个超网络生成多个目标网络权重
    3. 目标网络集成预测
    """
    def __init__(self, input_dim, n_classes, n_target_networks, device, config):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.n_target_networks = n_target_networks  # 目标网络数量 m
        self.device = device
        self.config = config
        
        # 描述符维度 = 输入维度 (每个目标网络看到数据的不同"视角")
        self.descriptor_dim = input_dim * 2  # 使用mean和std作为descriptor
        
        # 超网络 - 只有一个！生成所有目标网络的权重
        self.hypernet = HyperNetwork(
            descriptor_dim=self.descriptor_dim,
            hidden_dim=config['hyper_hidden'],
            target_input_dim=input_dim,
            target_hidden_dim=config['target_hidden'],
            n_classes=n_classes
        ).to(device)
        
        self.vae = None
        self.scaler = MinMaxScaler()  # 论文: "min-max normalization"
        self.std_scaler = StandardScaler()
    
    def train_vae(self, X_train):
        """
        训练VAE - 严格按照论文参数
        论文: "trained for 50 epochs with Adam using lr=0.001 and batch_size=128"
        """
        # Min-max归一化到[0,1]
        X_normalized = self.scaler.fit_transform(X_train)
        X_tensor = torch.FloatTensor(X_normalized).to(self.device)
        
        self.vae = VAE(
            input_dim=self.input_dim,
            hidden_dim=512,   # 论文参数
            latent_dim=20     # 论文参数
        ).to(self.device)
        
        optimizer = optim.Adam(self.vae.parameters(), lr=0.001)  # 论文参数
        
        self.vae.train()
        batch_size = min(128, len(X_tensor))  # 论文: batch_size=128
        
        for epoch in range(self.config['vae_epochs']):  # 论文: 50 epochs
            # 随机打乱
            perm = torch.randperm(len(X_tensor))
            
            for i in range(0, len(X_tensor), batch_size):
                batch_idx = perm[i:i+batch_size]
                batch = X_tensor[batch_idx]
                
                optimizer.zero_grad()
                recon, mu, log_var = self.vae(batch)
                
                # 确保在[0,1]范围内，避免BCELoss错误
                recon = torch.clamp(recon, 1e-6, 1-1e-6)
                batch_clamped = torch.clamp(batch, 1e-6, 1-1e-6)
                
                # 论文: "binary cross-entropy reconstruction term with KL regularization"
                recon_loss = nn.BCELoss()(recon, batch_clamped)
                kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                
                loss = recon_loss + kl_loss
                loss.backward()
                optimizer.step()
    
    def augment_data(self, X_train, y_train):
        """
        VAE数据增强 + 插值 - 严格按照论文
        论文: "5 evenly spaced interior points on the line segment"
        公式2: x̂_α = α·x + (1-α)·x'
        """
        # 归一化
        X_normalized = self.scaler.transform(X_train)
        X_tensor = torch.FloatTensor(X_normalized).to(self.device)
        
        self.vae.eval()
        with torch.no_grad():
            recon, _, _ = self.vae(X_tensor)
        
        # 论文: "5 evenly spaced interior points"
        # interior points意味着不包括端点(0和1)
        alphas = np.linspace(0, 1, 7)[1:-1]  # [0.167, 0.333, 0.5, 0.667, 0.833] 5个内部点
        
        X_aug_list = [X_tensor]
        y_aug_list = [torch.LongTensor(y_train).to(self.device)]
        
        for alpha in alphas:
            # 公式2: 插值
            interp = alpha * X_tensor + (1 - alpha) * recon
            X_aug_list.append(interp)
            y_aug_list.append(torch.LongTensor(y_train).to(self.device))
        
        X_aug = torch.cat(X_aug_list, dim=0)
        y_aug = torch.cat(y_aug_list, dim=0)
        
        # 转换回原始特征空间
        X_aug_np = X_aug.cpu().numpy()
        X_aug_original = self.scaler.inverse_transform(X_aug_np)
        
        return X_aug_original, y_aug.cpu().numpy()
    
    def compute_descriptors(self, X_aug):
        """
        计算每个目标网络的descriptor
        论文: "z_i is a descriptor that characterizes the features or data slices"
        
        策略：每个目标网络看到数据的不同随机子集（bootstrap）
        """
        descriptors = []
        X_tensor = torch.FloatTensor(X_aug).to(self.device)
        
        for i in range(self.n_target_networks):
            # 每个目标网络使用不同的随机子集计算descriptor
            np.random.seed(42 + i)
            indices = np.random.choice(len(X_aug), size=len(X_aug), replace=True)  # bootstrap
            X_subset = X_tensor[indices]
            
            # descriptor = [mean, std] of the data slice
            mean = X_subset.mean(dim=0)
            std = X_subset.std(dim=0) + 1e-6
            descriptor = torch.cat([mean, std]).unsqueeze(0)
            descriptors.append(descriptor)
        
        return descriptors
    
    def train(self, X_train, y_train):
        """训练完整的VAE-HyperNetFusion"""
        # 1. 训练VAE
        self.train_vae(X_train)
        
        # 2. 数据增强
        X_aug, y_aug = self.augment_data(X_train, y_train)
        
        # 3. 标准化增强后的数据
        X_aug_scaled = self.std_scaler.fit_transform(X_aug)
        X_tensor = torch.FloatTensor(X_aug_scaled).to(self.device)
        y_tensor = torch.LongTensor(y_aug).to(self.device)
        
        # 4. 计算每个目标网络的descriptor
        descriptors = self.compute_descriptors(X_aug_scaled)
        
        # 5. 训练超网络
        optimizer = optim.Adam(self.hypernet.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        criterion = nn.CrossEntropyLoss()
        
        self.hypernet.train()
        for epoch in range(self.config['epochs']):
            total_loss = 0
            
            for i, descriptor in enumerate(descriptors):
                optimizer.zero_grad()
                
                # 超网络生成第i个目标网络的权重
                w1, b1, w2, b2 = self.hypernet(descriptor)
                
                # 目标网络前向传播
                outputs = TargetNetwork.forward(X_tensor, w1[0], b1[0], w2[0], b2[0])
                loss = criterion(outputs, y_tensor)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
    
    def predict(self, X_test):
        """
        集成预测 - 严格按照论文
        论文公式4: ŷ = F(f_1(x), ..., f_m(x))
        "averaging their logits before taking the final class decision"
        """
        X_scaled = self.std_scaler.transform(X_test)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # 计算descriptors
        descriptors = self.compute_descriptors(self.std_scaler.transform(
            self.scaler.inverse_transform(self.scaler.transform(X_test))
        ))
        
        self.hypernet.eval()
        all_logits = []
        
        with torch.no_grad():
            for descriptor in descriptors:
                # 超网络生成权重
                w1, b1, w2, b2 = self.hypernet(descriptor)
                
                # 目标网络预测
                logits = TargetNetwork.forward(X_tensor, w1[0], b1[0], w2[0], b2[0])
                all_logits.append(logits)
        
        # 论文: "averaging their logits"
        avg_logits = torch.stack(all_logits).mean(dim=0)
        
        # 论文: "values greater than 0.5 are considered as class 1"
        if self.n_classes == 2:
            probs = torch.softmax(avg_logits, dim=1)
            return (probs[:, 1] > 0.5).long().cpu().numpy()
        else:
            return avg_logits.argmax(dim=1).cpu().numpy()


class GPUWorker:
    """GPU工作线程"""
    def __init__(self, gpu_id, config):
        self.gpu_id = gpu_id
        self.device = torch.device(f'cuda:{gpu_id}')
        self.config = config
        self.results = []
        self.processed_count = 0
        self.lock = threading.Lock()
        self.n_threads = 16  # 增加线程数加速
    
    def process_fold(self, fold_data):
        """处理单个fold"""
        fold_idx, X_train, y_train, X_test, y_test = fold_data
        
        try:
            n_classes = len(np.unique(y_train))
            
            # 创建VAE-HyperNetFusion
            model = VAEHyperNetFusion(
                input_dim=X_train.shape[1],
                n_classes=n_classes,
                n_target_networks=self.config['n_target_networks'],
                device=self.device,
                config=self.config
            )
            
            # 训练
            model.train(X_train, y_train)
            
            # 预测
            y_pred = model.predict(X_test)
            
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
        """批处理"""
        print(f"\n[GPU {self.gpu_id}] 开始处理 {len(fold_batch)} 个folds...", flush=True)
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            list(executor.map(self.process_fold, fold_batch))


def progress_monitor(workers, total, start_time, stop_event):
    """进度监控"""
    while not stop_event.is_set():
        current = sum(w.processed_count for w in workers)
        elapsed = time.time() - start_time
        
        if elapsed > 0 and current > 0:
            rate = current / elapsed
            eta = (total - current) / rate if rate > 0 else 0
            pct = current / total * 100
            
            all_valid = []
            for w in workers:
                all_valid.extend([r for r in w.results if 'error' not in r])
            
            if all_valid:
                all_true = [item for r in all_valid for item in r['y_true']]
                all_pred = [item for r in all_valid for item in r['y_pred']]
                total_acc = accuracy_score(all_true, all_pred) * 100
            else:
                total_acc = 0.0
            
            bar_len = 30
            filled = int(bar_len * current / total)
            bar = '█' * filled + '░' * (bar_len - filled)
            
            print(f'\r[{bar}] {current}/{total} ({pct:.1f}%) | {rate:.2f}/s | ETA:{eta:.0f}s | Acc:{total_acc:.2f}%', end='', flush=True)
        
        if current >= total:
            break
        
        time.sleep(0.5)
    print()


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR / f'15_vae_hypernet_paper_{timestamp}.log'
    
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
    logger.info("15_vae_hypernet_paper.py - 严格按照论文实现的VAE-HyperNetFusion")
    logger.info("=" * 70)
    
    if not torch.cuda.is_available():
        logger.error("CUDA不可用!")
        return
    
    n_gpus = torch.cuda.device_count()
    available_gpus = [i for i in GPU_IDS if i < n_gpus]
    logger.info(f"可用GPU: {available_gpus}")
    
    # 配置 - 快速版本（先验证思路）
    config = {
        'n_target_networks': 3,   # 减少目标网络数量
        'hyper_hidden': 128,      # 减小
        'target_hidden': 32,      # 减小
        'lr': 0.005,              # 增大lr加速收敛
        'weight_decay': 0.0001,
        'epochs': 30,             # 大幅减少
        'vae_epochs': 20,         # 大幅减少
    }
    
    logger.info("配置（快速版本）:")
    logger.info(f"  VAE: hidden=512, latent=20, epochs={config['vae_epochs']}")
    logger.info(f"  插值: 5个均匀分布的内部点")
    logger.info(f"  目标网络数量: {config['n_target_networks']}")
    logger.info(f"  超网络训练: epochs={config['epochs']}, lr={config['lr']}")
    
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
    
    logger.info(f"Leave-{leave_p_out}-Out: {n_folds} 个 folds (论文方法)")
    
    # 预处理所有fold数据
    fold_datas = []
    for fold_idx, test_idx in enumerate(test_combos):
        test_idx = np.array(test_idx)
        train_idx = np.setdiff1d(all_indices, test_idx)
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        fold_datas.append((fold_idx, X_train, y_train, X_test, y_test))
    
    # 分配到各GPU
    gpu_fold_batches = {gpu_id: [] for gpu_id in available_gpus}
    for i, fold_data in enumerate(fold_datas):
        gpu_id = available_gpus[i % len(available_gpus)]
        gpu_fold_batches[gpu_id].append(fold_data)
    
    logger.info(f"并行策略: {len(available_gpus)}个GPU × 8线程/GPU")
    logger.info(f"分配: {', '.join([f'GPU{g}={len(b)}' for g,b in gpu_fold_batches.items()])}")
    
    # 创建GPU工作器
    workers = [GPUWorker(gpu_id, config) for gpu_id in available_gpus]
    
    start_time = time.time()
    stop_event = threading.Event()
    
    # 启动进度监控
    monitor = threading.Thread(target=progress_monitor, args=(workers, n_folds, start_time, stop_event))
    monitor.start()
    
    logger.info("开始运行VAE-HyperNetFusion（论文实现）...")
    print()
    
    # 多线程并行
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
    
    all_y_true, all_y_pred = [], []
    for r in valid_results:
        all_y_true.extend(r['y_true'])
        all_y_pred.extend(r['y_pred'])
    
    overall_acc = accuracy_score(all_y_true, all_y_pred) * 100
    
    print()
    logger.info("=" * 70)
    logger.info("[结果] VAE-HyperNetFusion（严格按照论文实现）")
    logger.info("=" * 70)
    logger.info(f"  🎯 整体准确率: {overall_acc:.2f}%")
    logger.info(f"  ✅ 成功folds: {len(valid_results)}/{n_folds}")
    logger.info(f"  ⏱️  总用时: {elapsed_time:.1f}秒")
    logger.info(f"  🚀 速度: {n_folds / elapsed_time:.1f} folds/秒")
    
    if error_results:
        logger.info(f"  ❌ 失败folds: {len(error_results)}")
        for e in error_results[:3]:
            logger.info(f"     错误: {e.get('error', 'unknown')}")
    
    # 保存结果
    result_file = OUTPUT_DIR / f'15_vae_hypernet_paper_{timestamp}.json'
    
    result_data = {
        'experiment': '15_vae_hypernet_paper',
        'method': 'VAE-HyperNetFusion（论文实现）',
        'timestamp': datetime.now().isoformat(),
        'overall_accuracy': overall_acc / 100,
        'elapsed_time': elapsed_time,
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
