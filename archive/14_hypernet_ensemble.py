#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
14_hypernet_ensemble.py - 真正的HyperNetFusion集成版（仿RF多树）
================================================================
核心改进：
1. **多个HyperNetwork集成** - 像RF有多棵树，我们有多个超网络
2. **多样化增强** - 每个超网络用不同的VAE增强数据训练
3. **投票机制** - 多个TargetNetwork投票预测
4. **大胆并行** - 每个GPU并行27个fold

这才是真正仿照RF的HyperNetFusion设计！

运行: python 14_hypernet_ensemble.py
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
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    """变分自编码器 - 用于数据增强"""
    def __init__(self, input_dim, hidden_dim=64, latent_dim=4):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LeakyReLU(0.2)
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_var = nn.Linear(hidden_dim // 2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim), nn.LeakyReLU(0.2),
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
    
    def sample(self, n_samples, device):
        """从潜在空间采样生成新数据"""
        z = torch.randn(n_samples, self.fc_mu.out_features).to(device)
        return self.decode(z)


class HyperNetwork(nn.Module):
    """超网络：从训练数据统计量生成目标网络权重"""
    def __init__(self, input_dim, stat_dim, hidden_dim, target_hidden, n_classes, dropout=0.3):
        super(HyperNetwork, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(stat_dim, hidden_dim), nn.LeakyReLU(0.2), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LeakyReLU(0.2)
        )
        
        # 生成两层全连接网络的权重
        self.gen_w1 = nn.Linear(hidden_dim // 2, input_dim * target_hidden)
        self.gen_b1 = nn.Linear(hidden_dim // 2, target_hidden)
        self.gen_w2 = nn.Linear(hidden_dim // 2, target_hidden * n_classes)
        self.gen_b2 = nn.Linear(hidden_dim // 2, n_classes)
        
        self.input_dim = input_dim
        self.target_hidden = target_hidden
        self.n_classes = n_classes
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, stats):
        h = self.net(stats)
        w1 = self.gen_w1(h).view(-1, self.input_dim, self.target_hidden)
        b1 = self.gen_b1(h).view(-1, self.target_hidden)
        w2 = self.gen_w2(h).view(-1, self.target_hidden, self.n_classes)
        b2 = self.gen_b2(h).view(-1, self.n_classes)
        return w1, b1, w2, b2


def target_forward(x, w1, b1, w2, b2):
    """目标网络前向传播"""
    if w1.dim() == 3:
        h = torch.bmm(x.unsqueeze(1), w1).squeeze(1) + b1
    else:
        h = torch.mm(x, w1) + b1
    h = torch.relu(h)
    if w2.dim() == 3:
        out = torch.bmm(h.unsqueeze(1), w2).squeeze(1) + b2
    else:
        out = torch.mm(h, w2) + b2
    return out


class HyperNetEnsemble:
    """
    HyperNetFusion集成 - 仿照随机森林的多树设计
    每个超网络相当于RF中的一棵树
    """
    def __init__(self, n_estimators, input_dim, n_classes, device, config):
        self.n_estimators = n_estimators
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.device = device
        self.config = config
        
        # 统计量维度
        self.stat_dim = input_dim * 2 + input_dim * input_dim
        
        # 创建多个超网络（相当于多棵树）
        self.hypernets = []
        for i in range(n_estimators):
            # 每个超网络略有不同的配置，增加多样性
            hidden = config['hyper_hidden'] + (i % 3) * 16
            target_h = config['target_hidden'] + (i % 2) * 8
            
            hypernet = HyperNetwork(
                input_dim=input_dim,
                stat_dim=self.stat_dim,
                hidden_dim=hidden,
                target_hidden=target_h,
                n_classes=n_classes,
                dropout=config['dropout'] + (i % 5) * 0.02
            ).to(device)
            self.hypernets.append(hypernet)
        
        # VAE用于数据增强
        self.vaes = {}
    
    def compute_stats(self, X_tensor):
        """计算数据统计量作为超网络输入"""
        mean = X_tensor.mean(dim=0)
        std = X_tensor.std(dim=0) + 1e-6
        X_centered = X_tensor - mean
        cov = torch.mm(X_centered.T, X_centered) / (len(X_tensor) - 1 + 1e-6)
        cov_flat = cov.flatten()
        stats = torch.cat([mean, std, cov_flat])
        return stats.unsqueeze(0)
    
    def train_vae_for_class(self, X_cls, cls_label):
        """为单个类别训练VAE"""
        if len(X_cls) < 2:
            return None
        
        vae = VAE(self.input_dim, hidden_dim=64, latent_dim=4).to(self.device)
        optimizer = torch.optim.Adam(vae.parameters(), lr=0.002, weight_decay=1e-5)
        X_tensor = torch.FloatTensor(X_cls).to(self.device)
        
        vae.train()
        for epoch in range(self.config['vae_epochs']):
            optimizer.zero_grad()
            recon, mu, log_var = vae(X_tensor)
            
            # 重建损失
            recon_loss = nn.MSELoss()(recon, X_tensor)
            # KL散度
            kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
            # 总损失
            beta = min(1.0, epoch / 50) * 0.1  # 渐进KL权重
            loss = recon_loss + beta * kl_loss
            
            loss.backward()
            optimizer.step()
        
        return vae
    
    def generate_augmented_data(self, X, y, estimator_idx):
        """
        为特定的estimator生成增强数据
        每个estimator用不同的增强策略，增加多样性（类似RF的bootstrap）
        """
        X_aug_list = [torch.FloatTensor(X).to(self.device)]
        y_aug_list = [torch.LongTensor(y).to(self.device)]
        
        # 随机种子根据estimator_idx变化，确保每个超网络看到不同的增强数据
        np.random.seed(42 + estimator_idx)
        
        for cls in np.unique(y):
            X_cls = X[y == cls]
            if len(X_cls) < 2:
                continue
            
            # 为该类训练VAE（如果没有的话）
            vae_key = (cls, estimator_idx % 3)  # 共享部分VAE减少计算
            if vae_key not in self.vaes:
                self.vaes[vae_key] = self.train_vae_for_class(X_cls, cls)
            
            vae = self.vaes[vae_key]
            if vae is None:
                continue
            
            vae.eval()
            X_tensor = torch.FloatTensor(X_cls).to(self.device)
            
            with torch.no_grad():
                # 1. 重建插值
                recon = vae(X_tensor)[0]
                # 不同estimator用不同的插值比例
                alphas = np.linspace(0.2 + estimator_idx * 0.05, 0.8, self.config['num_interp'])
                for alpha in alphas:
                    aug_data = alpha * X_tensor + (1 - alpha) * recon
                    X_aug_list.append(aug_data)
                    y_aug_list.append(torch.full((len(X_cls),), cls, dtype=torch.long, device=self.device))
                
                # 2. 潜在空间采样
                n_new = max(2, len(X_cls) // 2)
                mu, log_var = vae.encode(X_tensor)
                for _ in range(2):
                    z = vae.reparameterize(mu, log_var)
                    # 添加噪声
                    z = z + torch.randn_like(z) * 0.1 * (estimator_idx % 3 + 1)
                    new_samples = vae.decode(z)
                    X_aug_list.append(new_samples)
                    y_aug_list.append(torch.full((len(X_cls),), cls, dtype=torch.long, device=self.device))
        
        return torch.cat(X_aug_list), torch.cat(y_aug_list)
    
    def train(self, X_train, y_train):
        """训练所有超网络"""
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        
        # 训练每个超网络
        for idx, hypernet in enumerate(self.hypernets):
            # 每个超网络用不同的增强数据（类似RF的bootstrap）
            X_aug, y_aug = self.generate_augmented_data(X_train, y_train, idx)
            
            optimizer = optim.Adam(hypernet.parameters(), 
                                   lr=self.config['lr'] * (0.8 + 0.4 * np.random.random()),
                                   weight_decay=self.config['weight_decay'])
            criterion = nn.CrossEntropyLoss()
            
            # 计算统计量
            stats = self.compute_stats(X_aug)
            
            # 训练
            hypernet.train()
            for epoch in range(self.config['epochs']):
                optimizer.zero_grad()
                w1, b1, w2, b2 = hypernet(stats)
                outputs = target_forward(X_aug, w1[0], b1[0], w2[0], b2[0])
                loss = criterion(outputs, y_aug)
                loss.backward()
                optimizer.step()
    
    def predict(self, X_test):
        """集成预测 - 投票机制"""
        X_tensor = torch.FloatTensor(X_test).to(self.device)
        
        all_probs = []
        for hypernet in self.hypernets:
            hypernet.eval()
            with torch.no_grad():
                # 使用测试数据的统计量
                stats = self.compute_stats(X_tensor)
                w1, b1, w2, b2 = hypernet(stats)
                outputs = target_forward(X_tensor, w1[0], b1[0], w2[0], b2[0])
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs)
        
        # 平均概率投票
        avg_probs = torch.stack(all_probs).mean(dim=0)
        return avg_probs.argmax(dim=1).cpu().numpy()


class GPUWorker:
    """GPU工作线程 - 优化并行度"""
    def __init__(self, gpu_id, config):
        self.gpu_id = gpu_id
        self.device = torch.device(f'cuda:{gpu_id}')
        self.config = config
        self.results = []
        self.processed_count = 0
        self.lock = threading.Lock()
        # 每个GPU 8个线程（太多会内存争抢）
        self.n_threads = 8
    
    def process_fold(self, fold_data):
        """处理单个fold"""
        fold_idx, X_train, y_train, X_test, y_test = fold_data
        
        try:
            input_dim = X_train.shape[1]
            n_classes = len(np.unique(y_train))
            
            # 创建HyperNet集成
            ensemble = HyperNetEnsemble(
                n_estimators=self.config['n_estimators'],
                input_dim=input_dim,
                n_classes=n_classes,
                device=self.device,
                config=self.config
            )
            
            # 训练
            ensemble.train(X_train, y_train)
            
            # 预测
            y_pred = ensemble.predict(X_test)
            
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
        """批处理 - 8线程并行"""
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
    log_file = LOG_DIR / f'14_hypernet_ensemble_{timestamp}.log'
    
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
    logger.info("14_hypernet_ensemble.py - 真正的HyperNetFusion集成版（仿RF多树）")
    logger.info("=" * 70)
    
    if not torch.cuda.is_available():
        logger.error("CUDA不可用!")
        return
    
    n_gpus = torch.cuda.device_count()
    available_gpus = [i for i in GPU_IDS if i < n_gpus]
    logger.info(f"可用GPU: {available_gpus}")
    
    for gpu_id in available_gpus:
        logger.info(f"  GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    
    # 配置 - 优化版（减少计算量但保持集成效果）
    config = {
        'n_estimators': 5,       # 5个超网络集成（足够了）
        'hyper_hidden': 96,
        'target_hidden': 32,
        'lr': 0.005,
        'weight_decay': 0.001,
        'dropout': 0.2,
        'epochs': 100,           # 减少epochs
        'vae_epochs': 50,        # 减少VAE epochs
        'num_interp': 3
    }
    
    logger.info(f"配置: {config}")
    logger.info(f"核心改进: {config['n_estimators']}个超网络集成（仿RF多树）")
    
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
    
    # 预处理所有fold数据
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
    
    # 分配到各GPU
    gpu_fold_batches = {gpu_id: [] for gpu_id in available_gpus}
    for i, fold_data in enumerate(fold_datas):
        gpu_id = available_gpus[i % len(available_gpus)]
        gpu_fold_batches[gpu_id].append(fold_data)
    
    logger.info(f"并行策略: 6个GPU × 8线程/GPU = 48并行")
    logger.info(f"分配: {', '.join([f'GPU{g}={len(b)}' for g,b in gpu_fold_batches.items()])}")
    
    # 创建GPU工作器
    workers = [GPUWorker(gpu_id, config) for gpu_id in available_gpus]
    
    start_time = time.time()
    stop_event = threading.Event()
    
    # 启动进度监控
    monitor = threading.Thread(target=progress_monitor, args=(workers, n_folds, start_time, stop_event))
    monitor.start()
    
    logger.info("开始运行HyperNetFusion集成（多超网络+VAE增强）...")
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
    accuracies = [r['accuracy'] for r in valid_results]
    
    mean_acc = np.mean(accuracies) * 100
    std_acc = np.std(accuracies) * 100
    
    all_y_true, all_y_pred = [], []
    for r in valid_results:
        all_y_true.extend(r['y_true'])
        all_y_pred.extend(r['y_pred'])
    
    overall_acc = accuracy_score(all_y_true, all_y_pred) * 100
    
    print()
    logger.info("=" * 70)
    logger.info("[结果] VAE-HyperNetFusion 集成版（仿RF多树）")
    logger.info("=" * 70)
    logger.info(f"  🎯 整体准确率: {overall_acc:.2f}%")
    logger.info(f"  📊 平均准确率: {mean_acc:.2f}% ± {std_acc:.2f}%")
    logger.info(f"  🌲 超网络数量: {config['n_estimators']} 个（集成）")
    logger.info(f"  ✅ 成功folds: {len(valid_results)}/{n_folds}")
    logger.info(f"  ⏱️  总用时: {elapsed_time:.1f}秒")
    logger.info(f"  🚀 速度: {n_folds / elapsed_time:.1f} folds/秒")
    
    if error_results:
        logger.info(f"  ❌ 失败folds: {len(error_results)}")
    
    # 保存结果
    result_file = OUTPUT_DIR / f'14_hypernet_ensemble_{timestamp}.json'
    
    result_data = {
        'experiment': '14_hypernet_ensemble',
        'method': 'VAE-HyperNetFusion集成（仿RF多树）',
        'timestamp': datetime.now().isoformat(),
        'mean_accuracy': mean_acc / 100,
        'std_accuracy': std_acc / 100,
        'overall_accuracy': overall_acc / 100,
        'elapsed_time': elapsed_time,
        'folds_per_second': n_folds / elapsed_time,
        'n_samples': n_samples,
        'n_folds': n_folds,
        'n_gpus': len(available_gpus),
        'n_estimators': config['n_estimators'],
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
