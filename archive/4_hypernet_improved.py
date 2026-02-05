#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VAE-HyperNetFusion V1 改进版
============================
针对小样本数据优化的 HyperNetFusion 神经网络

改进点：
1. 简化网络结构 - 减少参数量
2. 添加 Dropout 正则化
3. 添加权重衰减 (L2 正则化)
4. 添加早停机制
5. 添加梯度裁剪
6. 使用更多的 VAE 增强数据

作者: Jinming Zhang
日期: 2026-02-03
"""

import os
import sys
import json
import time
import warnings
import multiprocessing as mp
from datetime import datetime
from itertools import combinations
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

warnings.filterwarnings('ignore')


# ==================== 改进的 HyperNetFusion 模型 ====================
class ImprovedHyperNet(nn.Module):
    """
    改进的超网络 - 生成多个轻量级目标网络的参数
    
    针对小样本数据的改进：
    - 减少网络深度和宽度
    - 添加 Dropout
    - 使用 LayerNorm 替代 BatchNorm（小batch更稳定）
    """
    def __init__(self, input_dim, hidden_dim=32, num_target_nets=5, target_param_size=None, dropout=0.3):
        super(ImprovedHyperNet, self).__init__()
        
        self.num_target_nets = num_target_nets
        self.target_param_size = target_param_size
        
        # 轻量级超网络
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 为每个目标网络生成参数
        self.param_generators = nn.ModuleList([
            nn.Linear(hidden_dim, target_param_size) for _ in range(num_target_nets)
        ])
    
    def forward(self, x):
        """返回每个目标网络的参数"""
        h = self.encoder(x)
        # 对batch内的特征取平均，生成全局特征
        h_global = h.mean(dim=0) if h.dim() > 1 else h
        
        params = [gen(h_global) for gen in self.param_generators]
        return params


class ImprovedTargetNet(nn.Module):
    """
    改进的目标网络 - 轻量级分类器
    
    参数由超网络动态生成
    """
    def __init__(self, input_dim, hidden_dim=16, num_classes=2):
        super(ImprovedTargetNet, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # 计算需要的参数量
        self.w1_size = input_dim * hidden_dim
        self.b1_size = hidden_dim
        self.w2_size = hidden_dim * num_classes
        self.b2_size = num_classes
        
        self.total_params = self.w1_size + self.b1_size + self.w2_size + self.b2_size
    
    def forward(self, x, params):
        """使用给定参数进行前向传播"""
        # 解析参数
        idx = 0
        w1 = params[idx:idx + self.w1_size].view(self.hidden_dim, self.input_dim)
        idx += self.w1_size
        b1 = params[idx:idx + self.b1_size]
        idx += self.b1_size
        w2 = params[idx:idx + self.w2_size].view(self.num_classes, self.hidden_dim)
        idx += self.w2_size
        b2 = params[idx:idx + self.b2_size]
        
        # 前向传播
        h = torch.relu(torch.mm(x, w1.t()) + b1)
        out = torch.mm(h, w2.t()) + b2
        return out


class ImprovedHyperNetFusion(nn.Module):
    """
    改进的 HyperNetFusion - 针对小样本数据优化
    
    关键改进：
    1. 减少子网络数量 (10 -> 5)
    2. 减小网络宽度 (128 -> 32, 32 -> 16)
    3. 添加 Dropout 正则化
    4. 集成投票机制
    """
    def __init__(self, input_dim, hypernet_hidden_dim=32, target_hidden_dim=16, 
                 num_classes=2, num_target_nets=5, dropout=0.3):
        super(ImprovedHyperNetFusion, self).__init__()
        
        self.num_target_nets = num_target_nets
        self.num_classes = num_classes
        
        # 创建目标网络模板（用于计算参数量）
        self.target_template = ImprovedTargetNet(input_dim, target_hidden_dim, num_classes)
        target_param_size = self.target_template.total_params
        
        # 创建超网络
        self.hypernet = ImprovedHyperNet(
            input_dim=input_dim,
            hidden_dim=hypernet_hidden_dim,
            num_target_nets=num_target_nets,
            target_param_size=target_param_size,
            dropout=dropout
        )
        
        # 创建多个目标网络
        self.target_nets = nn.ModuleList([
            ImprovedTargetNet(input_dim, target_hidden_dim, num_classes) 
            for _ in range(num_target_nets)
        ])
        
        # 可学习的集成权重
        self.ensemble_weights = nn.Parameter(torch.ones(num_target_nets) / num_target_nets)
    
    def forward(self, x):
        """前向传播 - 集成多个目标网络的预测"""
        # 生成每个目标网络的参数
        all_params = self.hypernet(x)
        
        # 每个目标网络的预测
        all_outputs = []
        for i, (target_net, params) in enumerate(zip(self.target_nets, all_params)):
            out = target_net(x, params)
            all_outputs.append(out)
        
        # 加权平均（软投票）
        stacked = torch.stack(all_outputs, dim=0)  # (num_nets, batch, classes)
        weights = torch.softmax(self.ensemble_weights, dim=0).view(-1, 1, 1)
        ensemble_out = (stacked * weights).sum(dim=0)
        
        return ensemble_out
    
    def predict_proba(self, x):
        """返回概率预测"""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)


# ==================== VAE 数据增强 ====================
class VAE(nn.Module):
    """变分自编码器"""
    def __init__(self, input_dim, hidden_dim=32, latent_dim=2):
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
    """VAE 数据增强"""
    X_aug_list = [X.copy()]
    y_aug_list = [y.copy()]
    
    classes = np.unique(y)
    for cls in classes:
        X_cls = X[y == cls]
        if len(X_cls) < 2:
            continue
        
        # 训练VAE
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
        
        # 生成增强样本
        vae.eval()
        with torch.no_grad():
            recon, _, _ = vae(X_tensor)
            recon_np = recon.numpy()
        
        # 线性插值
        for alpha in np.linspace(0.1, 0.9, config['num_interpolation_points']):
            augmented = alpha * X_cls + (1 - alpha) * recon_np
            X_aug_list.append(augmented)
            y_aug_list.append(np.full(len(augmented), cls))
    
    return np.vstack(X_aug_list), np.hstack(y_aug_list)


# ==================== 单fold处理函数 ====================
def process_single_fold(fold_info, X, y, config):
    """处理单个fold"""
    fold_idx, (train_idx, test_idx) = fold_info
    
    try:
        # 分割数据
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # VAE增强
        X_aug, y_aug = vae_augment(X_train_scaled, y_train, config)
        
        # 转为张量
        X_train_t = torch.FloatTensor(X_aug)
        y_train_t = torch.LongTensor(y_aug)
        X_test_t = torch.FloatTensor(X_test_scaled)
        
        # 创建模型
        model = ImprovedHyperNetFusion(
            input_dim=X.shape[1],
            hypernet_hidden_dim=config['hypernet_hidden_dim'],
            target_hidden_dim=config['target_hidden_dim'],
            num_classes=len(np.unique(y)),
            num_target_nets=config['num_target_nets'],
            dropout=config['dropout']
        )
        
        # 训练配置
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
        
        # 创建 DataLoader
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        
        # 训练循环（带早停）
        best_loss = float('inf')
        patience_counter = 0
        
        model.train()
        for epoch in range(config['epochs']):
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_loss += loss.item()
            
            scheduler.step()
            
            # 早停检查
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config['patience']:
                    break
        
        # 预测
        model.eval()
        with torch.no_grad():
            proba = model.predict_proba(X_test_t).numpy()
            pred = np.argmax(proba, axis=1)
        
        accuracy = accuracy_score(y_test, pred)
        
        return {
            'fold_idx': fold_idx,
            'accuracy': accuracy,
            'y_true': y_test.tolist(),
            'y_pred': pred.tolist(),
            'y_prob': proba.tolist()
        }
    except Exception as e:
        return {
            'fold_idx': fold_idx,
            'accuracy': 0.0,
            'error': str(e)
        }


def run_experiment(X, y, config, n_processes=None):
    """运行实验"""
    n_samples = len(X)
    p = config['leave_p_out']
    
    # 生成所有fold
    all_indices = np.arange(n_samples)
    test_combinations = list(combinations(all_indices, p))
    n_folds = len(test_combinations)
    
    print(f"\n[INFO] Leave-{p}-Out 交叉验证")
    print(f"  - 样本数: {n_samples}, fold数: {n_folds}")
    
    # 准备fold信息
    fold_infos = []
    for fold_idx, test_idx in enumerate(test_combinations):
        test_idx = np.array(test_idx)
        train_idx = np.setdiff1d(all_indices, test_idx)
        fold_infos.append((fold_idx, (train_idx, test_idx)))
    
    # 确定进程数
    if n_processes is None:
        n_processes = min(mp.cpu_count(), n_folds, 32)
    
    print(f"  - 使用 {n_processes} 个进程")
    
    start_time = time.time()
    
    # 并行处理
    process_func = partial(process_single_fold, X=X, y=y, config=config)
    
    with mp.Pool(processes=n_processes) as pool:
        results = list(pool.imap(process_func, fold_infos, chunksize=max(1, n_folds // n_processes)))
    
    elapsed_time = time.time() - start_time
    
    # 统计结果
    accuracies = [r['accuracy'] for r in results if 'error' not in r]
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
    
    print(f"\n{'='*60}")
    print(f"[结果] Leave-{p}-Out 完成")
    print(f"  - 平均准确率: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"  - 整体准确率: {overall_acc:.2f}%")
    if overall_auc:
        print(f"  - AUC: {overall_auc:.4f}")
    print(f"  - 用时: {elapsed_time:.2f}秒")
    print(f"{'='*60}")
    
    return {
        'mean_accuracy': mean_acc / 100,
        'std_accuracy': std_acc / 100,
        'overall_accuracy': overall_acc / 100,
        'overall_auc': overall_auc,
        'elapsed_time': elapsed_time,
        'config': config
    }


def main():
    """主函数"""
    print("="*60)
    print("VAE-HyperNetFusion V1 改进版")
    print("针对小样本数据优化的神经网络方法")
    print("="*60)
    
    # 改进后的配置
    config = {
        # VAE参数
        'vae_latent_dim': 2,
        'vae_hidden_dim': 32,
        'vae_epochs': 100,
        'vae_lr': 0.001,
        'vae_kl_weight': 0.5,
        'num_interpolation_points': 10,  # 增加插值点
        
        # HyperNetFusion参数 (简化版)
        'hypernet_hidden_dim': 32,    # 减小 (原128)
        'target_hidden_dim': 16,      # 减小 (原32)
        'num_target_nets': 5,         # 减少 (原10)
        'dropout': 0.3,               # 新增
        
        # 训练参数
        'epochs': 200,
        'lr': 0.005,
        'weight_decay': 0.01,         # L2正则化
        'batch_size': 16,
        'patience': 30,               # 早停
        
        # 验证
        'leave_p_out': 2
    }
    
    # 加载数据
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'data', 'Data_for_Jinming.csv')
    
    if not os.path.exists(data_path):
        print(f"[错误] 找不到数据: {data_path}")
        sys.exit(1)
    
    print(f"\n[INFO] 加载数据: {data_path}")
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
    
    print(f"  - 特征: {feature_cols}")
    print(f"  - 样本数: {len(X)}, 特征数: {X.shape[1]}")
    print(f"  - 类别: {le.classes_}")
    
    # 运行实验
    n_cpu = mp.cpu_count()
    n_processes = min(32, n_cpu)
    
    results = run_experiment(X, y, config, n_processes=n_processes)
    
    # 保存结果
    output_dir = os.path.join(script_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = os.path.join(output_dir, f'v1_improved_results_{timestamp}.json')
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[INFO] 结果已保存: {result_file}")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
