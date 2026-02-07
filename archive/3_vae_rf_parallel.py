#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VAE-HyperNetFusion 多进程并行版本
使用 Leave-P-Out (p=2) 交叉验证
针对小样本表格数据的分类任务

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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from skopt import gp_minimize
from skopt.space import Integer

warnings.filterwarnings('ignore')

# ==================== VAE 数据增强模块 ====================
class VAE(nn.Module):
    """变分自编码器"""
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


class VAEDataAugmentation:
    """VAE数据增强类"""
    def __init__(self, latent_dim=2, hidden_dim=64, epochs=50, lr=0.001, 
                 kl_weight=0.5, num_interpolation_points=5):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.kl_weight = kl_weight
        self.num_interpolation_points = num_interpolation_points
    
    def augment(self, X, y, verbose=False):
        """对数据进行VAE增强"""
        X_aug_list = [X.copy()]
        y_aug_list = [y.copy()]
        
        classes = np.unique(y)
        for cls in classes:
            X_cls = X[y == cls]
            if len(X_cls) < 2:
                continue
            
            # 训练VAE
            input_dim = X_cls.shape[1]
            vae = VAE(input_dim, self.hidden_dim, self.latent_dim)
            optimizer = torch.optim.Adam(vae.parameters(), lr=self.lr)
            
            X_tensor = torch.FloatTensor(X_cls)
            vae.train()
            for _ in range(self.epochs):
                optimizer.zero_grad()
                recon, mu, log_var = vae(X_tensor)
                recon_loss = nn.MSELoss()(recon, X_tensor)
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = recon_loss + self.kl_weight * kl_loss / len(X_cls)
                loss.backward()
                optimizer.step()
            
            # 生成增强样本
            vae.eval()
            with torch.no_grad():
                recon, _, _ = vae(X_tensor)
                recon_np = recon.numpy()
            
            # 线性插值生成新样本
            for alpha in np.linspace(0.1, 0.9, self.num_interpolation_points):
                augmented = alpha * X_cls + (1 - alpha) * recon_np
                X_aug_list.append(augmented)
                y_aug_list.append(np.full(len(augmented), cls))
        
        return np.vstack(X_aug_list), np.hstack(y_aug_list)


# ==================== 单个fold的处理函数 ====================
def process_single_fold(fold_info, X, y, config):
    """处理单个fold的函数（用于多进程）"""
    fold_idx, (train_idx, test_idx) = fold_info
    
    try:
        # 分割数据
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 特征标准化（仅在训练集上fit）
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # VAE数据增强
        vae_aug = VAEDataAugmentation(
            latent_dim=config['vae_latent_dim'],
            hidden_dim=config['vae_hidden_dim'],
            epochs=config['vae_epochs'],
            lr=config['vae_lr'],
            kl_weight=config['vae_kl_weight'],
            num_interpolation_points=config['num_interpolation_points']
        )
        X_train_aug, y_train_aug = vae_aug.augment(X_train_scaled, y_train)
        
        # 训练随机森林分类器
        clf = RandomForestClassifier(
            n_estimators=config['rf_n_estimators'],
            max_depth=config['rf_max_depth'],
            random_state=42,
            n_jobs=1  # 每个进程内单线程
        )
        clf.fit(X_train_aug, y_train_aug)
        
        # 预测
        y_pred = clf.predict(X_test_scaled)
        y_prob = clf.predict_proba(X_test_scaled)
        
        # 计算准确率
        accuracy = accuracy_score(y_test, y_pred)
        
        return {
            'fold_idx': fold_idx,
            'accuracy': accuracy,
            'y_true': y_test.tolist(),
            'y_pred': y_pred.tolist(),
            'y_prob': y_prob.tolist(),
            'test_idx': test_idx.tolist()
        }
    except Exception as e:
        return {
            'fold_idx': fold_idx,
            'accuracy': 0.0,
            'error': str(e),
            'test_idx': test_idx.tolist()
        }


def run_parallel_cv(X, y, config, n_processes=None):
    """并行运行Leave-P-Out交叉验证"""
    n_samples = len(X)
    p = config['leave_p_out']
    
    # 生成所有fold的索引
    all_indices = np.arange(n_samples)
    test_combinations = list(combinations(all_indices, p))
    n_folds = len(test_combinations)
    
    print(f"\n[INFO] Leave-{p}-Out 交叉验证")
    print(f"  - 样本数: {n_samples}")
    print(f"  - 总fold数: {n_folds}")
    
    # 准备fold信息
    fold_infos = []
    for fold_idx, test_idx in enumerate(test_combinations):
        test_idx = np.array(test_idx)
        train_idx = np.setdiff1d(all_indices, test_idx)
        fold_infos.append((fold_idx, (train_idx, test_idx)))
    
    # 确定进程数
    if n_processes is None:
        n_processes = min(mp.cpu_count(), n_folds, 64)  # 最多64进程
    
    print(f"  - 使用进程数: {n_processes}")
    print(f"\n[INFO] 开始并行训练...")
    
    start_time = time.time()
    
    # 使用进程池并行处理
    process_func = partial(process_single_fold, X=X, y=y, config=config)
    
    with mp.Pool(processes=n_processes) as pool:
        results = list(pool.imap(process_func, fold_infos, chunksize=max(1, n_folds // n_processes)))
    
    elapsed_time = time.time() - start_time
    
    # 统计结果
    accuracies = [r['accuracy'] for r in results if 'error' not in r]
    errors = [r for r in results if 'error' in r]
    
    mean_acc = np.mean(accuracies) * 100
    std_acc = np.std(accuracies) * 100
    
    # 汇总所有预测用于计算整体指标
    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    
    for r in results:
        if 'error' not in r:
            all_y_true.extend(r['y_true'])
            all_y_pred.extend(r['y_pred'])
            all_y_prob.extend([p[1] if len(p) > 1 else p[0] for p in r['y_prob']])
    
    overall_acc = accuracy_score(all_y_true, all_y_pred) * 100
    
    # 计算AUC（如果是二分类）
    try:
        overall_auc = roc_auc_score(all_y_true, all_y_prob)
    except:
        overall_auc = None
    
    print(f"\n{'='*60}")
    print(f"[结果] Leave-{p}-Out 完成")
    print(f"  - 平均准确率: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"  - 整体准确率: {overall_acc:.2f}%")
    if overall_auc:
        print(f"  - 整体AUC: {overall_auc:.4f}")
    print(f"  - 用时: {elapsed_time:.2f}秒")
    if errors:
        print(f"  - 错误fold数: {len(errors)}")
    print(f"{'='*60}")
    
    return {
        'mean_accuracy': mean_acc / 100,
        'std_accuracy': std_acc / 100,
        'overall_accuracy': overall_acc / 100,
        'overall_auc': overall_auc,
        'elapsed_time': elapsed_time,
        'n_folds': n_folds,
        'n_errors': len(errors),
        'all_results': results
    }


def main():
    """主函数"""
    print("="*60)
    print("VAE-HyperNetFusion 多进程并行版本")
    print("="*60)
    
    # 配置参数
    config = {
        # VAE参数
        'vae_latent_dim': 2,
        'vae_hidden_dim': 64,
        'vae_epochs': 50,
        'vae_lr': 0.001,
        'vae_kl_weight': 0.5,
        'num_interpolation_points': 5,
        
        # 随机森林参数
        'rf_n_estimators': 150,
        'rf_max_depth': 5,
        
        # 交叉验证参数
        'leave_p_out': 2
    }
    
    # 确定数据路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'data', 'Data_for_Jinming.csv')
    
    if not os.path.exists(data_path):
        print(f"[错误] 找不到数据文件: {data_path}")
        sys.exit(1)
    
    # 加载数据
    print(f"\n[INFO] 加载数据: {data_path}")
    df = pd.read_csv(data_path)
    print(f"  - 数据形状: {df.shape}")
    print(f"  - 列名: {list(df.columns)}")
    
    # 准备特征和标签
    # 支持多种常见的标签列名
    possible_targets = ['Class', 'class', 'label', 'target', 'Group', 'group']
    possible_ids = ['Sample ID', 'sample_id', 'ID', 'id']
    
    target_col = None
    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break
    
    if target_col is None:
        print(f"[错误] 找不到标签列，可用列名: {list(df.columns)}")
        sys.exit(1)
    
    # 排除标签列和ID列
    exclude_cols = [target_col] + [col for col in possible_ids if col in df.columns]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].values.astype(np.float32)
    y_raw = df[target_col].values
    
    # 标签编码
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    print(f"  - 特征数: {X.shape[1]}")
    print(f"  - 样本数: {X.shape[0]}")
    print(f"  - 类别: {le.classes_}")
    print(f"  - 类别分布: {np.bincount(y)}")
    
    # 确定进程数（服务器有112核，使用64核）
    n_cpu = mp.cpu_count()
    n_processes = min(64, n_cpu)
    print(f"\n[INFO] 检测到 {n_cpu} 个CPU核心，使用 {n_processes} 个进程")
    
    # 运行并行交叉验证
    results = run_parallel_cv(X, y, config, n_processes=n_processes)
    
    # 保存结果
    output_dir = os.path.join(script_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = os.path.join(output_dir, f'parallel_results_{timestamp}.json')
    
    # 只保存汇总结果（不保存所有fold的详细结果以节省空间）
    summary = {
        'mean_accuracy': results['mean_accuracy'],
        'std_accuracy': results['std_accuracy'],
        'overall_accuracy': results['overall_accuracy'],
        'overall_auc': results['overall_auc'],
        'elapsed_time': results['elapsed_time'],
        'n_folds': results['n_folds'],
        'n_errors': results['n_errors'],
        'n_processes': n_processes,
        'config': config
    }
    
    with open(result_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[INFO] 结果已保存: {result_file}")
    
    # 保存混淆矩阵
    all_y_true = []
    all_y_pred = []
    for r in results['all_results']:
        if 'error' not in r:
            all_y_true.extend(r['y_true'])
            all_y_pred.extend(r['y_pred'])
    
    cm = confusion_matrix(all_y_true, all_y_pred)
    cm_file = os.path.join(output_dir, f'confusion_matrix_{timestamp}.csv')
    pd.DataFrame(cm).to_csv(cm_file, index=False)
    print(f"[INFO] 混淆矩阵已保存: {cm_file}")
    
    print(f"\n{'='*60}")
    print(f"[完成] 实验结束！")
    print(f"  Leave-P-Out 准确率: {results['mean_accuracy']*100:.2f}% ± {results['std_accuracy']*100:.2f}%")
    print(f"  整体准确率: {results['overall_accuracy']*100:.2f}%")
    if results['overall_auc']:
        print(f"  整体AUC: {results['overall_auc']:.4f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    # 设置多进程启动方式（macOS需要）
    mp.set_start_method('spawn', force=True)
    main()
