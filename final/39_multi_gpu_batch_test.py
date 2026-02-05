#!/usr/bin/env python3
"""
39_multi_gpu_batch_test.py - 多GPU并行批量测试所有数据集
充分利用6个A100 GPU，每个GPU并行处理多个数据集
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
import json
from datetime import datetime
from pathlib import Path
import random
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import traceback
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ============== 模型定义 ==============
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        hidden = max(32, input_dim * 4)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden // 2, latent_dim)
        self.fc_var = nn.Linear(hidden // 2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim)
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class HyperNetworkForTree(nn.Module):
    def __init__(self, input_dim, n_trees=15, tree_depth=3, num_classes=2):
        super().__init__()
        self.input_dim = input_dim
        self.n_trees = n_trees
        self.tree_depth = tree_depth
        self.num_classes = num_classes
        self.n_internal = 2 ** tree_depth - 1
        self.n_leaves = 2 ** tree_depth
        
        hidden_dim = max(64, input_dim * 8)
        
        self.data_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.context_dim = hidden_dim
        self.split_weight_gen = nn.Linear(self.context_dim, n_trees * self.n_internal * input_dim)
        self.split_bias_gen = nn.Linear(self.context_dim, n_trees * self.n_internal)
        self.leaf_gen = nn.Linear(self.context_dim, n_trees * self.n_leaves * num_classes)
        self.tree_weight_gen = nn.Linear(self.context_dim, n_trees)
    
    def forward(self, X_context):
        context = self.data_encoder(X_context)
        context = context.mean(dim=0)
        
        split_weights = self.split_weight_gen(context).view(self.n_trees, self.n_internal, self.input_dim)
        split_bias = self.split_bias_gen(context).view(self.n_trees, self.n_internal)
        leaf_logits = self.leaf_gen(context).view(self.n_trees, self.n_leaves, self.num_classes)
        tree_weights = F.softmax(self.tree_weight_gen(context), dim=0)
        
        return split_weights, split_bias, leaf_logits, tree_weights


class GeneratedTreeClassifier(nn.Module):
    def __init__(self, tree_depth=3, temperature=1.0):
        super().__init__()
        self.tree_depth = tree_depth
        self.temperature = temperature
    
    def forward(self, x, split_weights, split_bias, leaf_logits, tree_weights):
        batch_size = x.shape[0]
        n_trees = split_weights.shape[0]
        n_leaves = leaf_logits.shape[1]
        
        all_probs = []
        
        for t in range(n_trees):
            leaf_probs = torch.ones(batch_size, n_leaves, device=x.device)
            
            for depth in range(self.tree_depth):
                start_idx = 2 ** depth - 1
                n_nodes = 2 ** depth
                
                for node_offset in range(n_nodes):
                    node_idx = start_idx + node_offset
                    decision = torch.sigmoid(
                        (x @ split_weights[t, node_idx] + split_bias[t, node_idx]) / self.temperature
                    )
                    
                    left_start = node_offset * (n_leaves // n_nodes)
                    left_end = left_start + (n_leaves // n_nodes) // 2
                    right_end = left_start + (n_leaves // n_nodes)
                    
                    leaf_probs[:, left_start:left_end] *= (1 - decision).unsqueeze(1)
                    leaf_probs[:, left_end:right_end] *= decision.unsqueeze(1)
            
            tree_output = torch.softmax(leaf_logits[t], dim=-1)
            tree_pred = torch.einsum('bl,lc->bc', leaf_probs, tree_output)
            all_probs.append(tree_pred * tree_weights[t])
        
        return torch.stack(all_probs, dim=0).sum(dim=0)


class VAEHyperNetFusion(nn.Module):
    def __init__(self, input_dim, num_classes=2, n_trees=15, tree_depth=3, latent_dim=8):
        super().__init__()
        self.vae = VAE(input_dim, latent_dim)
        self.hypernet = HyperNetworkForTree(input_dim, n_trees, tree_depth, num_classes)
        self.tree_classifier = GeneratedTreeClassifier(tree_depth)
        self.num_classes = num_classes
    
    def augment_data(self, X, n_augment=100):
        self.vae.eval()
        with torch.no_grad():
            recon, mu, logvar = self.vae(X)
            augmented = [X]
            for _ in range(n_augment // len(X) + 1):
                z = self.vae.reparameterize(mu, logvar)
                generated = self.vae.decode(z)
                alpha = torch.rand(len(X), 1, device=X.device) * 0.3 + 0.7
                mixed = alpha * X + (1 - alpha) * generated
                augmented.append(mixed)
            return torch.cat(augmented, dim=0)[:n_augment]
    
    def forward(self, X_train, X_test):
        split_w, split_b, leaf_logits, tree_w = self.hypernet(X_train)
        return self.tree_classifier(X_test, split_w, split_b, leaf_logits, tree_w)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset(data_dir, dataset_name):
    """加载数据集 - 支持多种格式"""
    
    # 数据集配置
    dataset_configs = {
        '1.balloons': {
            'file': 'adult-stretch.data',
            'sep': ',',
            'header': None,
            'names': ['color', 'size', 'act', 'age', 'inflated']
        },
        '2lens': {
            'file': 'lenses.data',
            'sep': r'\s+',
            'header': None,
            'names': ['id', 'age', 'prescription', 'astigmatic', 'tear_rate', 'class'],
            'drop_cols': ['id']
        },
        '3.caesarian+section+classification+dataset': {
            'file': 'caesarian.csv',
            'sep': ',',
            'header': 0
        },
        '4.iris': {
            'file': 'iris.data',
            'sep': ',',
            'header': None,
            'names': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
        },
        '5.fertility': {
            'file': 'fertility_Diagnosis.txt',
            'sep': ',',
            'header': None,
            'names': ['season', 'age', 'diseases', 'accident', 'surgery', 'fever', 'alcohol', 'smoking', 'sitting', 'diagnosis']
        },
        '6.zoo': {
            'file': 'zoo.data',
            'sep': ',',
            'header': None,
            'names': ['name', 'hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 'predator', 'toothed', 'backbone', 'breathes', 'venomous', 'fins', 'legs', 'tail', 'domestic', 'catsize', 'type'],
            'drop_cols': ['name']
        },
        '7.seeds': {
            'file': 'seeds_dataset.txt',
            'sep': r'\s+',
            'header': None,
            'names': ['area', 'perimeter', 'compactness', 'length', 'width', 'asymmetry', 'groove', 'class']
        },
        '8.haberman+s+survival': {
            'file': 'haberman.data',
            'sep': ',',
            'header': None,
            'names': ['age', 'year', 'nodes', 'status']
        },
        '9.glass+identification': {
            'file': 'glass.data',
            'sep': ',',
            'header': None,
            'names': ['id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'class'],
            'drop_cols': ['id']
        },
        '10.yeast': {
            'file': 'yeast.data',
            'sep': r'\s+',
            'header': None,
            'names': ['name', 'mcg', 'gvh', 'alm', 'mit', 'erl', 'pox', 'vac', 'nuc', 'class'],
            'drop_cols': ['name']
        }
    }
    
    config = dataset_configs.get(dataset_name)
    if not config:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    file_path = os.path.join(data_dir, dataset_name, config['file'])
    
    # 读取数据
    if 'names' in config:
        df = pd.read_csv(file_path, sep=config['sep'], header=config.get('header'), names=config['names'])
    else:
        df = pd.read_csv(file_path, sep=config['sep'], header=config.get('header', 0))
    
    # 删除指定列
    if 'drop_cols' in config:
        df = df.drop(columns=config['drop_cols'], errors='ignore')
    
    # 分离特征和标签
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # 编码特征
    X_encoded = []
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X_encoded.append(le.fit_transform(X[col].astype(str)))
        else:
            X_encoded.append(X[col].values)
    
    X = np.column_stack(X_encoded).astype(np.float32)
    
    # 处理NaN
    X = np.nan_to_num(X, nan=0.0)
    
    # 编码标签
    le = LabelEncoder()
    y = le.fit_transform(y.astype(str))
    
    return X, y, le.classes_


def train_single_fold(fold_idx, train_idx, test_idx, X, y, num_classes, input_dim, device, seed):
    """训练单个fold"""
    set_seed(seed + fold_idx)
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_t = torch.FloatTensor(X_train_scaled).to(device)
    X_test_t = torch.FloatTensor(X_test_scaled).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    
    # 创建模型
    model = VAEHyperNetFusion(input_dim, num_classes, n_trees=15, tree_depth=3).to(device)
    
    # 训练VAE
    vae_optimizer = torch.optim.Adam(model.vae.parameters(), lr=0.001)
    model.vae.train()
    for _ in range(100):
        recon, mu, logvar = model.vae(X_train_t)
        recon_loss = F.mse_loss(recon, X_train_t)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(X_train_t)
        loss = recon_loss + 0.1 * kl_loss
        vae_optimizer.zero_grad()
        loss.backward()
        vae_optimizer.step()
    
    # 数据增强
    model.vae.eval()
    X_aug = model.augment_data(X_train_t, n_augment=max(200, len(X_train) * 5))
    y_aug = y_train_t.repeat((len(X_aug) // len(y_train_t)) + 1)[:len(X_aug)]
    
    # 训练HyperNet
    hypernet_optimizer = torch.optim.Adam(model.hypernet.parameters(), lr=0.001, weight_decay=1e-4)
    model.hypernet.train()
    
    best_loss = float('inf')
    patience = 0
    
    for epoch in range(300):
        probs = model(X_aug, X_aug)
        loss = F.cross_entropy(probs, y_aug)
        
        hypernet_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.hypernet.parameters(), 1.0)
        hypernet_optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience = 0
        else:
            patience += 1
            if patience > 50:
                break
    
    # 评估
    model.eval()
    with torch.no_grad():
        probs = model(X_aug, X_test_t)
        preds = probs.argmax(dim=1).cpu().numpy()
    
    acc = accuracy_score(y_test, preds)
    
    # 清理
    del model, X_train_t, X_test_t, X_aug, y_aug
    torch.cuda.empty_cache()
    
    return acc, y_test.tolist(), preds.tolist()


def run_dataset_on_gpu(args):
    """在指定GPU上运行数据集"""
    dataset_name, data_dir, gpu_id, n_folds, n_runs = args
    
    results = {
        'dataset': dataset_name,
        'gpu_id': gpu_id,
        'start_time': datetime.now().isoformat()
    }
    
    try:
        device = torch.device(f'cuda:{gpu_id}')
        
        # 加载数据
        X, y, classes = load_dataset(data_dir, dataset_name)
        n_samples, input_dim = X.shape
        num_classes = len(np.unique(y))
        
        results['n_samples'] = n_samples
        results['n_features'] = input_dim
        results['n_classes'] = num_classes
        results['class_distribution'] = {str(c): int((y == i).sum()) for i, c in enumerate(classes)}
        
        print(f"[GPU {gpu_id}] {dataset_name}: {n_samples} samples, {input_dim} features, {num_classes} classes")
        
        # RF基线
        rf_accs = []
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        for train_idx, test_idx in skf.split(X, y):
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X[train_idx], y[train_idx])
            rf_accs.append(accuracy_score(y[test_idx], rf.predict(X[test_idx])))
        
        results['rf_baseline'] = {
            'mean': float(np.mean(rf_accs) * 100),
            'std': float(np.std(rf_accs) * 100),
            'fold_accs': [a * 100 for a in rf_accs]
        }
        print(f"[GPU {gpu_id}] {dataset_name} RF: {results['rf_baseline']['mean']:.2f}% +/- {results['rf_baseline']['std']:.2f}%")
        
        # VAE-HyperNet 多次运行
        all_accs = []
        all_y_true = []
        all_y_pred = []
        
        for run_id in range(n_runs):
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42 + run_id)
            
            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                acc, y_true, y_pred = train_single_fold(
                    fold_idx, train_idx, test_idx, X, y, num_classes, input_dim, device, 42 + run_id * 100
                )
                all_accs.append(acc)
                all_y_true.extend(y_true)
                all_y_pred.extend(y_pred)
            
            print(f"[GPU {gpu_id}] {dataset_name} Run {run_id+1}/{n_runs}: {np.mean(all_accs[-n_folds:])*100:.2f}%")
        
        # 汇总
        results['vae_hypernet'] = {
            'mean': float(np.mean(all_accs) * 100),
            'std': float(np.std(all_accs) * 100),
            'all_fold_accs': [a * 100 for a in all_accs]
        }
        
        # 计算其他指标
        results['metrics'] = {
            'precision': float(precision_score(all_y_true, all_y_pred, average='weighted', zero_division=0) * 100),
            'recall': float(recall_score(all_y_true, all_y_pred, average='weighted', zero_division=0) * 100),
            'f1': float(f1_score(all_y_true, all_y_pred, average='weighted', zero_division=0) * 100)
        }
        
        results['end_time'] = datetime.now().isoformat()
        results['status'] = 'success'
        
        print(f"[GPU {gpu_id}] DONE {dataset_name}: VAE-HyperNet {results['vae_hypernet']['mean']:.2f}% +/- {results['vae_hypernet']['std']:.2f}% | RF {results['rf_baseline']['mean']:.2f}%")
        
    except Exception as e:
        results['status'] = 'error'
        results['error'] = str(e)
        results['traceback'] = traceback.format_exc()
        print(f"[GPU {gpu_id}] ERROR {dataset_name}: {e}")
    
    return results


def main():
    print("=" * 70)
    print("Multi-GPU Batch Test - VAE-HyperNet-Fusion")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    data_dir = "/data2/image_identification/data/小数据"
    output_dir = Path("/data2/image_identification/src/final/output")
    output_dir.mkdir(exist_ok=True)
    
    # 数据集列表
    datasets = [
        '1.balloons',
        '2lens',
        '3.caesarian+section+classification+dataset',
        '4.iris',
        '5.fertility',
        '6.zoo',
        '7.seeds',
        '8.haberman+s+survival',
        '9.glass+identification',
        '10.yeast'
    ]
    
    print(f"Datasets: {len(datasets)}")
    
    # 可用GPU (0-5)
    available_gpus = [0, 1, 2, 3, 4, 5]
    
    # 配置
    n_folds = 5
    n_runs = 3
    
    # 分配任务
    tasks = []
    for i, ds in enumerate(datasets):
        gpu_id = available_gpus[i % len(available_gpus)]
        tasks.append((ds, data_dir, gpu_id, n_folds, n_runs))
        print(f"  {ds} -> GPU {gpu_id}")
    
    print("\nStarting parallel test...")
    start_time = time.time()
    
    # 并行执行
    all_results = []
    with ProcessPoolExecutor(max_workers=len(available_gpus)) as executor:
        futures = {executor.submit(run_dataset_on_gpu, task): task[0] for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            all_results.append(result)
    
    total_time = time.time() - start_time
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON详细结果
    results_file = output_dir / f"results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'total_time_minutes': total_time / 60,
            'n_folds': n_folds,
            'n_runs': n_runs,
            'datasets': all_results
        }, f, indent=2)
    
    # CSV汇总
    summary_rows = []
    for r in sorted(all_results, key=lambda x: x.get('dataset', '')):
        if r.get('status') == 'success':
            summary_rows.append({
                'Dataset': r['dataset'],
                'Samples': r['n_samples'],
                'Features': r['n_features'],
                'Classes': r['n_classes'],
                'RF_Mean': r['rf_baseline']['mean'],
                'RF_Std': r['rf_baseline']['std'],
                'VAE_HyperNet_Mean': r['vae_hypernet']['mean'],
                'VAE_HyperNet_Std': r['vae_hypernet']['std'],
                'Precision': r['metrics']['precision'],
                'Recall': r['metrics']['recall'],
                'F1': r['metrics']['f1']
            })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_dir / f"summary_{timestamp}.csv"
    summary_df.to_csv(summary_csv, index=False)
    
    # 打印结果
    print("\n" + "=" * 90)
    print("RESULTS SUMMARY")
    print("=" * 90)
    print(f"{'Dataset':<40} {'N':<6} {'Feat':<6} {'Class':<6} {'RF':<12} {'VAE-HyperNet':<12}")
    print("-" * 90)
    
    for r in sorted(all_results, key=lambda x: x.get('dataset', '')):
        if r.get('status') == 'success':
            rf = f"{r['rf_baseline']['mean']:.1f}+/-{r['rf_baseline']['std']:.1f}"
            vae = f"{r['vae_hypernet']['mean']:.1f}+/-{r['vae_hypernet']['std']:.1f}"
            print(f"{r['dataset']:<40} {r['n_samples']:<6} {r['n_features']:<6} {r['n_classes']:<6} {rf:<12} {vae:<12}")
        else:
            print(f"{r.get('dataset', 'Unknown'):<40} ERROR: {r.get('error', '')[:30]}")
    
    print("-" * 90)
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Results saved: {results_file}")
    print(f"Summary saved: {summary_csv}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
