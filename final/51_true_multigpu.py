#!/usr/bin/env python3
"""
51_true_multigpu.py - 真正的多GPU并行版本
每个数据集分配到不同的GPU上同时运行
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from itertools import combinations
import warnings
import time
import logging
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess

warnings.filterwarnings('ignore')

# 配置日志
log_dir = "/data2/image_identification/src/output"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "51_multigpu_log.txt")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ========================= 模型定义 =========================

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        hidden = max(16, input_dim * 2)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_var = nn.Linear(hidden, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim)
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


class MultiHeadHyperNet(nn.Module):
    """多头HyperNet - 生成多个软决策树的权重"""
    def __init__(self, input_dim, n_classes, n_trees=15, n_heads=5):
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.n_trees = n_trees
        self.n_heads = n_heads
        self.tree_depth = 3
        self.n_leaves = 2 ** self.tree_depth
        self.n_internal = self.n_leaves - 1
        
        # 计算每棵树的参数量
        self.params_per_tree = (
            self.n_internal * input_dim +  # split weights
            self.n_internal +               # split bias
            self.n_leaves * n_classes       # leaf logits
        )
        self.total_params = self.params_per_tree * n_trees + n_trees  # +tree weights
        
        hidden_dim = 128
        
        # 多头生成器
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, self.total_params)
            ) for _ in range(n_heads)
        ])
        
        # 初始化
        for head in self.heads:
            for layer in head:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=0.5)
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x_summary):
        # 获取所有头的预测
        all_params = [head(x_summary) for head in self.heads]
        # 平均
        params = torch.stack(all_params).mean(dim=0)
        # 限制参数范围
        params = torch.tanh(params) * 2.0
        return params


class SoftDecisionTreeEnsemble(nn.Module):
    """软决策树集成"""
    def __init__(self, input_dim, n_classes, n_trees=15, depth=3):
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.n_trees = n_trees
        self.depth = depth
        self.n_leaves = 2 ** depth
        self.n_internal = self.n_leaves - 1
    
    def forward(self, x, params):
        batch_size = x.shape[0]
        idx = 0
        
        all_tree_probs = []
        
        for t in range(self.n_trees):
            # 提取参数
            split_w_size = self.n_internal * self.input_dim
            split_b_size = self.n_internal
            leaf_size = self.n_leaves * self.n_classes
            
            split_weights = params[idx:idx+split_w_size].view(self.n_internal, self.input_dim)
            idx += split_w_size
            split_bias = params[idx:idx+split_b_size]
            idx += split_b_size
            leaf_logits = params[idx:idx+leaf_size].view(self.n_leaves, self.n_classes)
            idx += leaf_size
            
            # 计算分裂概率
            split_probs = torch.sigmoid(x @ split_weights.T + split_bias)
            
            # 计算到达每个叶子的概率
            leaf_probs = torch.ones(batch_size, self.n_leaves, device=x.device)
            for leaf_idx in range(self.n_leaves):
                path = []
                node = leaf_idx + self.n_internal
                for _ in range(self.depth):
                    parent = (node - 1) // 2
                    is_right = (node % 2 == 0)
                    path.append((parent, is_right))
                    node = parent
                path.reverse()
                
                for parent, is_right in path:
                    if is_right:
                        leaf_probs[:, leaf_idx] *= split_probs[:, parent]
                    else:
                        leaf_probs[:, leaf_idx] *= (1 - split_probs[:, parent])
            
            # 叶子节点的类别分布
            leaf_class_probs = torch.softmax(leaf_logits, dim=1)
            tree_pred = leaf_probs @ leaf_class_probs
            all_tree_probs.append(tree_pred)
        
        # 获取树权重
        tree_weights = torch.softmax(params[idx:idx+self.n_trees], dim=0)
        
        # 加权平均
        final_pred = torch.zeros(batch_size, self.n_classes, device=x.device)
        for t, (tree_pred, weight) in enumerate(zip(all_tree_probs, tree_weights)):
            final_pred += weight * tree_pred
        
        return final_pred


class VAEHyperNetFusion(nn.Module):
    def __init__(self, input_dim, n_classes, n_trees=15, n_heads=5):
        super().__init__()
        self.hypernet = MultiHeadHyperNet(input_dim, n_classes, n_trees, n_heads)
        self.tree_ensemble = SoftDecisionTreeEnsemble(input_dim, n_classes, n_trees)
    
    def forward(self, x, x_summary):
        params = self.hypernet(x_summary)
        return self.tree_ensemble(x, params)


# ========================= 数据加载 =========================

def load_dataset(dataset_id, data_dir):
    """加载数据集"""
    datasets = {
        0: ("0.prostate", "prostate.csv"),
        1: ("1.balloons", "adult+stretch.data"),
        2: ("2.lens", "lenses.data"),
        3: ("3.caesarian", "caesarian.csv.arff"),
        4: ("4.iris", "iris.data"),
        5: ("5.fertility", "fertility_Diagnosis.txt"),
        6: ("6.zoo", "zoo.data"),
        7: ("7.seeds", "seeds_dataset.txt"),
        8: ("8.haberman+s+survival", "haberman.data"),
        9: ("9.glass+identification", "glass.data"),
        10: ("10.yeast", "yeast.data"),
    }
    
    if dataset_id not in datasets:
        return None, None, None
    
    folder, filename = datasets[dataset_id]
    
    if dataset_id == 0:
        filepath = os.path.join(data_dir, "prostate.csv")
        if not os.path.exists(filepath):
            filepath = "/data2/image_identification/src/final/prostate.csv"
        df = pd.read_csv(filepath)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        return X, y, folder
    
    base_path = os.path.join(data_dir, folder)
    filepath = os.path.join(base_path, filename)
    
    if not os.path.exists(filepath):
        return None, None, folder
    
    try:
        if dataset_id == 1:  # balloons
            df = pd.read_csv(filepath, header=None)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
        elif dataset_id == 2:  # lenses
            df = pd.read_csv(filepath, delim_whitespace=True, header=None)
            X = df.iloc[:, 1:-1].values
            y = df.iloc[:, -1].values
        elif dataset_id == 3:  # caesarian (ARFF)
            with open(filepath, 'r') as f:
                lines = f.readlines()
            data_start = False
            data_rows = []
            for line in lines:
                if '@data' in line.lower():
                    data_start = True
                    continue
                if data_start and line.strip() and not line.startswith('%'):
                    data_rows.append(line.strip().split(','))
            df = pd.DataFrame(data_rows)
            X = df.iloc[:, :-1].values.astype(float)
            y = df.iloc[:, -1].values
        elif dataset_id == 4:  # iris
            df = pd.read_csv(filepath, header=None)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
        elif dataset_id == 5:  # fertility
            df = pd.read_csv(filepath, header=None)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
        elif dataset_id == 6:  # zoo
            df = pd.read_csv(filepath, header=None)
            X = df.iloc[:, 1:-1].values
            y = df.iloc[:, -1].values
        elif dataset_id == 7:  # seeds
            df = pd.read_csv(filepath, delim_whitespace=True, header=None)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
        elif dataset_id == 8:  # haberman
            df = pd.read_csv(filepath, header=None)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
        elif dataset_id == 9:  # glass
            df = pd.read_csv(filepath, header=None)
            X = df.iloc[:, 1:-1].values
            y = df.iloc[:, -1].values
        elif dataset_id == 10:  # yeast
            df = pd.read_csv(filepath, delim_whitespace=True, header=None)
            X = df.iloc[:, 1:-1].values
            y = df.iloc[:, -1].values
        
        # 编码
        for col in range(X.shape[1]):
            try:
                X[:, col] = X[:, col].astype(float)
            except:
                le = LabelEncoder()
                X[:, col] = le.fit_transform(X[:, col].astype(str))
        X = X.astype(float)
        
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
        
        return X, y, folder
        
    except Exception as e:
        return None, None, folder


# ========================= 单数据集评估函数 =========================

def evaluate_single_dataset(args):
    """在指定GPU上评估单个数据集"""
    dataset_id, gpu_id, seed, data_dir = args
    
    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 加载数据
    X, y, name = load_dataset(dataset_id, data_dir)
    if X is None:
        return {
            'dataset_id': dataset_id,
            'name': name,
            'seed': seed,
            'rf_acc': 0.0,
            'vhn_acc': 0.0,
            'error': 'Failed to load'
        }
    
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 动态参数
    n_heads = max(5, n_classes * 2)
    n_trees = max(15, n_classes * 3)
    
    # LPO p=2 评估
    pairs = list(combinations(range(n_samples), 2))
    
    rf_preds = []
    rf_trues = []
    vhn_preds = []
    vhn_trues = []
    
    for i, j in pairs:
        test_idx = [i, j]
        train_idx = [k for k in range(n_samples) if k not in test_idx]
        
        X_train, y_train = X_scaled[train_idx], y[train_idx]
        X_test, y_test = X_scaled[test_idx], y[test_idx]
        
        # RF
        rf = RandomForestClassifier(n_estimators=100, random_state=seed)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_preds.extend(rf_pred)
        rf_trues.extend(y_test)
        
        # VAE数据增强
        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.LongTensor(y_train).to(device)
        
        vae = VAE(n_features, latent_dim=8).to(device)
        vae_opt = torch.optim.Adam(vae.parameters(), lr=0.01)
        
        for _ in range(100):
            recon, mu, log_var = vae(X_train_t)
            recon_loss = nn.MSELoss()(recon, X_train_t)
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = recon_loss + 0.001 * kl_loss
            vae_opt.zero_grad()
            loss.backward()
            vae_opt.step()
        
        # 生成增强数据
        vae.eval()
        with torch.no_grad():
            aug_samples = []
            aug_labels = []
            for _ in range(5):
                for c in range(n_classes):
                    mask = y_train_t == c
                    if mask.sum() > 0:
                        x_c = X_train_t[mask]
                        mu, log_var = vae.encode(x_c)
                        z = vae.reparameterize(mu, log_var)
                        z_noise = z + torch.randn_like(z) * 0.1
                        x_aug = vae.decode(z_noise)
                        aug_samples.append(x_aug)
                        aug_labels.append(torch.full((x_aug.shape[0],), c, device=device))
            
            if aug_samples:
                X_aug = torch.cat([X_train_t] + aug_samples, dim=0)
                y_aug = torch.cat([y_train_t] + aug_labels, dim=0)
            else:
                X_aug = X_train_t
                y_aug = y_train_t
        
        # 训练HyperNet
        model = VAEHyperNetFusion(n_features, n_classes, n_trees=n_trees, n_heads=n_heads).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        
        x_summary = X_aug.mean(dim=0, keepdim=True)
        
        for epoch in range(200):
            model.train()
            probs = model(X_aug, x_summary)
            loss = nn.CrossEntropyLoss()(probs, y_aug)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        # 预测
        model.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test).to(device)
            probs = model(X_test_t, x_summary)
            vhn_pred = probs.argmax(dim=1).cpu().numpy()
        
        vhn_preds.extend(vhn_pred)
        vhn_trues.extend(y_test)
    
    rf_acc = np.mean(np.array(rf_preds) == np.array(rf_trues)) * 100
    vhn_acc = np.mean(np.array(vhn_preds) == np.array(vhn_trues)) * 100
    
    return {
        'dataset_id': dataset_id,
        'name': name,
        'seed': seed,
        'n_samples': n_samples,
        'n_features': n_features,
        'n_classes': n_classes,
        'rf_acc': rf_acc,
        'vhn_acc': vhn_acc,
        'error': None
    }


def run_on_gpu(gpu_id, tasks, data_dir):
    """在单个GPU上运行多个任务"""
    results = []
    for dataset_id, seed in tasks:
        result = evaluate_single_dataset((dataset_id, gpu_id, seed, data_dir))
        results.append(result)
        print(f"GPU {gpu_id} | Dataset {dataset_id} | Seed {seed} | RF: {result['rf_acc']:.1f}% | VHN: {result['vhn_acc']:.1f}%", flush=True)
    return results


def main():
    logger.info("=" * 60)
    logger.info("51_true_multigpu.py - 真正的多GPU并行")
    logger.info("=" * 60)
    
    data_dir = "/data2/image_identification/src/final/data"
    n_gpus = 8
    seeds = [42, 123, 456, 789, 1000]
    dataset_ids = list(range(11))
    
    # 创建所有任务 (dataset_id, seed)
    all_tasks = [(d, s) for d in dataset_ids for s in seeds]
    logger.info(f"总任务数: {len(all_tasks)} (11数据集 x 5 seeds)")
    
    # 分配任务到GPU
    gpu_tasks = {i: [] for i in range(n_gpus)}
    for idx, task in enumerate(all_tasks):
        gpu_id = idx % n_gpus
        gpu_tasks[gpu_id].append(task)
    
    for gpu_id, tasks in gpu_tasks.items():
        logger.info(f"GPU {gpu_id}: {len(tasks)} 个任务")
    
    # 启动多进程
    logger.info("\n开始并行执行...")
    start_time = time.time()
    
    all_results = []
    
    # 使用 spawn 启动方式
    mp.set_start_method('spawn', force=True)
    
    with ProcessPoolExecutor(max_workers=n_gpus) as executor:
        futures = {
            executor.submit(run_on_gpu, gpu_id, tasks, data_dir): gpu_id
            for gpu_id, tasks in gpu_tasks.items()
        }
        
        for future in as_completed(futures):
            gpu_id = futures[future]
            try:
                results = future.result()
                all_results.extend(results)
                logger.info(f"GPU {gpu_id} 完成，返回 {len(results)} 个结果")
            except Exception as e:
                logger.error(f"GPU {gpu_id} 出错: {e}")
    
    elapsed = time.time() - start_time
    logger.info(f"\n总耗时: {elapsed/60:.1f} 分钟")
    
    # 汇总结果
    logger.info("\n" + "=" * 80)
    logger.info("最终结果")
    logger.info("=" * 80)
    
    # 按数据集汇总
    for dataset_id in dataset_ids:
        ds_results = [r for r in all_results if r['dataset_id'] == dataset_id and r['error'] is None]
        if ds_results:
            rf_accs = [r['rf_acc'] for r in ds_results]
            vhn_accs = [r['vhn_acc'] for r in ds_results]
            name = ds_results[0]['name']
            n_samples = ds_results[0]['n_samples']
            n_features = ds_results[0]['n_features']
            n_classes = ds_results[0]['n_classes']
            
            rf_mean, rf_std = np.mean(rf_accs), np.std(rf_accs)
            vhn_mean, vhn_std = np.mean(vhn_accs), np.std(vhn_accs)
            
            winner = "VAE-HyperNet ✅" if vhn_mean > rf_mean else "RF"
            
            logger.info(f"{dataset_id}. {name}: {n_samples} samples, {n_features} features, {n_classes} classes")
            logger.info(f"   RF: {rf_mean:.1f}% ± {rf_std:.1f}% | VAE-HyperNet: {vhn_mean:.1f}% ± {vhn_std:.1f}% | Winner: {winner}")
        else:
            logger.info(f"{dataset_id}. ERROR or no results")
    
    logger.info("\n完成!")


if __name__ == "__main__":
    main()
