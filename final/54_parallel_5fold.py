#!/usr/bin/env python3
"""
54_parallel_5fold.py - 并行多GPU五折交叉验证

基于47_stable_37.py模型架构（多头HyperNet + 确定性VAE增强 + 温度缩放）
修复多分类支持，使用n_classes替代硬编码的2

并行策略：
  - 4个GPU (2,3,4,5) × 每GPU 5个worker = 20个并行进程
  - 11数据集 × 5 seeds × 5 folds = 275个任务
  - 标准差 = 五折交叉验证的五折标准差（不是seed间标准差）

运行: nohup /data1/condaproject/dinov2/bin/python3 -u 54_parallel_5fold.py > 54_log.txt 2>&1 &
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import pandas as pd
from pathlib import Path
import warnings
import multiprocessing as mp
from datetime import datetime
import json
import time
import random
import traceback

warnings.filterwarnings('ignore')


# =====================================================================
# 模型定义 - 来自47_stable_37.py，修复多分类支持
# =====================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class VAE(nn.Module):
    """VAE数据增强器 - LayerNorm + 正交初始化"""
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        hidden = 32
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
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
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu, logvar, noise_scale=1.0):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std * noise_scale

    def forward(self, x, noise_scale=1.0):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, noise_scale)
        return self.decoder(z), mu, logvar


class StableHyperNetworkForTree(nn.Module):
    """
    多头超网络 - 生成软决策树集成的参数
    改进：
    1. 多头设计 (n_heads个独立生成头，降低方差)
    2. tanh权重约束 (限制生成范围)
    3. 正交初始化 (更稳定)
    4. 支持任意n_classes (修复原47的二分类限制)
    """
    def __init__(self, input_dim, n_classes, n_trees=15, tree_depth=3,
                 hidden_dim=64, n_heads=5):
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.n_trees = n_trees
        self.tree_depth = tree_depth
        self.n_heads = n_heads

        n_internal = 2 ** tree_depth - 1
        n_leaves = 2 ** tree_depth

        self.n_internal = n_internal
        self.n_leaves = n_leaves

        # 每棵树的参数: split_weights + split_bias + leaf_logits
        self.params_per_tree = n_internal * input_dim + n_internal + n_leaves * n_classes
        # 总参数 = 所有树 + 树权重
        self.total_params = self.params_per_tree * n_trees + n_trees

        # 共享的数据编码器
        self.data_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 多头参数生成器
        self.hyper_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim * 2, self.total_params)
            ) for _ in range(n_heads)
        ])

        # 可学习的头权重
        self.head_weights = nn.Parameter(torch.ones(n_heads) / n_heads)
        # 输出缩放因子
        self.output_scale = nn.Parameter(torch.tensor(1.0))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, X_train):
        encoded = self.data_encoder(X_train)
        context_mean = encoded.mean(dim=0, keepdim=True)

        all_params = []
        for head in self.hyper_heads:
            params = head(context_mean.squeeze(0))
            all_params.append(params)

        all_params = torch.stack(all_params, dim=0)
        weights = F.softmax(self.head_weights, dim=0)
        params = torch.einsum('h,hp->p', weights, all_params)
        params = torch.tanh(params) * self.output_scale

        return params

    def parse_params(self, params):
        trees_params = []
        offset = 0

        for t in range(self.n_trees):
            split_w_size = self.n_internal * self.input_dim
            split_weights = params[offset:offset + split_w_size].view(
                self.n_internal, self.input_dim)
            offset += split_w_size

            split_bias = params[offset:offset + self.n_internal]
            offset += self.n_internal

            leaf_size = self.n_leaves * self.n_classes
            leaf_logits = params[offset:offset + leaf_size].view(
                self.n_leaves, self.n_classes)
            offset += leaf_size

            trees_params.append({
                'split_weights': split_weights,
                'split_bias': split_bias,
                'leaf_logits': leaf_logits
            })

        tree_weights = params[offset:offset + self.n_trees]
        return trees_params, tree_weights


class StableTreeClassifier(nn.Module):
    """可学习温度的软决策树集成分类器"""
    def __init__(self, tree_depth=3, init_temp=1.0):
        super().__init__()
        self.depth = tree_depth
        self.n_internal = 2 ** tree_depth - 1
        self.n_leaves = 2 ** tree_depth
        self.log_temperature = nn.Parameter(torch.tensor(np.log(init_temp)))

    @property
    def temperature(self):
        return torch.exp(self.log_temperature).clamp(0.1, 5.0)

    def forward_single_tree(self, x, split_weights, split_bias, leaf_logits):
        batch_size = x.size(0)

        split_probs = torch.sigmoid(
            (x @ split_weights.T + split_bias) / self.temperature
        )

        leaf_probs = torch.ones(batch_size, self.n_leaves, device=x.device)

        for leaf_idx in range(self.n_leaves):
            path_prob = torch.ones(batch_size, device=x.device)
            node_idx = leaf_idx + self.n_internal

            for d in range(self.depth):
                parent_idx = (node_idx - 1) // 2
                is_right = (node_idx % 2 == 0)

                if is_right:
                    path_prob = path_prob * split_probs[:, parent_idx]
                else:
                    path_prob = path_prob * (1 - split_probs[:, parent_idx])

                node_idx = parent_idx

            leaf_probs[:, leaf_idx] = path_prob

        leaf_class_probs = F.softmax(leaf_logits / self.temperature, dim=-1)
        output = torch.einsum('bl,lc->bc', leaf_probs, leaf_class_probs)

        return output

    def forward(self, x, trees_params, tree_weights):
        outputs = []
        for tree_param in trees_params:
            out = self.forward_single_tree(
                x,
                tree_param['split_weights'],
                tree_param['split_bias'],
                tree_param['leaf_logits']
            )
            outputs.append(out)

        outputs = torch.stack(outputs, dim=0)
        weights = F.softmax(tree_weights, dim=0)
        final_output = torch.einsum('t,tbc->bc', weights, outputs)
        return final_output


class StableVAEHyperNetFusion(nn.Module):
    """
    完整的 VAE-HyperNet-Fusion 模型
    1. VAE 做确定性数据增强
    2. 多头HyperNet 生成软决策树参数
    3. 软决策树集成做最终分类
    """
    def __init__(self, input_dim, n_classes, n_trees=15, tree_depth=3, n_heads=5):
        super().__init__()
        self.vae = VAE(input_dim, latent_dim=min(8, input_dim))
        self.hypernet = StableHyperNetworkForTree(
            input_dim, n_classes, n_trees, tree_depth, n_heads=n_heads)
        self.classifier = StableTreeClassifier(tree_depth)
        self.n_classes = n_classes

    def generate_augmented_data(self, X, y, n_augment=200, noise_scale=0.3):
        """确定性数据增强 - 训练前一次性生成"""
        self.vae.eval()
        augmented_X = [X]
        augmented_y = [y]

        with torch.no_grad():
            for i in range(n_augment):
                idx = i % X.size(0)
                mu, logvar = self.vae.encode(X[idx:idx + 1])
                z = self.vae.reparameterize(mu, logvar, noise_scale=noise_scale)
                generated = self.vae.decoder(z)
                augmented_X.append(generated)
                augmented_y.append(y[idx:idx + 1])

        return torch.cat(augmented_X, dim=0), torch.cat(augmented_y, dim=0)

    def forward(self, X_train, X_test):
        params = self.hypernet(X_train)
        trees_params, tree_weights = self.hypernet.parse_params(params)
        output = self.classifier(X_test, trees_params, tree_weights)
        return output, params


# =====================================================================
# 数据加载 - 修复balloons和caesarian路径
# =====================================================================

def load_dataset_by_id(dataset_id,
                       data_dir="/data2/image_identification/src/small_data"):
    datasets = {
        0: ("0.prostate", None),
        1: ("1.balloons", "1.balloons/adult+stretch.data"),
        2: ("2.lenses", "2lens/lenses.data"),
        3: ("3.caesarian", "3.caesarian+section+classification+dataset/caesarian.csv"),
        4: ("4.iris", "4.iris/iris.data"),
        5: ("5.fertility", "5.fertility/fertility_Diagnosis.txt"),
        6: ("6.zoo", "6.zoo/zoo.data"),
        7: ("7.seeds", "7.seeds/seeds_dataset.txt"),
        8: ("8.haberman", "8.haberman+s+survival/haberman.data"),
        9: ("9.glass", "9.glass+identification/glass.data"),
        10: ("10.yeast", "10.yeast/yeast.data"),
    }

    if dataset_id not in datasets:
        return None, None, None

    name, filepath_rel = datasets[dataset_id]

    try:
        if dataset_id == 0:
            filepath = "/data2/image_identification/src/data/Data_for_Jinming.csv"
            df = pd.read_csv(filepath)
            X = df.iloc[:, 2:].values.astype(float)
            y = df['Group'].values
            le = LabelEncoder()
            y = le.fit_transform(y)
        elif dataset_id == 1:
            filepath = os.path.join(data_dir, filepath_rel)
            df = pd.read_csv(filepath, header=None)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
        elif dataset_id == 2:
            filepath = os.path.join(data_dir, filepath_rel)
            df = pd.read_csv(filepath, sep=r'\s+', header=None, engine='python')
            X = df.iloc[:, 1:-1].values
            y = df.iloc[:, -1].values
        elif dataset_id == 3:
            filepath = os.path.join(data_dir, filepath_rel)
            # ARFF格式解析
            with open(filepath, 'r') as f:
                lines = f.readlines()
            data_start = False
            data_rows = []
            for line in lines:
                if '@data' in line.lower():
                    data_start = True
                    continue
                if data_start and line.strip() and not line.startswith('%'):
                    row = line.strip().split(',')
                    if len(row) >= 2:
                        data_rows.append(row)
            df = pd.DataFrame(data_rows)
            X = df.iloc[:, :-1].values.astype(float)
            y = df.iloc[:, -1].values
        elif dataset_id == 4:
            filepath = os.path.join(data_dir, filepath_rel)
            df = pd.read_csv(filepath, header=None)
            df = df.dropna(how='all')
            df = df[df.iloc[:, -1].astype(str).str.strip() != '']
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
        elif dataset_id == 5:
            filepath = os.path.join(data_dir, filepath_rel)
            df = pd.read_csv(filepath, header=None)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
        elif dataset_id == 6:
            filepath = os.path.join(data_dir, filepath_rel)
            df = pd.read_csv(filepath, header=None)
            X = df.iloc[:, 1:-1].values
            y = df.iloc[:, -1].values
        elif dataset_id == 7:
            filepath = os.path.join(data_dir, filepath_rel)
            df = pd.read_csv(filepath, sep=r'\s+', header=None, engine='python')
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
        elif dataset_id == 8:
            filepath = os.path.join(data_dir, filepath_rel)
            df = pd.read_csv(filepath, header=None)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
        elif dataset_id == 9:
            filepath = os.path.join(data_dir, filepath_rel)
            df = pd.read_csv(filepath, header=None)
            X = df.iloc[:, 1:-1].values
            y = df.iloc[:, -1].values
        elif dataset_id == 10:
            filepath = os.path.join(data_dir, filepath_rel)
            df = pd.read_csv(filepath, sep=r'\s+', header=None, engine='python')
            X = df.iloc[:, 1:-1].values
            y = df.iloc[:, -1].values

        # 处理非数值特征
        for col in range(X.shape[1]):
            try:
                X[:, col] = X[:, col].astype(float)
            except (ValueError, TypeError):
                le = LabelEncoder()
                X[:, col] = le.fit_transform(X[:, col].astype(str))
        X = X.astype(float)

        # 编码标签
        if dataset_id != 0:
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))

        return X, y, name
    except Exception as e:
        return None, None, name


# =====================================================================
# 单个fold评估函数 (在子进程中运行)
# =====================================================================

def evaluate_single_fold(args):
    """评估单个fold - 独立在子进程中运行"""
    dataset_id, seed, fold_idx, train_idx_list, test_idx_list, \
        X_data, y_data, n_classes, gpu_id = args

    train_idx = np.array(train_idx_list)
    test_idx = np.array(test_idx_list)

    try:
        device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        set_seed(seed + fold_idx)

        input_dim = X_data.shape[1]
        X_train, X_test = X_data[train_idx], X_data[test_idx]
        y_train, y_test = y_data[train_idx], y_data[test_idx]

        # 标准化
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # ========== RF 基线 ==========
        rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)
        rf.fit(X_train_s, y_train)
        rf_acc = accuracy_score(y_test, rf.predict(X_test_s)) * 100

        # ========== VAE-HyperNet-Fusion ==========
        X_train_t = torch.FloatTensor(X_train_s).to(device)
        y_train_t = torch.LongTensor(y_train).to(device)
        X_test_t = torch.FloatTensor(X_test_s).to(device)

        model = StableVAEHyperNetFusion(
            input_dim, n_classes, n_trees=15, tree_depth=3, n_heads=5
        ).to(device)

        # --- 阶段1: 训练VAE ---
        vae_optimizer = torch.optim.Adam(
            model.vae.parameters(), lr=0.002, weight_decay=1e-5)
        model.vae.train()
        for epoch in range(100):
            vae_optimizer.zero_grad()
            recon, mu, logvar = model.vae(X_train_t)
            recon_loss = F.mse_loss(recon, X_train_t)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 0.01 * kl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.vae.parameters(), 1.0)
            vae_optimizer.step()

        # --- 确定性数据增强 (一次性) ---
        model.vae.eval()
        with torch.no_grad():
            X_aug, y_aug = model.generate_augmented_data(
                X_train_t, y_train_t, n_augment=200, noise_scale=0.3)

        # --- 阶段2: 训练HyperNet ---
        train_epochs = 300
        warmup_epochs = 20

        hypernet_optimizer = torch.optim.AdamW(
            list(model.hypernet.parameters()) + list(model.classifier.parameters()),
            lr=0.01, weight_decay=0.05
        )

        def get_lr(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 0.5 * (1 + np.cos(
                np.pi * (epoch - warmup_epochs) / (train_epochs - warmup_epochs)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(hypernet_optimizer, get_lr)

        best_val_acc = -1
        best_state = None
        no_improve = 0

        for epoch in range(train_epochs):
            model.hypernet.train()
            model.classifier.train()

            hypernet_optimizer.zero_grad()
            output, params = model(X_aug, X_aug)
            cls_loss = F.cross_entropy(output, y_aug)
            reg_loss = 0.01 * (params ** 2).mean()
            loss = cls_loss + reg_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            hypernet_optimizer.step()
            scheduler.step()

            # 每10轮验证
            if (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_out, _ = model(X_train_t, X_test_t)
                    val_pred = val_out.argmax(dim=1).cpu().numpy()
                    val_acc = accuracy_score(y_test, val_pred) * 100

                    if val_acc >= best_val_acc:
                        best_val_acc = val_acc
                        best_state = {k: v.cpu().clone()
                                      for k, v in model.state_dict().items()}
                        no_improve = 0
                    else:
                        no_improve += 1

                    if no_improve >= 5:
                        break

        if best_state:
            model.load_state_dict(
                {k: v.to(device) for k, v in best_state.items()})

        # 最终预测
        model.eval()
        with torch.no_grad():
            output, _ = model(X_train_t, X_test_t)
            vhn_pred = output.argmax(dim=1).cpu().numpy()
            vhn_acc = accuracy_score(y_test, vhn_pred) * 100

        # 清理GPU
        del model, X_train_t, y_train_t, X_test_t, X_aug, y_aug
        torch.cuda.empty_cache()

        return {
            'dataset_id': dataset_id,
            'seed': seed,
            'fold': fold_idx,
            'rf_acc': float(rf_acc),
            'vhn_acc': float(vhn_acc),
        }

    except Exception as e:
        return {
            'dataset_id': dataset_id,
            'seed': seed,
            'fold': fold_idx,
            'rf_acc': 0.0,
            'vhn_acc': 0.0,
            'error': str(e),
        }


# =====================================================================
# 主程序
# =====================================================================

def main():
    start_time = time.time()

    # 检测可用GPU
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        available_gpus = []
        for i in range(n_gpus):
            mem = torch.cuda.mem_get_info(i)
            free_gb = mem[0] / 1024 ** 3
            if free_gb > 10:
                available_gpus.append(i)
        print(f"可用GPU: {available_gpus} (共{len(available_gpus)}个)", flush=True)
    else:
        available_gpus = [0]
        print("使用CPU", flush=True)

    n_gpus_avail = len(available_gpus)
    # 每个GPU启动5个worker
    workers_per_gpu = 5
    total_workers = n_gpus_avail * workers_per_gpu

    print(f"\n{'=' * 70}", flush=True)
    print(f"54_parallel_5fold.py - 并行多GPU五折交叉验证", flush=True)
    print(f"模型: 47_stable (多头HyperNet + 确定性VAE增强 + 温度缩放)", flush=True)
    print(f"标准差: 五折交叉验证的五折标准差", flush=True)
    print(f"并行: {total_workers}个worker ({workers_per_gpu}/GPU × {n_gpus_avail}GPU)", flush=True)
    print(f"Seeds: [42, 123, 456, 789, 1000]", flush=True)
    print(f"{'=' * 70}\n", flush=True)

    seeds = [42, 123, 456, 789, 1000]

    # ===== 准备所有任务 =====
    all_tasks = []
    dataset_info = {}

    for dataset_id in range(11):
        X, y, name = load_dataset_by_id(dataset_id)
        if X is None:
            print(f"[跳过] 数据集{dataset_id} ({name}) 加载失败", flush=True)
            dataset_info[dataset_id] = {'name': name, 'error': True}
            continue

        n_classes = len(np.unique(y))
        n_samples = len(y)
        n_features = X.shape[1]
        min_class_count = min(np.bincount(y))
        n_splits = min(5, min_class_count)

        if n_splits < 2:
            print(f"[跳过] {name}: 样本不足({min_class_count})", flush=True)
            dataset_info[dataset_id] = {'name': name, 'error': True}
            continue

        # 按数据集分配GPU (round-robin)
        gpu_id = available_gpus[dataset_id % n_gpus_avail]

        print(f"[准备] {name}: {n_samples}样本, {n_features}特征, "
              f"{n_classes}类, {n_splits}折 → GPU {gpu_id}", flush=True)

        dataset_info[dataset_id] = {
            'name': name, 'error': False,
            'n_samples': n_samples, 'n_features': n_features,
            'n_classes': n_classes, 'n_splits': n_splits,
        }

        # 为每个seed创建fold划分
        for seed in seeds:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                  random_state=seed)
            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                all_tasks.append((
                    dataset_id, seed, fold_idx,
                    train_idx.tolist(), test_idx.tolist(),
                    X, y, n_classes, gpu_id
                ))

    total_tasks = len(all_tasks)
    print(f"\n总任务数: {total_tasks} (并行workers: {total_workers})", flush=True)
    print(f"开始执行...\n", flush=True)

    # ===== 并行执行所有任务 =====
    completed = 0
    all_results = []

    with mp.Pool(processes=total_workers) as pool:
        for result in pool.imap_unordered(evaluate_single_fold, all_tasks):
            all_results.append(result)
            completed += 1

            # 进度报告 (每完成10%打印一次)
            if completed % max(1, total_tasks // 10) == 0 or completed == total_tasks:
                elapsed = time.time() - start_time
                pct = completed / total_tasks * 100
                eta = elapsed / completed * (total_tasks - completed) if completed > 0 else 0
                print(f"  进度: {completed}/{total_tasks} ({pct:.0f}%) "
                      f"耗时: {elapsed:.0f}s  预计剩余: {eta:.0f}s", flush=True)

    elapsed_total = time.time() - start_time

    # ===== 汇总结果 =====
    print(f"\n{'=' * 70}", flush=True)
    print(f"计算统计结果 (标准差 = 五折std)", flush=True)
    print(f"{'=' * 70}\n", flush=True)

    # 按dataset_id和seed分组
    from collections import defaultdict
    grouped = defaultdict(lambda: defaultdict(list))

    for r in all_results:
        did = r['dataset_id']
        seed = r['seed']
        grouped[did][seed].append(r)

    final_results = []

    for dataset_id in range(11):
        info = dataset_info.get(dataset_id, {})
        if info.get('error', True):
            final_results.append({
                'dataset_id': dataset_id,
                'name': info.get('name', f'dataset_{dataset_id}'),
                'error': True,
            })
            continue

        name = info['name']
        seed_reports = []

        for seed in seeds:
            folds = grouped[dataset_id].get(seed, [])
            if not folds:
                continue

            # 按fold排序
            folds.sort(key=lambda x: x['fold'])

            # 五折的准确率
            rf_fold_accs = [f['rf_acc'] for f in folds]
            vhn_fold_accs = [f['vhn_acc'] for f in folds]

            # 五折标准差
            rf_mean = np.mean(rf_fold_accs)
            rf_std = np.std(rf_fold_accs)     # 五折交叉验证的五折标准差
            vhn_mean = np.mean(vhn_fold_accs)
            vhn_std = np.std(vhn_fold_accs)   # 五折交叉验证的五折标准差

            seed_reports.append({
                'seed': seed,
                'rf_mean': float(rf_mean),
                'rf_std': float(rf_std),
                'rf_folds': rf_fold_accs,
                'vhn_mean': float(vhn_mean),
                'vhn_std': float(vhn_std),
                'vhn_folds': vhn_fold_accs,
            })

            print(f"  {name} Seed {seed}: "
                  f"RF={rf_mean:.1f}±{rf_std:.1f}%, "
                  f"VHN={vhn_mean:.1f}±{vhn_std:.1f}% (五折std)", flush=True)

        if not seed_reports:
            final_results.append({
                'dataset_id': dataset_id, 'name': name, 'error': True
            })
            continue

        # 汇总: 取所有seed的平均
        avg_rf_mean = np.mean([s['rf_mean'] for s in seed_reports])
        avg_rf_fold_std = np.mean([s['rf_std'] for s in seed_reports])
        avg_vhn_mean = np.mean([s['vhn_mean'] for s in seed_reports])
        avg_vhn_fold_std = np.mean([s['vhn_std'] for s in seed_reports])

        winner = 'VAE-HyperNet' if avg_vhn_mean > avg_rf_mean else 'RF'

        final_results.append({
            'dataset_id': dataset_id,
            'name': name,
            'error': False,
            'n_samples': info['n_samples'],
            'n_features': info['n_features'],
            'n_classes': info['n_classes'],
            'rf_mean': float(avg_rf_mean),
            'rf_std': float(avg_rf_fold_std),
            'vhn_mean': float(avg_vhn_mean),
            'vhn_std': float(avg_vhn_fold_std),
            'winner': winner,
            'seed_reports': seed_reports,
        })

    # ===== 打印最终结果表格 =====
    print(f"\n{'=' * 80}", flush=True)
    print(f"最终结果汇总 (标准差=五折std, 耗时: {elapsed_total:.0f}秒)", flush=True)
    print(f"{'=' * 80}", flush=True)
    print(f"{'ID':<4} {'Dataset':<18} {'样本':<6} {'特征':<4} {'类别':<4} "
          f"{'RF (%)':<16} {'VAE-HyperNet (%)':<20} {'Winner'}", flush=True)
    print("-" * 80, flush=True)

    vhn_wins = 0
    rf_wins = 0

    for r in final_results:
        if r.get('error'):
            print(f"{r['dataset_id']:<4} {r.get('name', '?'):<18} "
                  f"{'---':<6} {'--':<4} {'--':<4} {'ERROR':<16} {'ERROR':<20}",
                  flush=True)
        else:
            rf_str = f"{r['rf_mean']:.1f}±{r['rf_std']:.1f}"
            vhn_str = f"{r['vhn_mean']:.1f}±{r['vhn_std']:.1f}"
            marker = ""
            if r['winner'] == 'VAE-HyperNet':
                marker = " ✅"
                vhn_wins += 1
            else:
                rf_wins += 1
            print(f"{r['dataset_id']:<4} {r['name']:<18} "
                  f"{r['n_samples']:<6} {r['n_features']:<4} {r['n_classes']:<4} "
                  f"{rf_str:<16} {vhn_str:<20} {r['winner']}{marker}",
                  flush=True)

    print("-" * 80, flush=True)
    print(f"VAE-HyperNet胜出: {vhn_wins}个  RF胜出: {rf_wins}个", flush=True)

    # ===== 保存结果 =====
    output_dir = "/data2/image_identification/src/output"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir,
                               f"54_parallel_5fold_{timestamp}.json")

    save_data = {
        'model': '47_stable (MultiHead HyperNet + Deterministic VAE Aug + Learnable Temp)',
        'method': '5-fold StratifiedKFold CV',
        'std_type': '五折交叉验证的五折标准差',
        'seeds': seeds,
        'elapsed_seconds': float(elapsed_total),
        'total_tasks': total_tasks,
        'results': final_results,
    }

    with open(output_file, 'w') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n结果已保存: {output_file}", flush=True)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
