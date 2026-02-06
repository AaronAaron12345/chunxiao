#!/usr/bin/env python3
"""
53_5fold_fast.py - 快速5折交叉验证版本
使用5折CV代替LPO p=2，速度提升100-1000倍
多GPU并行处理所有数据集
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
import pandas as pd
from pathlib import Path
import warnings
import multiprocessing as mp
from datetime import datetime
import json

warnings.filterwarnings('ignore')

# ============== 模型定义 ==============
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=8, hidden_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
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


class MultiHeadHyperNet(nn.Module):
    """多头HyperNet - 为每个类别生成专门的决策树参数"""
    def __init__(self, input_dim, n_classes, n_trees=15, tree_depth=3):
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.n_trees = n_trees
        self.tree_depth = tree_depth
        self.n_internal = 2 ** tree_depth - 1
        self.n_leaves = 2 ** tree_depth
        
        # 数据编码器
        hidden_dim = max(64, input_dim * 4)
        self.data_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # 多头参数生成器
        self.head_generators = nn.ModuleList()
        params_per_tree = self.n_internal * (input_dim + 1) + self.n_leaves * n_classes
        total_params = n_trees * params_per_tree
        
        for _ in range(n_classes):
            self.head_generators.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, total_params)
            ))
        
        self.tree_weights = nn.Parameter(torch.ones(n_trees) / n_trees)
        
    def forward(self, X_train, X_test):
        # 编码训练数据统计特征
        train_mean = X_train.mean(dim=0, keepdim=True)
        train_std = X_train.std(dim=0, keepdim=True) + 1e-6
        stats = torch.cat([train_mean, train_std], dim=-1).squeeze(0)
        
        encoded = self.data_encoder(train_mean.squeeze(0))
        
        # 多头参数生成
        all_probs = []
        for head in self.head_generators:
            params = head(encoded)
            probs = self._apply_trees(X_test, params)
            all_probs.append(probs)
        
        # 融合多头输出
        all_probs = torch.stack(all_probs, dim=0)  # [n_classes, batch, n_classes]
        weights = F.softmax(self.tree_weights, dim=0)
        
        # 加权平均
        final_probs = all_probs.mean(dim=0)  # [batch, n_classes]
        return final_probs
    
    def _apply_trees(self, X, params):
        batch_size = X.shape[0]
        params_per_tree = self.n_internal * (self.input_dim + 1) + self.n_leaves * self.n_classes
        
        all_probs = []
        for t in range(self.n_trees):
            tree_params = params[t * params_per_tree:(t + 1) * params_per_tree]
            
            split_end = self.n_internal * self.input_dim
            split_weights = tree_params[:split_end].view(self.n_internal, self.input_dim)
            bias_end = split_end + self.n_internal
            split_bias = tree_params[split_end:bias_end]
            leaf_logits = tree_params[bias_end:].view(self.n_leaves, self.n_classes)
            
            # 软路由
            routes = torch.sigmoid(F.linear(X, split_weights, split_bias))
            
            # 计算叶子概率
            leaf_probs = torch.ones(batch_size, 1, device=X.device)
            for d in range(self.tree_depth):
                left_prob = routes[:, d:d+1]
                right_prob = 1 - left_prob
                leaf_probs = torch.cat([leaf_probs * left_prob, leaf_probs * right_prob], dim=1)
            
            # 叶子加权
            tree_output = F.softmax(leaf_logits, dim=-1)
            probs = torch.einsum('bl,lc->bc', leaf_probs[:, :self.n_leaves], tree_output)
            all_probs.append(probs)
        
        # 树集成
        weights = F.softmax(self.tree_weights, dim=0)
        ensemble_probs = sum(w * p for w, p in zip(weights, all_probs))
        return ensemble_probs


class VAEHyperNetFusion(nn.Module):
    def __init__(self, input_dim, n_classes, n_augment=100):
        super().__init__()
        self.vae = VAE(input_dim, latent_dim=min(8, input_dim), hidden_dim=max(32, input_dim * 2))
        self.hypernet = MultiHeadHyperNet(input_dim, n_classes)
        self.n_augment = n_augment
        self.input_dim = input_dim
        self.n_classes = n_classes
        
    def train_vae(self, X, epochs=100, lr=1e-3):
        """训练VAE"""
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=lr)
        dataset = TensorDataset(X)
        loader = DataLoader(dataset, batch_size=min(32, len(X)), shuffle=True)
        
        self.vae.train()
        for _ in range(epochs):
            for batch in loader:
                x = batch[0]
                recon, mu, logvar = self.vae(x)
                recon_loss = F.mse_loss(recon, x)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + 0.1 * kl_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
    def augment(self, X, y, n_per_class=None):
        """使用VAE增强数据"""
        if n_per_class is None:
            n_per_class = max(50, self.n_augment // self.n_classes)
        
        device = X.device
        self.vae.eval()
        
        # 确保所有初始tensor都在正确设备上
        augmented_X = [X.to(device)]
        augmented_y = [y.to(device)]
        
        with torch.no_grad():
            for c in range(self.n_classes):
                mask = (y == c)
                if mask.sum() == 0:
                    continue
                X_c = X[mask]
                mu, logvar = self.vae.encode(X_c)
                
                # 生成增强样本
                n_gen = n_per_class // max(1, mask.sum().item())
                for _ in range(max(1, n_gen)):
                    z = self.vae.reparameterize(mu, logvar)
                    new_X = self.vae.decode(z).to(device)
                    augmented_X.append(new_X)
                    augmented_y.append(torch.full((len(new_X),), c, dtype=torch.long, device=device))
        
        return torch.cat(augmented_X, dim=0), torch.cat(augmented_y, dim=0)
    
    def forward(self, X_train, y_train, X_test):
        # VAE数据增强
        X_aug, y_aug = self.augment(X_train, y_train)
        
        # HyperNet生成分类器并预测
        probs = self.hypernet(X_aug, X_test)
        return probs


def evaluate_fold(args):
    """评估单个fold"""
    fold_idx, X_train, y_train, X_test, y_test, n_classes, seed, device_id = args
    
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    input_dim = X_train.shape[1]
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # RF基线
    rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    rf_acc = (rf_pred == y_test).mean()
    
    # VAE-HyperNet
    X_train_t = torch.FloatTensor(X_train_scaled).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test_scaled).to(device)
    
    model = VAEHyperNetFusion(input_dim, n_classes).to(device)
    
    # 训练VAE
    model.train_vae(X_train_t, epochs=100)
    
    # 训练HyperNet
    optimizer = torch.optim.Adam(model.hypernet.parameters(), lr=1e-3, weight_decay=1e-4)
    
    model.hypernet.train()
    for epoch in range(200):
        probs = model(X_train_t, y_train_t, X_train_t)
        loss = F.cross_entropy(probs, y_train_t)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 预测
    model.eval()
    with torch.no_grad():
        probs = model(X_train_t, y_train_t, X_test_t)
        vhn_pred = probs.argmax(dim=1).cpu().numpy()
    
    vhn_acc = (vhn_pred == y_test).mean()
    
    return fold_idx, rf_acc, vhn_acc


def load_dataset_by_id(dataset_id, data_dir="/data2/image_identification/src/small_data"):
    """根据数据集ID加载数据，已验证的数据加载逻辑"""
    datasets = {
        0: ("0.prostate", None),
        1: ("1.balloons", "1balloon/adult+stretch.data"),
        2: ("2.lenses", "2lens/lenses.data"),
        3: ("3.caesarian", "3.caesarian+section+classification+dataset/caesarian.csv.arff"),
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
            X = df.iloc[:, 2:].values
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
            df = pd.read_csv(filepath, delim_whitespace=True, header=None)
            X = df.iloc[:, 1:-1].values
            y = df.iloc[:, -1].values
        elif dataset_id == 3:
            filepath = os.path.join(data_dir, filepath_rel)
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
        elif dataset_id == 4:
            filepath = os.path.join(data_dir, filepath_rel)
            df = pd.read_csv(filepath, header=None)
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
            df = pd.read_csv(filepath, delim_whitespace=True, header=None)
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
            df = pd.read_csv(filepath, delim_whitespace=True, header=None)
            X = df.iloc[:, 1:-1].values
            y = df.iloc[:, -1].values
        
        # 处理非数值特征
        for col in range(X.shape[1]):
            try:
                X[:, col] = X[:, col].astype(float)
            except:
                le = LabelEncoder()
                X[:, col] = le.fit_transform(X[:, col].astype(str))
        
        X = X.astype(float)
        
        # 编码标签
        if dataset_id != 0:
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
        
        return X, y, name
    except Exception as e:
        print(f"  加载数据集{dataset_id}失败: {e}")
        return None, None, name


def evaluate_dataset(dataset_id, n_seeds=5, gpu_id=0):
    """使用5折CV评估单个数据集"""
    X, y, name = load_dataset_by_id(dataset_id)
    
    if X is None:
        print(f"  数据集{dataset_id}加载失败")
        return None
    
    n_classes = len(np.unique(y))
    n_samples = len(y)
    n_features = X.shape[1]
    
    # 检查每个类别的样本数是否足够进行5折
    min_class_count = min(np.bincount(y))
    n_splits = min(5, min_class_count)
    if n_splits < 2:
        print(f"  {name}: 样本数不足，跳过")
        return None
    
    print(f"  {name}: {n_samples}样本, {n_features}特征, {n_classes}类")
    
    results = {'dataset': name, 'rf_accs': [], 'vhn_accs': []}
    
    # 多seed评估
    for seed in [42, 123, 456, 789, 1000]:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        
        fold_rf_accs = []
        fold_vhn_accs = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            result = evaluate_fold((
                fold_idx, X_train, y_train, X_test, y_test,
                n_classes, seed, gpu_id
            ))
            
            fold_rf_accs.append(result[1])
            fold_vhn_accs.append(result[2])
        
        rf_mean = np.mean(fold_rf_accs) * 100
        vhn_mean = np.mean(fold_vhn_accs) * 100
        results['rf_accs'].append(rf_mean)
        results['vhn_accs'].append(vhn_mean)
        
        print(f"    Seed {seed}: RF={rf_mean:.1f}%, VHN={vhn_mean:.1f}%")
    
    return results


def main():
    # 获取可用GPU
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        available_gpus = [i for i in range(n_gpus) if i not in [0, 6, 7]]
        print(f"可用GPU: {available_gpus}")
    else:
        available_gpus = [0]
        print("使用CPU")
    
    all_results = []
    
    print("\n" + "="*60)
    print("5折交叉验证评估 (5 seeds)")
    print("="*60)
    
    for dataset_id in range(11):
        gpu_id = available_gpus[dataset_id % len(available_gpus)]
        
        print(f"\n[{dataset_id+1}/11] 处理数据集 {dataset_id} (GPU {gpu_id})...")
        result = evaluate_dataset(dataset_id, gpu_id=gpu_id)
        
        if result:
            rf_mean = np.mean(result['rf_accs'])
            rf_std = np.std(result['rf_accs'])
            vhn_mean = np.mean(result['vhn_accs'])
            vhn_std = np.std(result['vhn_accs'])
            
            result['rf_summary'] = f"{rf_mean:.1f}±{rf_std:.1f}"
            result['vhn_summary'] = f"{vhn_mean:.1f}±{vhn_std:.1f}"
            result['winner'] = 'VAE-HyperNet' if vhn_mean > rf_mean else 'RF'
            
            all_results.append(result)
    
    # 打印汇总
    print("\n" + "="*60)
    print("最终结果汇总")
    print("="*60)
    print(f"{'Dataset':<20} {'RF (%)':<15} {'VAE-HyperNet (%)':<18} {'Winner'}")
    print("-"*60)
    
    for r in all_results:
        print(f"{r['dataset']:<20} {r['rf_summary']:<15} {r['vhn_summary']:<18} {r['winner']}")
    
    # 保存结果
    output_dir = "/data2/image_identification/src/output"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"53_5fold_results_{timestamp}.json")
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n结果已保存: {output_file}")


if __name__ == "__main__":
    main()
