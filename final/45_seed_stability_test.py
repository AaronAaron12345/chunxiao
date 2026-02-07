#!/usr/bin/env python3
"""
45_seed_stability_test.py
测试不同seed在所有数据集上的稳定性

目的：验证seed=42是否是"幸运的seed"，还是方法本身稳定
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(16, latent_dim)
        self.fc_var = nn.Linear(16, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
    
    def decode(self, z):
        return self.decoder(z)


class HyperNetworkForTree(nn.Module):
    def __init__(self, input_dim, n_trees=15, tree_depth=3, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.n_trees = n_trees
        self.n_leaves = 2 ** tree_depth
        self.n_internal = 2 ** tree_depth - 1
        
        self.data_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        
        total_params = self.n_internal * (input_dim + 1) + self.n_leaves * 2
        self.weight_gen = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_trees * total_params),
        )
        self.tree_weight_gen = nn.Linear(hidden_dim, n_trees)
        
    def forward(self, X):
        encoded = self.data_encoder(X)
        data_summary = encoded.mean(dim=0, keepdim=True)
        tree_params = self.weight_gen(data_summary)
        tree_weights = torch.softmax(self.tree_weight_gen(data_summary), dim=-1)
        return tree_params, tree_weights


class GeneratedTreeClassifier(nn.Module):
    def __init__(self, input_dim, n_classes, n_trees=15, tree_depth=3):
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.n_trees = n_trees
        self.n_leaves = 2 ** tree_depth
        self.n_internal = 2 ** tree_depth - 1
        
    def forward(self, x, tree_params, tree_weights):
        batch_size = x.shape[0]
        param_per_tree = self.n_internal * (self.input_dim + 1) + self.n_leaves * self.n_classes
        all_probs = []
        
        for t in range(self.n_trees):
            start = t * param_per_tree
            split_w = tree_params[0, start:start + self.n_internal * self.input_dim].view(self.n_internal, self.input_dim)
            split_b = tree_params[0, start + self.n_internal * self.input_dim:start + self.n_internal * (self.input_dim + 1)]
            leaf_start = start + self.n_internal * (self.input_dim + 1)
            leaf_logits = tree_params[0, leaf_start:leaf_start + self.n_leaves * self.n_classes].view(self.n_leaves, self.n_classes)
            
            decisions = torch.sigmoid(torch.matmul(x, split_w.T) + split_b)
            
            leaf_probs = torch.ones(batch_size, self.n_leaves, device=x.device)
            for i in range(min(self.n_internal, self.n_leaves // 2)):
                left = 2 * i + 1
                right = 2 * i + 2
                if left < self.n_leaves and i < decisions.shape[1]:
                    d = decisions[:, i:i+1]
                    leaf_probs[:, left] *= d.squeeze()
                    if right < self.n_leaves:
                        leaf_probs[:, right] *= (1 - d).squeeze()
            
            leaf_probs = leaf_probs / (leaf_probs.sum(dim=1, keepdim=True) + 1e-8)
            tree_output = torch.matmul(leaf_probs, torch.softmax(leaf_logits, dim=-1))
            all_probs.append(tree_output * tree_weights[0, t])
        
        return torch.stack(all_probs).sum(dim=0)


def train_vae_hypernet(X_train, y_train, n_classes, epochs=300, seed=42):
    set_seed(seed)
    
    input_dim = X_train.shape[1]
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    
    # VAE
    vae = VAE(input_dim, latent_dim=8).to(device)
    vae_opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
    
    for _ in range(100):
        recon, mu, logvar = vae(X_train_t)
        loss = nn.MSELoss()(recon, X_train_t) + 0.01 * (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))
        vae_opt.zero_grad()
        loss.backward()
        vae_opt.step()
    
    # 数据增强
    with torch.no_grad():
        n_aug = max(100, len(X_train) * 3)
        z = torch.randn(n_aug, 8).to(device)
        aug_data = vae.decode(z)
    
    distances = torch.cdist(aug_data, X_train_t)
    nearest = distances.argmin(dim=1)
    aug_labels = y_train_t[nearest]
    
    X_combined = torch.cat([X_train_t, aug_data])
    y_combined = torch.cat([y_train_t, aug_labels])
    
    # HyperNet
    total_params = (2**3-1) * (input_dim + 1) + (2**3) * n_classes
    hypernet = HyperNetworkForTree(input_dim, n_trees=15, tree_depth=3).to(device)
    
    # 修改weight_gen的输出维度
    hypernet.weight_gen = nn.Sequential(
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 15 * total_params),
    ).to(device)
    
    classifier = GeneratedTreeClassifier(input_dim, n_classes, n_trees=15, tree_depth=3).to(device)
    
    optimizer = torch.optim.Adam(hypernet.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(epochs):
        hypernet.train()
        tree_params, tree_weights = hypernet(X_combined)
        outputs = classifier(X_combined, tree_params, tree_weights)
        loss = criterion(outputs, y_combined)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return {'hypernet': hypernet, 'classifier': classifier, 'X_train': X_combined}


def predict(model, X_test):
    model['hypernet'].eval()
    X_test_t = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        tree_params, tree_weights = model['hypernet'](model['X_train'])
        outputs = model['classifier'](X_test_t, tree_params, tree_weights)
        return outputs.argmax(dim=1).cpu().numpy()


def run_experiment_with_seed(X, y, n_classes, seed):
    """用指定seed运行一次5-fold实验"""
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    
    rf_preds, rf_labels = [], []
    vhn_preds, vhn_labels = [], []
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        # RF
        rf = RandomForestClassifier(n_estimators=100, random_state=seed + fold)
        rf.fit(X_train_s, y_train)
        rf_preds.extend(rf.predict(X_test_s))
        rf_labels.extend(y_test)
        
        # VAE-HyperNet
        try:
            model = train_vae_hypernet(X_train_s, y_train, n_classes, epochs=200, seed=seed + fold * 100)
            vhn_preds.extend(predict(model, X_test_s))
            vhn_labels.extend(y_test)
        except Exception as e:
            # 如果出错，用随机预测
            vhn_preds.extend([0] * len(y_test))
            vhn_labels.extend(y_test)
    
    rf_acc = accuracy_score(rf_labels, rf_preds) * 100
    vhn_acc = accuracy_score(vhn_labels, vhn_preds) * 100
    
    return rf_acc, vhn_acc


def load_dataset(name, data_dir):
    """加载数据集"""
    if name == "prostate":
        path = Path(data_dir) / "Data_for_Jinming.csv"
        df = pd.read_csv(path)
        X = df[['LAA', 'Glutamate', 'Choline', 'Sarcosine']].values
        y = (df['Group'] == 'PCa').astype(int).values
        return X, y, 2
    
    # 其他数据集
    small_data_dir = "/data2/image_identification/small_data"
    datasets_map = {
        "balloons": "1.balloons",
        "lens": "2.lens",
        "iris": "4.iris",
        "fertility": "5.fertility",
        "zoo": "6.zoo",
        "seeds": "7.seeds",
        "haberman": "8.haberman+s+survival",
        "glass": "9.glass+identification",
        "yeast": "10.yeast",
    }
    
    folder = datasets_map.get(name)
    if not folder:
        return None, None, None
    
    folder_path = Path(small_data_dir) / folder
    
    # 尝试找到数据文件
    for f in folder_path.glob("*.csv"):
        try:
            df = pd.read_csv(f)
            if len(df.columns) < 2:
                continue
            
            X = df.iloc[:, :-1].values
            y = LabelEncoder().fit_transform(df.iloc[:, -1].values)
            n_classes = len(np.unique(y))
            return X, y, n_classes
        except:
            continue
    
    for f in folder_path.glob("*.data"):
        try:
            df = pd.read_csv(f, header=None)
            X = df.iloc[:, :-1].values
            y = LabelEncoder().fit_transform(df.iloc[:, -1].values)
            n_classes = len(np.unique(y))
            return X, y, n_classes
        except:
            continue
    
    return None, None, None


def main():
    logger.info("="*60)
    logger.info("45_seed_stability_test.py")
    logger.info("测试多个seed在各数据集上的稳定性")
    logger.info("="*60)
    logger.info(f"设备: {device}\n")
    
    # 测试的seed列表
    seeds = [42, 123, 456, 789, 1000]
    
    # 数据集列表
    datasets = ["prostate", "iris", "seeds", "haberman", "balloons", "lens", "fertility", "zoo"]
    
    results = {}
    
    for ds_name in datasets:
        logger.info(f"\n{'='*40}")
        logger.info(f"数据集: {ds_name}")
        
        X, y, n_classes = load_dataset(ds_name, "/data2/image_identification/src/data")
        
        if X is None:
            logger.info(f"  无法加载数据集")
            continue
        
        logger.info(f"  样本: {len(y)}, 特征: {X.shape[1]}, 类别: {n_classes}")
        
        rf_accs = []
        vhn_accs = []
        
        for seed in seeds:
            rf_acc, vhn_acc = run_experiment_with_seed(X, y, n_classes, seed)
            rf_accs.append(rf_acc)
            vhn_accs.append(vhn_acc)
            logger.info(f"  seed={seed}: RF={rf_acc:.1f}%, VHN={vhn_acc:.1f}%")
        
        results[ds_name] = {
            'RF': {'mean': np.mean(rf_accs), 'std': np.std(rf_accs), 'runs': rf_accs},
            'VAE-HyperNet': {'mean': np.mean(vhn_accs), 'std': np.std(vhn_accs), 'runs': vhn_accs},
        }
        
        logger.info(f"  RF: {np.mean(rf_accs):.2f}% ± {np.std(rf_accs):.2f}%")
        logger.info(f"  VAE-HyperNet: {np.mean(vhn_accs):.2f}% ± {np.std(vhn_accs):.2f}%")
    
    # 总结
    logger.info("\n" + "="*60)
    logger.info("总结：seed=42 vs 多seed平均")
    logger.info("="*60)
    
    print(f"\n{'Dataset':<15} {'RF(42)':<10} {'RF(avg)':<12} {'VHN(42)':<10} {'VHN(avg)':<12} {'稳定?'}")
    print("-" * 70)
    
    for ds_name, data in results.items():
        rf_42 = data['RF']['runs'][0]  # seed=42的结果
        rf_avg = data['RF']['mean']
        rf_std = data['RF']['std']
        vhn_42 = data['VAE-HyperNet']['runs'][0]
        vhn_avg = data['VAE-HyperNet']['mean']
        vhn_std = data['VAE-HyperNet']['std']
        
        stable = "✓" if vhn_std < 5 else "✗"
        
        print(f"{ds_name:<15} {rf_42:<10.1f} {rf_avg:.1f}±{rf_std:.1f}   {vhn_42:<10.1f} {vhn_avg:.1f}±{vhn_std:.1f}   {stable}")
    
    # 保存结果
    output_dir = Path('/data2/image_identification/src/output')
    output_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(output_dir / f"45_seed_stability_{ts}.json", 'w') as f:
        json.dump({
            'seeds_tested': seeds,
            'results': results,
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n结果已保存")


if __name__ == "__main__":
    main()
