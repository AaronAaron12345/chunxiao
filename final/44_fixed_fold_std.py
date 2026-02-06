#!/usr/bin/env python3
"""
44_fixed_fold_std.py
固定K折划分，只改变模型训练的随机性来计算标准差

这更接近37_true_hypernet.py的设置：
- K折划分使用固定seed=42
- 标准差来自模型训练的随机性（不同的初始化）
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
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
        self.tree_depth = tree_depth
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
        
    def forward(self, X_train_batch):
        encoded = self.data_encoder(X_train_batch)
        data_summary = encoded.mean(dim=0, keepdim=True)
        tree_params = self.weight_gen(data_summary)
        tree_weights = torch.softmax(self.tree_weight_gen(data_summary), dim=-1)
        return tree_params, tree_weights


class GeneratedTreeClassifier(nn.Module):
    def __init__(self, input_dim, n_trees=15, tree_depth=3):
        super().__init__()
        self.input_dim = input_dim
        self.n_trees = n_trees
        self.tree_depth = tree_depth
        self.n_leaves = 2 ** tree_depth
        self.n_internal = 2 ** tree_depth - 1
        
    def forward(self, x, tree_params, tree_weights):
        batch_size = x.shape[0]
        param_per_tree = self.n_internal * (self.input_dim + 1) + self.n_leaves * 2
        all_probs = []
        
        for t in range(self.n_trees):
            start = t * param_per_tree
            split_w = tree_params[0, start:start + self.n_internal * self.input_dim].view(self.n_internal, self.input_dim)
            split_b = tree_params[0, start + self.n_internal * self.input_dim:start + self.n_internal * (self.input_dim + 1)]
            leaf_start = start + self.n_internal * (self.input_dim + 1)
            leaf_logits = tree_params[0, leaf_start:leaf_start + self.n_leaves * 2].view(self.n_leaves, 2)
            
            decisions = torch.sigmoid(torch.matmul(x, split_w.T) + split_b)
            
            leaf_probs = torch.ones(batch_size, self.n_leaves, device=x.device)
            for i in range(self.n_internal):
                left = 2 * i + 1
                right = 2 * i + 2
                d = decisions[:, i:i+1]
                if left < self.n_leaves:
                    leaf_probs[:, left] *= d.squeeze()
                if right < self.n_leaves:
                    leaf_probs[:, right] *= (1 - d).squeeze()
            
            leaf_probs = leaf_probs / (leaf_probs.sum(dim=1, keepdim=True) + 1e-8)
            tree_output = torch.matmul(leaf_probs, torch.softmax(leaf_logits, dim=-1))
            all_probs.append(tree_output * tree_weights[0, t])
        
        return torch.stack(all_probs).sum(dim=0)


def train_vae_hypernet(X_train, y_train, X_test, y_test, epochs=300, seed=42):
    set_seed(seed)
    
    input_dim = X_train.shape[1]
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    
    vae = VAE(input_dim, latent_dim=8).to(device)
    vae_opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
    
    for _ in range(100):
        recon, mu, logvar = vae(X_train_t)
        loss = nn.MSELoss()(recon, X_train_t) + 0.01 * (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))
        vae_opt.zero_grad()
        loss.backward()
        vae_opt.step()
    
    with torch.no_grad():
        n_aug = max(100, len(X_train) * 3)
        z = torch.randn(n_aug, 8).to(device)
        aug_data = vae.decode(z)
    
    distances = torch.cdist(aug_data, X_train_t)
    nearest = distances.argmin(dim=1)
    aug_labels = y_train_t[nearest]
    
    X_combined = torch.cat([X_train_t, aug_data])
    y_combined = torch.cat([y_train_t, aug_labels])
    
    hypernet = HyperNetworkForTree(input_dim, n_trees=15, tree_depth=3).to(device)
    classifier = GeneratedTreeClassifier(input_dim, n_trees=15, tree_depth=3).to(device)
    
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
        return outputs[:, 1].cpu().numpy()


def load_data():
    for path in ['/data2/image_identification/src/data/Data_for_Jinming.csv', 'data/Data_for_Jinming.csv']:
        if Path(path).exists():
            df = pd.read_csv(path)
            X = df[['LAA', 'Glutamate', 'Choline', 'Sarcosine']].values
            y = (df['Group'] == 'PCa').astype(int).values
            return X, y
    raise FileNotFoundError("找不到数据")


def main():
    logger.info("="*60)
    logger.info("44_fixed_fold_std.py")
    logger.info("固定K折划分 (seed=42)，计算模型训练的标准差")
    logger.info("="*60)
    logger.info(f"设备: {device}")
    
    X, y = load_data()
    logger.info(f"数据: {len(y)} 样本, {X.shape[1]} 特征\n")
    
    # 固定K折划分 (与37_true_hypernet.py相同)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    n_repeats = 5  # 重复次数
    n_voting_runs = 5  # 投票运行次数
    
    results = {'RF': [], 'VAE-HyperNet': [], 'VAE-HyperNet(投票)': []}
    
    logger.info(f"K折划分: 固定seed=42")
    logger.info(f"重复次数: {n_repeats} (改变模型训练seed)")
    logger.info(f"投票运行: {n_voting_runs}\n")
    
    for repeat in range(n_repeats):
        logger.info(f"[重复 {repeat+1}/{n_repeats}]")
        model_base_seed = 100 + repeat * 1000  # 每次重复用不同的模型seed
        
        rf_preds, rf_labels = [], []
        vhn_single_preds, vhn_single_labels = [], []
        vhn_vote_preds, vhn_vote_labels = [], []
        
        for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            
            # RF (使用变化的seed)
            rf = RandomForestClassifier(n_estimators=100, random_state=model_base_seed + fold)
            rf.fit(X_train_s, y_train)
            rf_preds.extend(rf.predict(X_test_s))
            rf_labels.extend(y_test)
            
            # VAE-HyperNet 单次
            seed_single = model_base_seed + fold * 100
            model = train_vae_hypernet(X_train_s, y_train, X_test_s, y_test, epochs=300, seed=seed_single)
            probs = predict(model, X_test_s)
            vhn_single_preds.extend((probs > 0.5).astype(int))
            vhn_single_labels.extend(y_test)
            
            # VAE-HyperNet 投票
            all_probs = []
            for run in range(n_voting_runs):
                run_seed = model_base_seed + fold * 100 + run
                model = train_vae_hypernet(X_train_s, y_train, X_test_s, y_test, epochs=300, seed=run_seed)
                all_probs.append(predict(model, X_test_s))
            
            avg_probs = np.mean(all_probs, axis=0)
            vhn_vote_preds.extend((avg_probs > 0.5).astype(int))
            vhn_vote_labels.extend(y_test)
        
        rf_acc = accuracy_score(rf_labels, rf_preds) * 100
        vhn_acc = accuracy_score(vhn_single_labels, vhn_single_preds) * 100
        vhn_vote_acc = accuracy_score(vhn_vote_labels, vhn_vote_preds) * 100
        
        results['RF'].append(rf_acc)
        results['VAE-HyperNet'].append(vhn_acc)
        results['VAE-HyperNet(投票)'].append(vhn_vote_acc)
        
        logger.info(f"  RF: {rf_acc:.2f}%, VAE-HyperNet: {vhn_acc:.2f}%, VAE-HyperNet(投票): {vhn_vote_acc:.2f}%")
    
    # 统计
    logger.info("\n" + "="*60)
    logger.info("最终结果 (固定K折划分, 均值 ± 标准差)")
    logger.info("="*60)
    
    for method in results:
        accs = results[method]
        mean = np.mean(accs)
        std = np.std(accs)
        logger.info(f"{method:25s}: {mean:.2f}% ± {std:.2f}%  {[f'{a:.1f}' for a in accs]}")
    
    best = max(results.items(), key=lambda x: np.mean(x[1]))
    logger.info(f"\n🏆 最佳: {best[0]} ({np.mean(best[1]):.2f}% ± {np.std(best[1]):.2f}%)")
    
    # 保存
    output_dir = Path('/data2/image_identification/src/output')
    output_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(output_dir / f"44_fixed_fold_{ts}.json", 'w') as f:
        json.dump({
            'experiment': 'Dataset 0 - Fixed K-Fold Split (seed=42)',
            'n_repeats': n_repeats,
            'n_voting_runs': n_voting_runs,
            'kfold_seed': 42,
            'results': {k: {'mean': np.mean(v), 'std': np.std(v), 'runs': v} for k, v in results.items()},
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n结果已保存")
    logger.info("\n" + "="*60)
    logger.info("说明:")
    logger.info("  - K折划分固定 (seed=42)，与37_true_hypernet.py一致")
    logger.info("  - 标准差来自模型训练的随机性（不同初始化）")
    logger.info("  - 这更接近论文中报告结果的标准差定义")
    logger.info("="*60)


if __name__ == "__main__":
    main()
