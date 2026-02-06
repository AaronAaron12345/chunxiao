#!/usr/bin/env python3
"""
42_test_dataset0_with_std.py
测试Dataset 0 (Prostate Cancer) 并记录标准差

关键说明：
- 5次运行使用不同随机种子(seed)，所以结果不同
- 标准差通过多次独立重复实验计算
- 每次重复使用完全独立的5-fold交叉验证
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
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============== VAE ==============
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(16, latent_dim)
        self.fc_var = nn.Linear(16, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
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


# ============== HyperNetwork ==============
class HyperNetworkForTree(nn.Module):
    """超网络：根据训练数据生成软决策树的权重"""
    
    def __init__(self, input_dim, n_trees=15, tree_depth=3, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.n_trees = n_trees
        self.tree_depth = tree_depth
        self.n_leaves = 2 ** tree_depth
        self.n_internal = 2 ** tree_depth - 1
        
        # 数据编码器
        self.data_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # 权重生成器
        total_tree_params = self.n_internal * (input_dim + 1) + self.n_leaves * 2
        self.weight_gen = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_trees * total_tree_params),
        )
        
        self.tree_weight_gen = nn.Linear(hidden_dim, n_trees)
        
    def forward(self, X_train_batch):
        # 编码训练数据
        encoded = self.data_encoder(X_train_batch)
        data_summary = encoded.mean(dim=0, keepdim=True)
        
        # 生成树参数
        tree_params = self.weight_gen(data_summary)
        tree_weights = torch.softmax(self.tree_weight_gen(data_summary), dim=-1)
        
        return tree_params, tree_weights


class GeneratedTreeClassifier(nn.Module):
    """使用HyperNet生成的权重进行分类"""
    
    def __init__(self, input_dim, n_trees=15, tree_depth=3):
        super().__init__()
        self.input_dim = input_dim
        self.n_trees = n_trees
        self.tree_depth = tree_depth
        self.n_leaves = 2 ** tree_depth
        self.n_internal = 2 ** tree_depth - 1
        
    def forward(self, x, tree_params, tree_weights):
        batch_size = x.shape[0]
        n_trees = self.n_trees
        
        # 解析参数
        param_per_tree = self.n_internal * (self.input_dim + 1) + self.n_leaves * 2
        all_probs = []
        
        for t in range(n_trees):
            start = t * param_per_tree
            split_w = tree_params[0, start:start + self.n_internal * self.input_dim].view(self.n_internal, self.input_dim)
            split_b = tree_params[0, start + self.n_internal * self.input_dim:start + self.n_internal * (self.input_dim + 1)]
            leaf_start = start + self.n_internal * (self.input_dim + 1)
            leaf_logits = tree_params[0, leaf_start:leaf_start + self.n_leaves * 2].view(self.n_leaves, 2)
            
            # 软决策
            decisions = torch.sigmoid(torch.matmul(x, split_w.T) + split_b)
            
            # 计算叶子概率
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
        
        final_probs = torch.stack(all_probs).sum(dim=0)
        return final_probs


def train_vae_hypernet(X_train, y_train, X_test, y_test, epochs=300, seed=42):
    """训练VAE-HyperNet模型"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    input_dim = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    
    # 转换数据
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    
    # VAE数据增强
    vae = VAE(input_dim).to(device)
    vae_opt = torch.optim.Adam(vae.parameters(), lr=0.001)
    
    for _ in range(100):
        recon, mu, logvar = vae(X_train_t)
        recon_loss = nn.MSELoss()(recon, X_train_t)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + 0.01 * kl_loss
        vae_opt.zero_grad()
        loss.backward()
        vae_opt.step()
    
    # 生成增强数据
    with torch.no_grad():
        n_aug = max(100, len(X_train) * 3)
        z = torch.randn(n_aug, 8).to(device)
        aug_data = vae.decode(z)
    
    # 为增强数据分配标签 (使用最近邻)
    distances = torch.cdist(aug_data, X_train_t)
    nearest = distances.argmin(dim=1)
    aug_labels = y_train_t[nearest]
    
    X_combined = torch.cat([X_train_t, aug_data])
    y_combined = torch.cat([y_train_t, aug_labels])
    
    # HyperNet
    hypernet = HyperNetworkForTree(input_dim, n_trees=20).to(device)
    classifier = GeneratedTreeClassifier(input_dim, n_trees=20).to(device)
    
    optimizer = torch.optim.Adam(hypernet.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        hypernet.train()
        tree_params, tree_weights = hypernet(X_combined)
        outputs = classifier(X_combined, tree_params, tree_weights)
        loss = criterion(outputs, y_combined)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return {'hypernet': hypernet, 'classifier': classifier, 'vae': vae}


def predict_vae_hypernet(model, X_train, X_test):
    """预测"""
    X_train_t = torch.FloatTensor(X_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    
    model['hypernet'].eval()
    with torch.no_grad():
        tree_params, tree_weights = model['hypernet'](X_train_t)
        outputs = model['classifier'](X_test_t, tree_params, tree_weights)
        preds = outputs[:, 1].cpu().numpy()  # 正类概率
    return preds


def load_prostate_data():
    """加载前列腺癌数据"""
    paths = [
        "/data2/image_identification/src/data/Data_for_Jinming.csv",
        "/Users/jinmingzhang/D/1/1Postg/Sem_13_Thesis/5_2026.01.28/src/data/Data_for_Jinming.csv",
        "data/Data_for_Jinming.csv",
    ]
    for path in paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            break
    else:
        raise FileNotFoundError("找不到数据集")
    
    # 使用特定列
    X = df[['LAA', 'Glutamate', 'Choline', 'Sarcosine']].values
    y = (df['Group'] == 'PCa').astype(int).values
    
    return X, y


def run_experiment(X, y, n_repeats=5, n_folds=5, n_voting_runs=5):
    """
    运行完整实验
    
    参数:
        n_repeats: 重复实验次数 (用于计算标准差)
        n_folds: K折交叉验证的K值
        n_voting_runs: 每个fold内的投票运行次数 (使用不同seed)
    """
    results = {
        'RF': [],
        'VAE+RF': [],
        'VAE-HyperNet': [],
        'VAE-HyperNet(投票)': [],
    }
    
    logger.info(f"\n{'='*60}")
    logger.info(f"实验设置:")
    logger.info(f"  - 数据: {len(y)} 样本")
    logger.info(f"  - 重复次数: {n_repeats} (用于计算标准差)")
    logger.info(f"  - K折: {n_folds}")
    logger.info(f"  - 投票运行次数: {n_voting_runs} (每个fold内)")
    logger.info(f"{'='*60}\n")
    
    for repeat in range(n_repeats):
        logger.info(f"\n[重复 {repeat+1}/{n_repeats}]")
        
        # 使用不同的随机种子进行K折划分
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42 + repeat * 1000)
        
        rf_preds_all, rf_labels_all = [], []
        vae_rf_preds_all, vae_rf_labels_all = [], []
        vhn_preds_all, vhn_labels_all = [], []
        vhn_vote_preds_all, vhn_vote_labels_all = [], []
        
        for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            
            # ============ RF 基线 ============
            rf = RandomForestClassifier(n_estimators=100, random_state=42 + repeat)
            rf.fit(X_train_s, y_train)
            rf_pred = rf.predict(X_test_s)
            rf_preds_all.extend(rf_pred)
            rf_labels_all.extend(y_test)
            
            # ============ VAE + RF ============
            # VAE数据增强
            vae = VAE(X_train.shape[1]).to(device)
            vae_opt = torch.optim.Adam(vae.parameters(), lr=0.001)
            X_train_t = torch.FloatTensor(X_train_s).to(device)
            
            for _ in range(100):
                recon, mu, logvar = vae(X_train_t)
                recon_loss = nn.MSELoss()(recon, X_train_t)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + 0.01 * kl_loss
                vae_opt.zero_grad()
                loss.backward()
                vae_opt.step()
            
            with torch.no_grad():
                n_aug = max(100, len(X_train) * 3)
                z = torch.randn(n_aug, 8).to(device)
                aug_data = vae.decode(z).cpu().numpy()
            
            distances = np.linalg.norm(aug_data[:, None] - X_train_s[None, :], axis=2)
            nearest = distances.argmin(axis=1)
            aug_labels = y_train[nearest]
            
            X_aug = np.vstack([X_train_s, aug_data])
            y_aug = np.concatenate([y_train, aug_labels])
            
            rf_vae = RandomForestClassifier(n_estimators=100, random_state=42 + repeat)
            rf_vae.fit(X_aug, y_aug)
            vae_rf_pred = rf_vae.predict(X_test_s)
            vae_rf_preds_all.extend(vae_rf_pred)
            vae_rf_labels_all.extend(y_test)
            
            # ============ VAE-HyperNet (单次) ============
            base_seed = 42 + repeat * 1000 + fold * 100
            model = train_vae_hypernet(X_train_s, y_train, X_test_s, y_test,
                                       epochs=300, seed=base_seed)
            vhn_probs = predict_vae_hypernet(model, X_train_s, X_test_s)
            vhn_pred = (vhn_probs > 0.5).astype(int)
            vhn_preds_all.extend(vhn_pred)
            vhn_labels_all.extend(y_test)
            
            # ============ VAE-HyperNet (投票) ============
            # 关键：使用不同seed运行多次，然后投票
            all_run_probs = []
            for run in range(n_voting_runs):
                run_seed = base_seed + run  # 每次运行使用不同seed
                model = train_vae_hypernet(X_train_s, y_train, X_test_s, y_test,
                                           epochs=300, seed=run_seed)
                probs = predict_vae_hypernet(model, X_train_s, X_test_s)
                all_run_probs.append(probs)
            
            # 软投票：取5次运行的平均概率
            avg_probs = np.mean(all_run_probs, axis=0)
            voted_pred = (avg_probs > 0.5).astype(int)
            vhn_vote_preds_all.extend(voted_pred)
            vhn_vote_labels_all.extend(y_test)
        
        # 计算本次重复的准确率
        rf_acc = accuracy_score(rf_labels_all, rf_preds_all) * 100
        vae_rf_acc = accuracy_score(vae_rf_labels_all, vae_rf_preds_all) * 100
        vhn_acc = accuracy_score(vhn_labels_all, vhn_preds_all) * 100
        vhn_vote_acc = accuracy_score(vhn_vote_labels_all, vhn_vote_preds_all) * 100
        
        results['RF'].append(rf_acc)
        results['VAE+RF'].append(vae_rf_acc)
        results['VAE-HyperNet'].append(vhn_acc)
        results['VAE-HyperNet(投票)'].append(vhn_vote_acc)
        
        logger.info(f"  RF: {rf_acc:.2f}%, VAE+RF: {vae_rf_acc:.2f}%, "
                    f"VAE-HyperNet: {vhn_acc:.2f}%, VAE-HyperNet(投票): {vhn_vote_acc:.2f}%")
    
    return results


def main():
    logger.info("="*60)
    logger.info("42_test_dataset0_with_std.py")
    logger.info("测试Dataset 0 (Prostate Cancer) - 记录完整标准差")
    logger.info("="*60)
    
    # 加载数据
    X, y = load_prostate_data()
    logger.info(f"\n数据: {len(y)} 样本, {X.shape[1]} 特征")
    logger.info(f"设备: {device}")
    
    # 运行实验 (5次重复，用于计算标准差)
    results = run_experiment(X, y, n_repeats=5, n_folds=5, n_voting_runs=5)
    
    # 统计结果
    logger.info("\n" + "="*60)
    logger.info("最终结果 (5次独立重复实验)")
    logger.info("="*60)
    
    summary = []
    for method, accs in results.items():
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        summary.append({
            'Method': method,
            'Mean': mean_acc,
            'Std': std_acc,
            'Runs': accs
        })
        logger.info(f"  {method:25s}: {mean_acc:.2f}% ± {std_acc:.2f}%  {[f'{a:.1f}' for a in accs]}")
    
    # 找出最佳方法
    best_method = max(summary, key=lambda x: x['Mean'])
    logger.info(f"\n🏆 最佳方法: {best_method['Method']} ({best_method['Mean']:.2f}% ± {best_method['Std']:.2f}%)")
    
    # 保存结果
    output_dir = '/data2/image_identification/src/output' if os.path.exists('/data2') else './output'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"42_dataset0_with_std_{timestamp}.json")
    
    import json
    with open(output_file, 'w') as f:
        json.dump({
            'experiment': 'Dataset 0 (Prostate Cancer) with Standard Deviation',
            'n_repeats': 5,
            'n_folds': 5,
            'n_voting_runs': 5,
            'results': {k: {'mean': np.mean(v), 'std': np.std(v), 'runs': v} 
                        for k, v in results.items()},
            'timestamp': timestamp
        }, f, indent=2)
    
    logger.info(f"\n结果已保存: {output_file}")
    
    # 解释说明
    logger.info("\n" + "="*60)
    logger.info("说明:")
    logger.info("  1. 标准差来源: 5次独立重复实验 (不同的K折划分)")
    logger.info("  2. 投票机制: 每个fold内运行5次(不同seed)，取平均概率投票")
    logger.info("  3. 为什么5次运行不同: 神经网络初始化使用不同随机种子")
    logger.info("="*60)


if __name__ == "__main__":
    main()
