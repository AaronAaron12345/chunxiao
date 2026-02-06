#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
40_test_dataset0.py - 测试第0个数据集（Prostate Cancer）

使用与39相同的方法测试用户原始的Prostate Cancer数据集
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ============== VAE 模型 ==============
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        hidden_dim = max(16, input_dim * 2)
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
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

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

# ============== 软决策树 ==============
class SoftDecisionTree(nn.Module):
    def __init__(self, input_dim, num_classes, depth=4):
        super().__init__()
        self.depth = depth
        self.num_classes = num_classes
        self.num_inner = 2 ** depth - 1
        self.num_leaves = 2 ** depth
        
        self.inner_nodes = nn.Linear(input_dim, self.num_inner)
        self.leaf_distributions = nn.Parameter(torch.randn(self.num_leaves, num_classes))
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        decisions = torch.sigmoid(self.inner_nodes(x))
        
        path_probs = torch.ones(batch_size, 1, device=x.device)
        
        for d in range(self.depth):
            start_idx = 2 ** d - 1
            end_idx = 2 ** (d + 1) - 1
            
            level_decisions = decisions[:, start_idx:end_idx]
            
            left_probs = path_probs * level_decisions
            right_probs = path_probs * (1 - level_decisions)
            
            path_probs = torch.cat([left_probs, right_probs], dim=1)
            
            indices = torch.arange(path_probs.shape[1], device=x.device)
            left_indices = indices[::2]
            right_indices = indices[1::2]
            reordered = torch.zeros_like(path_probs)
            reordered[:, :len(left_indices)] = path_probs[:, left_indices]
            reordered[:, len(left_indices):] = path_probs[:, right_indices]
            path_probs = reordered
        
        leaf_probs = torch.softmax(self.leaf_distributions, dim=1)
        output = torch.matmul(path_probs, leaf_probs)
        
        return output

# ============== HyperNetwork ==============
class HyperNetwork(nn.Module):
    def __init__(self, input_dim, num_classes, num_trees=15, tree_depth=4):
        super().__init__()
        self.num_trees = num_trees
        self.tree_depth = tree_depth
        self.num_classes = num_classes
        self.input_dim = input_dim
        
        self.num_inner = 2 ** tree_depth - 1
        self.num_leaves = 2 ** tree_depth
        
        context_dim = 64
        self.context_encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, context_dim),
            nn.ReLU()
        )
        
        inner_params = self.num_inner * (input_dim + 1)
        leaf_params = self.num_leaves * num_classes
        params_per_tree = inner_params + leaf_params
        
        self.hypernet = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_trees * params_per_tree)
        )
        
        self.tree_weights = nn.Parameter(torch.ones(num_trees) / num_trees)
    
    def forward(self, x, context):
        batch_size = x.shape[0]
        
        ctx = self.context_encoder(context.mean(dim=0, keepdim=True))
        params = self.hypernet(ctx).squeeze(0)
        
        inner_params = self.num_inner * (self.input_dim + 1)
        leaf_params = self.num_leaves * self.num_classes
        params_per_tree = inner_params + leaf_params
        
        all_outputs = []
        
        for t in range(self.num_trees):
            start = t * params_per_tree
            tree_params = params[start:start + params_per_tree]
            
            inner_w = tree_params[:self.num_inner * self.input_dim].view(self.num_inner, self.input_dim)
            inner_b = tree_params[self.num_inner * self.input_dim:inner_params].view(self.num_inner)
            leaf_dist = tree_params[inner_params:].view(self.num_leaves, self.num_classes)
            
            decisions = torch.sigmoid(torch.matmul(x, inner_w.t()) + inner_b)
            
            path_probs = torch.ones(batch_size, 1, device=x.device)
            
            for d in range(self.tree_depth):
                start_idx = 2 ** d - 1
                end_idx = 2 ** (d + 1) - 1
                
                level_decisions = decisions[:, start_idx:end_idx]
                
                left_probs = path_probs * level_decisions
                right_probs = path_probs * (1 - level_decisions)
                
                path_probs = torch.cat([left_probs, right_probs], dim=1)
                
                indices = torch.arange(path_probs.shape[1], device=x.device)
                left_indices = indices[::2]
                right_indices = indices[1::2]
                reordered = torch.zeros_like(path_probs)
                reordered[:, :len(left_indices)] = path_probs[:, left_indices]
                reordered[:, len(left_indices):] = path_probs[:, right_indices]
                path_probs = reordered
            
            leaf_probs = torch.softmax(leaf_dist, dim=1)
            tree_output = torch.matmul(path_probs, leaf_probs)
            all_outputs.append(tree_output)
        
        outputs = torch.stack(all_outputs, dim=0)
        weights = torch.softmax(self.tree_weights, dim=0)
        final_output = torch.einsum('tbc,t->bc', outputs, weights)
        
        return final_output

# ============== 数据增强函数 ==============
def augment_data_with_vae(X_train, y_train, vae, num_augment=100, device='cpu'):
    """使用VAE进行数据增强"""
    X_tensor = torch.FloatTensor(X_train).to(device)
    
    vae.eval()
    with torch.no_grad():
        mu, logvar = vae.encode(X_tensor)
    
    augmented_X = []
    augmented_y = []
    
    classes = np.unique(y_train)
    samples_per_class = num_augment // len(classes)
    
    for cls in classes:
        cls_indices = np.where(y_train == cls)[0]
        cls_mu = mu[cls_indices]
        cls_logvar = logvar[cls_indices]
        
        for _ in range(samples_per_class):
            idx = np.random.randint(len(cls_indices))
            z = vae.reparameterize(cls_mu[idx:idx+1], cls_logvar[idx:idx+1])
            
            noise = torch.randn_like(z) * 0.1
            z = z + noise
            
            with torch.no_grad():
                new_sample = vae.decode(z)
            
            augmented_X.append(new_sample.cpu().numpy())
            augmented_y.append(cls)
    
    augmented_X = np.vstack(augmented_X)
    augmented_y = np.array(augmented_y)
    
    X_combined = np.vstack([X_train, augmented_X])
    y_combined = np.concatenate([y_train, augmented_y])
    
    return X_combined, y_combined

# ============== 主测试函数 ==============
def test_dataset0():
    """测试第0个数据集 - Prostate Cancer"""
    
    print("=" * 60)
    print("测试数据集 0: Prostate Cancer")
    print("=" * 60)
    
    # 加载数据
    data_path = "../data/Data_for_Jinming.csv"
    if not os.path.exists(data_path):
        data_path = "/data2/image_identification/data/Data_for_Jinming.csv"
    
    df = pd.read_csv(data_path)
    
    # 处理数据
    X = df[['LAA', 'Glutamate', 'Choline', 'Sarcosine']].values
    y = (df['Group'] == 'PCa').astype(int).values
    
    print(f"数据形状: {X.shape}")
    print(f"类别分布: {np.bincount(y)}")
    
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    
    # 多次运行
    n_runs = 3
    n_folds = 5
    
    rf_results = []
    vae_hypernet_results = []
    
    for run in range(n_runs):
        print(f"\n--- 运行 {run + 1}/{n_runs} ---")
        np.random.seed(run * 42)
        torch.manual_seed(run * 42)
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=run * 42)
        
        rf_fold_accs = []
        vae_hypernet_fold_accs = []
        vae_hypernet_fold_prec = []
        vae_hypernet_fold_rec = []
        vae_hypernet_fold_f1 = []
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # 标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # --- RF 基线 ---
            rf = RandomForestClassifier(n_estimators=100, random_state=run * 42)
            rf.fit(X_train_scaled, y_train)
            rf_pred = rf.predict(X_test_scaled)
            rf_acc = accuracy_score(y_test, rf_pred)
            rf_fold_accs.append(rf_acc)
            
            # --- VAE-HyperNet ---
            # 训练VAE
            vae = VAE(n_features, latent_dim=8).to(device)
            vae_optimizer = optim.Adam(vae.parameters(), lr=0.001)
            
            X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
            
            vae.train()
            for epoch in range(100):
                vae_optimizer.zero_grad()
                recon, mu, logvar = vae(X_train_tensor)
                loss = vae_loss(recon, X_train_tensor, mu, logvar)
                loss.backward()
                vae_optimizer.step()
            
            # 数据增强
            X_aug, y_aug = augment_data_with_vae(
                X_train_scaled, y_train, vae, 
                num_augment=200, device=device
            )
            
            # 训练HyperNet
            hypernet = HyperNetwork(
                input_dim=n_features,
                num_classes=n_classes,
                num_trees=15,
                tree_depth=4
            ).to(device)
            
            optimizer = optim.Adam(hypernet.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            X_aug_tensor = torch.FloatTensor(X_aug).to(device)
            y_aug_tensor = torch.LongTensor(y_aug).to(device)
            
            hypernet.train()
            for epoch in range(200):
                optimizer.zero_grad()
                outputs = hypernet(X_aug_tensor, X_aug_tensor)
                loss = criterion(outputs, y_aug_tensor)
                loss.backward()
                optimizer.step()
            
            # 预测
            hypernet.eval()
            X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
            
            with torch.no_grad():
                outputs = hypernet(X_test_tensor, X_aug_tensor)
                _, preds = torch.max(outputs, 1)
                preds = preds.cpu().numpy()
            
            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, average='weighted', zero_division=0)
            rec = recall_score(y_test, preds, average='weighted', zero_division=0)
            f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
            
            vae_hypernet_fold_accs.append(acc)
            vae_hypernet_fold_prec.append(prec)
            vae_hypernet_fold_rec.append(rec)
            vae_hypernet_fold_f1.append(f1)
            
            print(f"  Fold {fold + 1}: RF={rf_acc*100:.1f}%, VAE-HyperNet={acc*100:.1f}%")
        
        rf_results.append(np.mean(rf_fold_accs) * 100)
        vae_hypernet_results.append(np.mean(vae_hypernet_fold_accs) * 100)
    
    # 汇总结果
    results = {
        "dataset": "0.Prostate_Cancer",
        "samples": n_samples,
        "features": n_features,
        "classes": n_classes,
        "RF": {
            "accuracy_mean": np.mean(rf_results),
            "accuracy_std": np.std(rf_results)
        },
        "VAE_HyperNet": {
            "accuracy_mean": np.mean(vae_hypernet_results),
            "accuracy_std": np.std(vae_hypernet_results),
            "precision_mean": np.mean(vae_hypernet_fold_prec) * 100,
            "recall_mean": np.mean(vae_hypernet_fold_rec) * 100,
            "f1_mean": np.mean(vae_hypernet_fold_f1) * 100
        }
    }
    
    print("\n" + "=" * 60)
    print("最终结果 - Prostate Cancer (Dataset 0)")
    print("=" * 60)
    print(f"样本数: {n_samples}, 特征数: {n_features}, 类别数: {n_classes}")
    print(f"RF:           {results['RF']['accuracy_mean']:.1f}% ± {results['RF']['accuracy_std']:.1f}%")
    print(f"VAE-HyperNet: {results['VAE_HyperNet']['accuracy_mean']:.1f}% ± {results['VAE_HyperNet']['accuracy_std']:.1f}%")
    print(f"  Precision: {results['VAE_HyperNet']['precision_mean']:.1f}%")
    print(f"  Recall:    {results['VAE_HyperNet']['recall_mean']:.1f}%")
    print(f"  F1-Score:  {results['VAE_HyperNet']['f1_mean']:.1f}%")
    
    winner = "VAE-HyperNet" if results['VAE_HyperNet']['accuracy_mean'] > results['RF']['accuracy_mean'] else "RF"
    print(f"\n获胜者: {winner}")
    
    # 保存结果
    os.makedirs('output', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(f'output/dataset0_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存到: output/dataset0_results_{timestamp}.json")
    
    return results

if __name__ == "__main__":
    test_dataset0()
