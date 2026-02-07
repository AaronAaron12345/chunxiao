#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
41_test_dataset0_optimized.py - 优化版测试第0个数据集（Prostate Cancer）

使用更多数据增强和更多运行次数来获得更稳定的结果
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
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
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
        hidden_dim = max(32, input_dim * 4)
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
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

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss

# ============== HyperNetwork ==============
class HyperNetwork(nn.Module):
    def __init__(self, input_dim, num_classes, num_trees=20, tree_depth=4):
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
            nn.Dropout(0.1),
            nn.Linear(32, context_dim),
            nn.ReLU()
        )
        
        inner_params = self.num_inner * (input_dim + 1)
        leaf_params = self.num_leaves * num_classes
        params_per_tree = inner_params + leaf_params
        
        self.hypernet = nn.Sequential(
            nn.Linear(context_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_trees * params_per_tree)
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
def augment_data_with_vae(X_train, y_train, vae, num_augment=300, device='cpu'):
    """使用VAE进行数据增强，增加更多样本"""
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
            # 随机选择两个同类样本进行插值
            idx1 = np.random.randint(len(cls_indices))
            idx2 = np.random.randint(len(cls_indices))
            
            alpha = np.random.random()
            z_mu = alpha * cls_mu[idx1] + (1 - alpha) * cls_mu[idx2]
            z_logvar = alpha * cls_logvar[idx1] + (1 - alpha) * cls_logvar[idx2]
            
            z = vae.reparameterize(z_mu.unsqueeze(0), z_logvar.unsqueeze(0))
            
            # 添加小噪声
            noise = torch.randn_like(z) * 0.05
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
    
    print("=" * 70)
    print("测试数据集 0: Prostate Cancer (优化版)")
    print("=" * 70)
    
    # 加载数据
    data_path = "../data/Data_for_Jinming.csv"
    if not os.path.exists(data_path):
        data_path = "/data2/image_identification/data/Data_for_Jinming.csv"
    
    df = pd.read_csv(data_path)
    
    # 处理数据
    X = df[['LAA', 'Glutamate', 'Choline', 'Sarcosine']].values
    y = (df['Group'] == 'PCa').astype(int).values
    
    print(f"数据形状: {X.shape}")
    print(f"类别分布: non-PCa={np.sum(y==0)}, PCa={np.sum(y==1)}")
    
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    
    # 更多运行次数以获得稳定结果
    n_runs = 5
    n_folds = 5
    
    rf_all_results = []
    vae_rf_all_results = []
    vae_hypernet_all_results = []
    
    all_metrics = {
        'RF': {'acc': [], 'prec': [], 'rec': [], 'f1': []},
        'VAE_RF': {'acc': [], 'prec': [], 'rec': [], 'f1': []},
        'VAE_HyperNet': {'acc': [], 'prec': [], 'rec': [], 'f1': []}
    }
    
    for run in range(n_runs):
        print(f"\n{'='*30} 运行 {run + 1}/{n_runs} {'='*30}")
        np.random.seed(run * 42 + 123)
        torch.manual_seed(run * 42 + 123)
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=run * 42 + 123)
        
        rf_fold_accs = []
        vae_rf_fold_accs = []
        vae_hypernet_fold_accs = []
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # 标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # --- 1. RF 基线 ---
            rf = RandomForestClassifier(n_estimators=100, random_state=run * 42)
            rf.fit(X_train_scaled, y_train)
            rf_pred = rf.predict(X_test_scaled)
            rf_acc = accuracy_score(y_test, rf_pred)
            rf_fold_accs.append(rf_acc)
            
            # --- 2. 训练VAE ---
            vae = VAE(n_features, latent_dim=8).to(device)
            vae_optimizer = optim.Adam(vae.parameters(), lr=0.002)
            
            X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
            
            vae.train()
            for epoch in range(200):
                vae_optimizer.zero_grad()
                recon, mu, logvar = vae(X_train_tensor)
                loss = vae_loss(recon, X_train_tensor, mu, logvar, beta=0.5)
                loss.backward()
                vae_optimizer.step()
            
            # 数据增强
            X_aug, y_aug = augment_data_with_vae(
                X_train_scaled, y_train, vae, 
                num_augment=300, device=device
            )
            
            # --- 3. VAE + RF ---
            rf_aug = RandomForestClassifier(n_estimators=100, random_state=run * 42)
            rf_aug.fit(X_aug, y_aug)
            vae_rf_pred = rf_aug.predict(X_test_scaled)
            vae_rf_acc = accuracy_score(y_test, vae_rf_pred)
            vae_rf_fold_accs.append(vae_rf_acc)
            
            # --- 4. VAE + HyperNet ---
            hypernet = HyperNetwork(
                input_dim=n_features,
                num_classes=n_classes,
                num_trees=20,
                tree_depth=4
            ).to(device)
            
            optimizer = optim.Adam(hypernet.parameters(), lr=0.002)
            criterion = nn.CrossEntropyLoss()
            
            X_aug_tensor = torch.FloatTensor(X_aug).to(device)
            y_aug_tensor = torch.LongTensor(y_aug).to(device)
            
            hypernet.train()
            best_loss = float('inf')
            patience = 0
            
            for epoch in range(300):
                optimizer.zero_grad()
                outputs = hypernet(X_aug_tensor, X_aug_tensor)
                loss = criterion(outputs, y_aug_tensor)
                loss.backward()
                optimizer.step()
                
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    patience = 0
                else:
                    patience += 1
                
                if patience > 30:
                    break
            
            # 预测
            hypernet.eval()
            X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
            
            with torch.no_grad():
                outputs = hypernet(X_test_tensor, X_aug_tensor)
                _, preds = torch.max(outputs, 1)
                preds = preds.cpu().numpy()
            
            vae_hypernet_acc = accuracy_score(y_test, preds)
            vae_hypernet_fold_accs.append(vae_hypernet_acc)
            
            print(f"  Fold {fold + 1}: RF={rf_acc*100:.1f}%, VAE+RF={vae_rf_acc*100:.1f}%, VAE+HyperNet={vae_hypernet_acc*100:.1f}%")
        
        rf_all_results.append(np.mean(rf_fold_accs) * 100)
        vae_rf_all_results.append(np.mean(vae_rf_fold_accs) * 100)
        vae_hypernet_all_results.append(np.mean(vae_hypernet_fold_accs) * 100)
    
    # 汇总结果
    results = {
        "dataset": "0.Prostate_Cancer",
        "samples": n_samples,
        "features": n_features,
        "classes": n_classes,
        "n_runs": n_runs,
        "n_folds": n_folds,
        "RF": {
            "accuracy_mean": np.mean(rf_all_results),
            "accuracy_std": np.std(rf_all_results),
            "all_runs": rf_all_results
        },
        "VAE_RF": {
            "accuracy_mean": np.mean(vae_rf_all_results),
            "accuracy_std": np.std(vae_rf_all_results),
            "all_runs": vae_rf_all_results
        },
        "VAE_HyperNet": {
            "accuracy_mean": np.mean(vae_hypernet_all_results),
            "accuracy_std": np.std(vae_hypernet_all_results),
            "all_runs": vae_hypernet_all_results
        }
    }
    
    print("\n" + "=" * 70)
    print("最终结果 - Prostate Cancer (Dataset 0)")
    print("=" * 70)
    print(f"样本数: {n_samples}, 特征数: {n_features}, 类别数: {n_classes}")
    print(f"运行次数: {n_runs}, 折数: {n_folds}")
    print("-" * 70)
    print(f"{'方法':<20} {'准确率':<20} {'各次运行结果'}")
    print("-" * 70)
    print(f"{'RF':<20} {results['RF']['accuracy_mean']:.2f}% ± {results['RF']['accuracy_std']:.2f}%  {[f'{x:.1f}' for x in rf_all_results]}")
    print(f"{'VAE+RF':<20} {results['VAE_RF']['accuracy_mean']:.2f}% ± {results['VAE_RF']['accuracy_std']:.2f}%  {[f'{x:.1f}' for x in vae_rf_all_results]}")
    print(f"{'VAE+HyperNet':<20} {results['VAE_HyperNet']['accuracy_mean']:.2f}% ± {results['VAE_HyperNet']['accuracy_std']:.2f}%  {[f'{x:.1f}' for x in vae_hypernet_all_results]}")
    print("-" * 70)
    
    # 找出获胜者
    methods = ['RF', 'VAE_RF', 'VAE_HyperNet']
    means = [results['RF']['accuracy_mean'], results['VAE_RF']['accuracy_mean'], results['VAE_HyperNet']['accuracy_mean']]
    winner = methods[np.argmax(means)]
    print(f"\n🏆 获胜者: {winner} ({max(means):.2f}%)")
    
    # 保存结果
    os.makedirs('output', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(f'output/dataset0_optimized_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 保存CSV格式便于查看
    csv_data = {
        "Dataset": ["0.Prostate_Cancer"],
        "Samples": [n_samples],
        "Features": [n_features],
        "Classes": [n_classes],
        "RF_Mean": [results['RF']['accuracy_mean']],
        "RF_Std": [results['RF']['accuracy_std']],
        "VAE_RF_Mean": [results['VAE_RF']['accuracy_mean']],
        "VAE_RF_Std": [results['VAE_RF']['accuracy_std']],
        "VAE_HyperNet_Mean": [results['VAE_HyperNet']['accuracy_mean']],
        "VAE_HyperNet_Std": [results['VAE_HyperNet']['accuracy_std']],
        "Winner": [winner]
    }
    pd.DataFrame(csv_data).to_csv(f'output/dataset0_summary_{timestamp}.csv', index=False)
    
    print(f"\n结果已保存到:")
    print(f"  - output/dataset0_optimized_{timestamp}.json")
    print(f"  - output/dataset0_summary_{timestamp}.csv")
    
    return results

if __name__ == "__main__":
    test_dataset0()
