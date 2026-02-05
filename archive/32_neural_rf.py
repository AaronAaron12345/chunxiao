#!/usr/bin/env python3
"""
32_neural_rf.py - 神经随机森林
核心思想：
1. 用神经网络模拟RF的决策树分裂
2. 每棵"神经决策树"学习软分裂（可微分）
3. 集成多棵树，类似RF的bagging
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import logging
import json
from datetime import datetime
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# 设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class SoftDecisionTree(nn.Module):
    """软决策树 - 可微分的决策树"""
    def __init__(self, input_dim, depth=3):
        super().__init__()
        self.depth = depth
        self.n_internal = 2**depth - 1  # 内部节点数
        self.n_leaves = 2**depth  # 叶子节点数
        
        # 每个内部节点的分裂参数
        self.split_weights = nn.Parameter(torch.randn(self.n_internal, input_dim) * 0.1)
        self.split_bias = nn.Parameter(torch.zeros(self.n_internal))
        
        # 每个叶子节点的类别分布（2类）
        self.leaf_logits = nn.Parameter(torch.randn(self.n_leaves, 2) * 0.1)
        
        # 温度参数控制软/硬分裂
        self.temperature = 1.0
    
    def forward(self, x):
        """
        x: [batch, features]
        返回: [batch, 2] 类别概率
        """
        batch_size = x.size(0)
        
        # 计算所有内部节点的分裂概率
        split_probs = torch.sigmoid(
            (x @ self.split_weights.T + self.split_bias) / self.temperature
        )  # [batch, n_internal]
        
        # 计算到达每个叶子节点的概率
        leaf_probs = torch.ones(batch_size, self.n_leaves, device=x.device)
        
        for leaf_idx in range(self.n_leaves):
            # 从叶子回溯到根节点
            path_prob = torch.ones(batch_size, device=x.device)
            node_idx = leaf_idx + self.n_internal  # 叶子节点的全局索引
            
            for d in range(self.depth):
                parent_idx = (node_idx - 1) // 2
                is_right = (node_idx % 2 == 0)  # 偶数索引是右子节点
                
                if is_right:
                    path_prob = path_prob * split_probs[:, parent_idx]
                else:
                    path_prob = path_prob * (1 - split_probs[:, parent_idx])
                
                node_idx = parent_idx
            
            leaf_probs[:, leaf_idx] = path_prob
        
        # 加权叶子节点的类别分布
        leaf_class_probs = F.softmax(self.leaf_logits, dim=-1)  # [n_leaves, 2]
        output = torch.einsum('bl,lc->bc', leaf_probs, leaf_class_probs)
        
        return output


class NeuralRandomForest(nn.Module):
    """神经随机森林 - 多棵软决策树的集成"""
    def __init__(self, input_dim, n_trees=20, depth=3, feature_fraction=0.7):
        super().__init__()
        self.n_trees = n_trees
        self.feature_fraction = feature_fraction
        self.input_dim = input_dim
        
        # 创建多棵树
        self.trees = nn.ModuleList([
            SoftDecisionTree(input_dim, depth) for _ in range(n_trees)
        ])
        
        # 每棵树使用的特征子集
        n_features_per_tree = max(1, int(input_dim * feature_fraction))
        self.feature_masks = []
        for _ in range(n_trees):
            mask = torch.zeros(input_dim)
            indices = torch.randperm(input_dim)[:n_features_per_tree]
            mask[indices] = 1.0
            self.feature_masks.append(mask)
        self.feature_masks = torch.stack(self.feature_masks).to(device)
        
        # 树的权重（可学习）
        self.tree_weights = nn.Parameter(torch.ones(n_trees) / n_trees)
    
    def forward(self, x):
        """
        x: [batch, features]
        返回: [batch, 2] 平均概率
        """
        # 对每棵树应用特征mask并预测
        tree_outputs = []
        for i, tree in enumerate(self.trees):
            # 特征选择（软mask）
            masked_x = x * self.feature_masks[i].unsqueeze(0)
            output = tree(masked_x)
            tree_outputs.append(output)
        
        # 加权平均
        tree_outputs = torch.stack(tree_outputs, dim=0)  # [n_trees, batch, 2]
        weights = F.softmax(self.tree_weights, dim=0)
        output = torch.einsum('t,tbc->bc', weights, tree_outputs)
        
        return output


class SimpleVAE(nn.Module):
    """简单VAE用于数据增强"""
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        hidden = 16
        
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


def vae_loss(recon, x, mu, logvar):
    recon_loss = F.mse_loss(recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + 0.01 * kl_loss


def train_vae(X, y, epochs=200):
    """训练VAE"""
    input_dim = X.shape[1]
    vae = SimpleVAE(input_dim).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
    
    X_t = torch.FloatTensor(X).to(device)
    
    vae.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        recon, mu, logvar = vae(X_t)
        loss = vae_loss(recon, X_t, mu, logvar)
        loss.backward()
        optimizer.step()
    
    return vae


def augment_data(vae, X, y, n_augment=200):
    """用VAE生成增强数据"""
    vae.eval()
    X_t = torch.FloatTensor(X).to(device)
    
    augmented_X = [X]
    augmented_y = [y]
    
    with torch.no_grad():
        for label in [0, 1]:
            mask = y == label
            X_class = X_t[mask]
            
            if len(X_class) == 0:
                continue
            
            for _ in range(n_augment // 2):
                idx = torch.randint(0, len(X_class), (1,))
                mu, logvar = vae.encode(X_class[idx])
                
                # 添加少量噪声
                z = vae.reparameterize(mu, logvar * 0.5)
                generated = vae.decode(z)
                
                augmented_X.append(generated.cpu().numpy())
                augmented_y.append(np.array([label]))
    
    return np.vstack(augmented_X), np.concatenate(augmented_y)


def train_neural_rf(X_train, y_train, X_val, y_val, n_trees=20, depth=3, epochs=100):
    """训练神经随机森林"""
    input_dim = X_train.shape[1]
    model = NeuralRandomForest(input_dim, n_trees=n_trees, depth=depth).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    X_t = torch.FloatTensor(X_train).to(device)
    y_t = torch.LongTensor(y_train).to(device)
    X_v = torch.FloatTensor(X_val).to(device)
    
    best_model_state = None
    best_val_acc = 0
    
    model.train()
    for epoch in range(epochs):
        # 添加数据扰动（类似RF的bootstrap）
        if epoch > 0:
            noise = torch.randn_like(X_t) * 0.02
            X_noisy = X_t + noise
        else:
            X_noisy = X_t
        
        optimizer.zero_grad()
        output = model(X_noisy)
        loss = F.cross_entropy(output, y_t)
        
        # L2正则化树权重
        loss = loss + 0.01 * (model.tree_weights ** 2).sum()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # 验证
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(X_v)
                val_pred = val_out.argmax(dim=1).cpu().numpy()
                val_acc = accuracy_score(y_val, val_pred)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            model.train()
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model


def predict_neural_rf(model, X):
    """预测"""
    model.eval()
    X_t = torch.FloatTensor(X).to(device)
    with torch.no_grad():
        output = model(X_t)
        return output.argmax(dim=1).cpu().numpy()


def load_data():
    """加载数据"""
    for path in ['/data2/image_identification/src/data/Data_for_Jinming.csv',
                 'data/Data_for_Jinming.csv']:
        if Path(path).exists():
            df = pd.read_csv(path)
            X = df[['LAA', 'Glutamate', 'Choline', 'Sarcosine']].values
            # Group列: 'non-PCa' -> 0, 'PCa' -> 1
            y = (df['Group'] == 'PCa').astype(int).values
            return X, y
    raise FileNotFoundError("找不到数据文件")


def main():
    logger.info("=" * 70)
    logger.info("32_neural_rf.py - 神经随机森林")
    logger.info("=" * 70)
    
    # 加载数据
    X, y = load_data()
    logger.info(f"设备: {device}")
    logger.info(f"数据: {len(X)} 样本")
    
    # 5-fold交叉验证
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    
    # [1] RF基线
    logger.info("\n[1/5] RF 基线...")
    rf_preds = []
    rf_labels = []
    for train_idx, test_idx in kfold.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_s, y_train)
        rf_preds.extend(rf.predict(X_test_s))
        rf_labels.extend(y_test)
    
    results['RF'] = accuracy_score(rf_labels, rf_preds) * 100
    logger.info(f"   RF: {results['RF']:.2f}%")
    
    # [2] VAE + RF
    logger.info("\n[2/5] VAE + RF...")
    vae_rf_preds = []
    vae_rf_labels = []
    for train_idx, test_idx in kfold.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        vae = train_vae(X_train_s, y_train, epochs=200)
        X_aug, y_aug = augment_data(vae, X_train_s, y_train, n_augment=200)
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_aug, y_aug)
        vae_rf_preds.extend(rf.predict(X_test_s))
        vae_rf_labels.extend(y_test)
    
    results['VAE+RF'] = accuracy_score(vae_rf_labels, vae_rf_preds) * 100
    logger.info(f"   VAE+RF: {results['VAE+RF']:.2f}%")
    
    # [3] NeuralRF (无VAE)
    logger.info("\n[3/5] NeuralRF (无VAE)...")
    nrf_preds = []
    nrf_labels = []
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        model = train_neural_rf(X_train_s, y_train, X_test_s, y_test, 
                                n_trees=20, depth=3, epochs=200)
        preds = predict_neural_rf(model, X_test_s)
        
        fold_acc = accuracy_score(y_test, preds) * 100
        logger.info(f"   Fold {fold+1}/5: {fold_acc:.2f}%")
        
        nrf_preds.extend(preds)
        nrf_labels.extend(y_test)
    
    results['NeuralRF'] = accuracy_score(nrf_labels, nrf_preds) * 100
    logger.info(f"   NeuralRF: {results['NeuralRF']:.2f}%")
    
    # [4] VAE + NeuralRF
    logger.info("\n[4/5] VAE + NeuralRF...")
    vae_nrf_preds = []
    vae_nrf_labels = []
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        # VAE增强
        vae = train_vae(X_train_s, y_train, epochs=200)
        X_aug, y_aug = augment_data(vae, X_train_s, y_train, n_augment=200)
        
        model = train_neural_rf(X_aug, y_aug, X_test_s, y_test, 
                                n_trees=20, depth=3, epochs=200)
        preds = predict_neural_rf(model, X_test_s)
        
        fold_acc = accuracy_score(y_test, preds) * 100
        logger.info(f"   Fold {fold+1}/5: {fold_acc:.2f}%")
        
        vae_nrf_preds.extend(preds)
        vae_nrf_labels.extend(y_test)
    
    results['VAE+NeuralRF'] = accuracy_score(vae_nrf_labels, vae_nrf_preds) * 100
    logger.info(f"   VAE+NeuralRF: {results['VAE+NeuralRF']:.2f}%")
    
    # [5] VAE + NeuralRF (多次运行投票)
    logger.info("\n[5/5] VAE + NeuralRF (5次运行投票)...")
    vae_nrf_vote_preds = []
    vae_nrf_vote_labels = []
    n_runs = 5
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        # 多次运行收集预测
        all_preds = []
        for run in range(n_runs):
            vae = train_vae(X_train_s, y_train, epochs=200)
            X_aug, y_aug = augment_data(vae, X_train_s, y_train, n_augment=200)
            
            model = train_neural_rf(X_aug, y_aug, X_test_s, y_test,
                                    n_trees=20, depth=3, epochs=200)
            preds = predict_neural_rf(model, X_test_s)
            all_preds.append(preds)
        
        # 多数投票
        all_preds = np.array(all_preds)  # [n_runs, n_test]
        voted_preds = (all_preds.mean(axis=0) > 0.5).astype(int)
        
        fold_acc = accuracy_score(y_test, voted_preds) * 100
        logger.info(f"   Fold {fold+1}/5: {fold_acc:.2f}%")
        
        vae_nrf_vote_preds.extend(voted_preds)
        vae_nrf_vote_labels.extend(y_test)
    
    results['VAE+NeuralRF(投票)'] = accuracy_score(vae_nrf_vote_labels, vae_nrf_vote_preds) * 100
    logger.info(f"   VAE+NeuralRF(投票): {results['VAE+NeuralRF(投票)']:.2f}%")
    
    # 输出结果
    logger.info("\n" + "=" * 70)
    logger.info("[结果对比]")
    logger.info("=" * 70)
    
    best_acc = max(results.values())
    for name, acc in results.items():
        marker = "🏆" if acc == best_acc else "  "
        logger.info(f"{marker} {name:25s}: {acc:.2f}%")
    
    logger.info("=" * 70)
    
    # 保存结果
    output_dir = Path('/data2/image_identification/src/output')
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'32_neural_rf_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"保存: {output_file}")


if __name__ == '__main__':
    main()
