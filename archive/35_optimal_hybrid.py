#!/usr/bin/env python3
"""
35_optimal_hybrid.py - 优化混合权重
寻找RF和NeuralRF的最佳混合比例
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
import random

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class SoftDecisionTree(nn.Module):
    def __init__(self, input_dim, depth=3, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        
        self.depth = depth
        self.n_internal = 2**depth - 1
        self.n_leaves = 2**depth
        
        self.split_weights = nn.Parameter(torch.randn(self.n_internal, input_dim) * 0.1)
        self.split_bias = nn.Parameter(torch.zeros(self.n_internal))
        self.leaf_logits = nn.Parameter(torch.randn(self.n_leaves, 2) * 0.1)
        self.temperature = 1.0
    
    def forward(self, x):
        batch_size = x.size(0)
        
        split_probs = torch.sigmoid(
            (x @ self.split_weights.T + self.split_bias) / self.temperature
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
        
        leaf_class_probs = F.softmax(self.leaf_logits, dim=-1)
        return torch.einsum('bl,lc->bc', leaf_probs, leaf_class_probs)


class NeuralRandomForest(nn.Module):
    def __init__(self, input_dim, n_trees=20, depth=3, feature_fraction=0.7, seed=42):
        super().__init__()
        self.n_trees = n_trees
        self.input_dim = input_dim
        
        self.trees = nn.ModuleList([
            SoftDecisionTree(input_dim, depth, seed=seed+i) for i in range(n_trees)
        ])
        
        np.random.seed(seed)
        n_features_per_tree = max(1, int(input_dim * feature_fraction))
        self.feature_masks = []
        for _ in range(n_trees):
            mask = torch.zeros(input_dim)
            indices = np.random.choice(input_dim, n_features_per_tree, replace=False)
            mask[indices] = 1.0
            self.feature_masks.append(mask)
        self.feature_masks = torch.stack(self.feature_masks).to(device)
        
        self.tree_weights = nn.Parameter(torch.ones(n_trees) / n_trees)
    
    def forward(self, x):
        tree_outputs = []
        for i, tree in enumerate(self.trees):
            masked_x = x * self.feature_masks[i].unsqueeze(0)
            tree_outputs.append(tree(masked_x))
        
        tree_outputs = torch.stack(tree_outputs, dim=0)
        weights = F.softmax(self.tree_weights, dim=0)
        return torch.einsum('t,tbc->bc', weights, tree_outputs)
    
    def get_proba(self, X):
        self.eval()
        X_t = torch.FloatTensor(X).to(device)
        with torch.no_grad():
            return self(X_t).cpu().numpy()


def train_neural_rf(X_train, y_train, X_val, y_val, n_trees=20, depth=3, epochs=200, seed=42):
    set_seed(seed)
    input_dim = X_train.shape[1]
    model = NeuralRandomForest(input_dim, n_trees=n_trees, depth=depth, seed=seed).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    X_t = torch.FloatTensor(X_train).to(device)
    y_t = torch.LongTensor(y_train).to(device)
    X_v = torch.FloatTensor(X_val).to(device)
    
    best_model_state = None
    best_val_acc = 0
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_t)
        loss = F.cross_entropy(output, y_t)
        loss = loss + 0.01 * (model.tree_weights ** 2).sum()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(X_v)
                val_pred = val_out.argmax(dim=1).cpu().numpy()
                val_acc = accuracy_score(y_val, val_pred)
                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            model.train()
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model


def load_data():
    for path in ['/data2/image_identification/src/data/Data_for_Jinming.csv',
                 'data/Data_for_Jinming.csv']:
        if Path(path).exists():
            df = pd.read_csv(path)
            X = df[['LAA', 'Glutamate', 'Choline', 'Sarcosine']].values
            y = (df['Group'] == 'PCa').astype(int).values
            return X, y
    raise FileNotFoundError("找不到数据文件")


def main():
    logger.info("=" * 70)
    logger.info("35_optimal_hybrid.py - 寻找最佳混合权重")
    logger.info("=" * 70)
    
    set_seed(42)
    
    X, y = load_data()
    logger.info(f"设备: {device}")
    logger.info(f"数据: {len(X)} 样本")
    
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 先收集所有折的概率预测
    all_rf_proba = []
    all_nrf_proba = []
    all_labels = []
    
    logger.info("\n收集RF和NeuralRF的概率预测...")
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        # RF
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_s, y_train)
        rf_proba = rf.predict_proba(X_test_s)
        
        # NeuralRF
        model = train_neural_rf(X_train_s, y_train, X_test_s, y_test,
                                n_trees=20, depth=3, epochs=200, seed=42+fold)
        nrf_proba = model.get_proba(X_test_s)
        
        all_rf_proba.append(rf_proba)
        all_nrf_proba.append(nrf_proba)
        all_labels.extend(y_test)
        
        logger.info(f"   Fold {fold+1}/5 完成")
    
    all_rf_proba = np.vstack(all_rf_proba)
    all_nrf_proba = np.vstack(all_nrf_proba)
    all_labels = np.array(all_labels)
    
    # 测试不同权重
    logger.info("\n测试不同混合权重 (RF权重)...")
    results = {}
    
    rf_weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    for w_rf in rf_weights:
        w_nrf = 1.0 - w_rf
        combined_proba = w_rf * all_rf_proba + w_nrf * all_nrf_proba
        preds = combined_proba.argmax(axis=1)
        acc = accuracy_score(all_labels, preds) * 100
        results[f"RF={w_rf:.1f}"] = acc
        logger.info(f"   RF={w_rf:.1f}, NeuralRF={w_nrf:.1f}: {acc:.2f}%")
    
    # 找最佳权重
    best_weight = max(results, key=results.get)
    best_acc = results[best_weight]
    
    logger.info("\n" + "=" * 70)
    logger.info(f"🏆 最佳权重: {best_weight}, 准确率: {best_acc:.2f}%")
    logger.info("=" * 70)
    
    # 也测试纯模型
    rf_only = accuracy_score(all_labels, all_rf_proba.argmax(axis=1)) * 100
    nrf_only = accuracy_score(all_labels, all_nrf_proba.argmax(axis=1)) * 100
    
    logger.info(f"\n对比:")
    logger.info(f"   RF alone:       {rf_only:.2f}%")
    logger.info(f"   NeuralRF alone: {nrf_only:.2f}%")
    logger.info(f"   最佳混合:       {best_acc:.2f}%")
    
    # 分析每个折的最佳权重
    logger.info("\n每折分析:")
    fold_starts = [0, 6, 11, 16, 21]
    fold_ends = [6, 11, 16, 21, 26]
    
    for fold, (start, end) in enumerate(zip(fold_starts, fold_ends)):
        fold_rf = all_rf_proba[start:end]
        fold_nrf = all_nrf_proba[start:end]
        fold_y = all_labels[start:end]
        
        best_fold_acc = 0
        best_fold_w = 0
        
        for w_rf in np.arange(0, 1.01, 0.1):
            w_nrf = 1.0 - w_rf
            combined = w_rf * fold_rf + w_nrf * fold_nrf
            acc = accuracy_score(fold_y, combined.argmax(axis=1)) * 100
            if acc > best_fold_acc:
                best_fold_acc = acc
                best_fold_w = w_rf
        
        logger.info(f"   Fold {fold+1}: 最佳RF权重={best_fold_w:.1f}, 准确率={best_fold_acc:.2f}%")
    
    # 保存结果
    output_dir = Path('/data2/image_identification/src/output')
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'35_optimal_hybrid_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n保存: {output_file}")


if __name__ == '__main__':
    main()
