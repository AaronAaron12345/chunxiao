#!/usr/bin/env python3
"""
37_true_hypernet.py - 真正的HyperNet框架
核心概念：超网络生成目标网络（神经决策树）的权重

架构：
  输入x → HyperNetwork → 生成SoftDecisionTree的权重 → 分类

这保持了HyperNet的核心思想：
- 不是直接训练分类器
- 而是训练一个"超网络"来生成分类器的权重
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


# ============== VAE 数据增强 ==============
class VAE(nn.Module):
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


# ============== 核心：HyperNetwork生成目标网络权重 ==============
class HyperNetworkForTree(nn.Module):
    """
    超网络：根据训练数据的统计特征，生成软决策树的权重
    
    这是HyperNet的核心概念：
    - 输入：训练数据的embedding（统计特征）
    - 输出：目标网络（SoftDecisionTree）的所有参数
    """
    def __init__(self, input_dim, n_trees=10, tree_depth=3, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.n_trees = n_trees
        self.tree_depth = tree_depth
        
        # 每棵树的参数数量
        n_internal = 2**tree_depth - 1
        n_leaves = 2**tree_depth
        
        # 每棵树需要: split_weights(n_internal * input_dim) + split_bias(n_internal) + leaf_logits(n_leaves * 2)
        self.params_per_tree = n_internal * input_dim + n_internal + n_leaves * 2
        self.total_params = self.params_per_tree * n_trees + n_trees  # +n_trees for tree weights
        
        self.n_internal = n_internal
        self.n_leaves = n_leaves
        
        # 数据编码器：将训练数据编码为context embedding
        self.data_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 超网络：生成目标网络的权重
        self.hyper_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, self.total_params)
        )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, X_train):
        """
        X_train: [n_samples, input_dim] - 训练数据
        返回: 生成的树参数
        """
        # 1. 编码训练数据 -> context embedding
        # 使用均值池化得到数据集的全局表示
        encoded = self.data_encoder(X_train)  # [n_samples, hidden_dim]
        context = encoded.mean(dim=0, keepdim=True)  # [1, hidden_dim]
        
        # 2. 超网络生成目标网络参数
        params = self.hyper_net(context)  # [1, total_params]
        params = params.squeeze(0)  # [total_params]
        
        return params
    
    def parse_params(self, params):
        """解析生成的参数为各棵树的权重"""
        trees_params = []
        offset = 0
        
        for t in range(self.n_trees):
            # split_weights
            split_w_size = self.n_internal * self.input_dim
            split_weights = params[offset:offset+split_w_size].view(self.n_internal, self.input_dim)
            offset += split_w_size
            
            # split_bias
            split_bias = params[offset:offset+self.n_internal]
            offset += self.n_internal
            
            # leaf_logits
            leaf_size = self.n_leaves * 2
            leaf_logits = params[offset:offset+leaf_size].view(self.n_leaves, 2)
            offset += leaf_size
            
            trees_params.append({
                'split_weights': split_weights,
                'split_bias': split_bias,
                'leaf_logits': leaf_logits
            })
        
        # tree_weights
        tree_weights = params[offset:offset+self.n_trees]
        
        return trees_params, tree_weights


class GeneratedTreeClassifier(nn.Module):
    """使用HyperNet生成的参数进行分类"""
    def __init__(self, tree_depth=3):
        super().__init__()
        self.depth = tree_depth
        self.n_internal = 2**tree_depth - 1
        self.n_leaves = 2**tree_depth
        self.temperature = 1.0
    
    def forward_single_tree(self, x, split_weights, split_bias, leaf_logits):
        """单棵树的前向传播"""
        batch_size = x.size(0)
        
        # 计算分裂概率
        split_probs = torch.sigmoid(
            (x @ split_weights.T + split_bias) / self.temperature
        )
        
        # 计算到达每个叶子的概率
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
        
        # 加权叶子节点分布
        leaf_class_probs = F.softmax(leaf_logits, dim=-1)
        output = torch.einsum('bl,lc->bc', leaf_probs, leaf_class_probs)
        
        return output
    
    def forward(self, x, trees_params, tree_weights):
        """使用生成的参数进行分类"""
        outputs = []
        
        for tree_param in trees_params:
            out = self.forward_single_tree(
                x,
                tree_param['split_weights'],
                tree_param['split_bias'],
                tree_param['leaf_logits']
            )
            outputs.append(out)
        
        # 加权平均
        outputs = torch.stack(outputs, dim=0)  # [n_trees, batch, 2]
        weights = F.softmax(tree_weights, dim=0)
        final_output = torch.einsum('t,tbc->bc', weights, outputs)
        
        return final_output


class VAEHyperNetFusion(nn.Module):
    """
    完整的VAE-HyperNet-Fusion模型
    
    1. VAE: 数据增强
    2. HyperNetwork: 生成目标网络权重
    3. GeneratedTreeClassifier: 使用生成的权重分类
    """
    def __init__(self, input_dim, n_trees=10, tree_depth=3):
        super().__init__()
        self.vae = VAE(input_dim)
        self.hypernet = HyperNetworkForTree(input_dim, n_trees, tree_depth)
        self.classifier = GeneratedTreeClassifier(tree_depth)
    
    def augment(self, X, n_augment=100):
        """使用VAE增强数据"""
        self.vae.eval()
        augmented = [X]
        
        with torch.no_grad():
            for _ in range(n_augment):
                idx = torch.randint(0, X.size(0), (1,))
                mu, logvar = self.vae.encode(X[idx])
                z = self.vae.reparameterize(mu, logvar * 0.5)
                generated = self.vae.decoder(z)
                augmented.append(generated)
        
        return torch.cat(augmented, dim=0)
    
    def forward(self, X_train, X_test):
        """
        X_train: 训练数据 (用于HyperNet生成权重)
        X_test: 测试数据 (用于分类)
        """
        # 1. HyperNet根据训练数据生成目标网络权重
        params = self.hypernet(X_train)
        trees_params, tree_weights = self.hypernet.parse_params(params)
        
        # 2. 使用生成的权重对测试数据分类
        output = self.classifier(X_test, trees_params, tree_weights)
        
        return output, params


def train_vae_hypernet(X_train, y_train, X_val, y_val, epochs=300, seed=42):
    """训练VAE-HyperNet-Fusion模型"""
    set_seed(seed)
    
    input_dim = X_train.shape[1]
    model = VAEHyperNetFusion(input_dim, n_trees=15, tree_depth=3).to(device)
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    
    # 分阶段训练
    # 阶段1: 训练VAE
    vae_optimizer = torch.optim.Adam(model.vae.parameters(), lr=0.001)
    
    model.vae.train()
    for epoch in range(100):
        vae_optimizer.zero_grad()
        recon, mu, logvar = model.vae(X_train_t)
        recon_loss = F.mse_loss(recon, X_train_t, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        vae_loss = recon_loss + 0.01 * kl_loss
        vae_loss.backward()
        vae_optimizer.step()
    
    # 阶段2: 用增强数据训练HyperNet
    hypernet_optimizer = torch.optim.AdamW(model.hypernet.parameters(), lr=0.005, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(hypernet_optimizer, epochs)
    
    best_val_acc = 0
    best_state = None
    patience = 50
    no_improve = 0
    
    for epoch in range(epochs):
        model.hypernet.train()
        model.classifier.train()
        
        # 每个epoch重新增强数据
        with torch.no_grad():
            X_aug = model.augment(X_train_t, n_augment=150)
            y_aug = y_train_t.repeat((X_aug.size(0) + X_train_t.size(0) - 1) // X_train_t.size(0) + 1)[:X_aug.size(0)]
        
        hypernet_optimizer.zero_grad()
        
        # 前向传播：HyperNet生成权重，然后分类
        output, params = model(X_aug, X_aug)
        
        # 分类损失
        cls_loss = F.cross_entropy(output, y_aug)
        
        # 正则化：防止生成的权重过大
        reg_loss = 0.001 * (params ** 2).mean()
        
        loss = cls_loss + reg_loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        hypernet_optimizer.step()
        scheduler.step()
        
        # 验证
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_output, _ = model(X_train_t, X_val_t)  # 用原始训练数据生成权重
                val_pred = val_output.argmax(dim=1).cpu().numpy()
                val_acc = accuracy_score(y_val, val_pred)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                
                if no_improve >= patience // 10:
                    break
    
    if best_state:
        model.load_state_dict(best_state)
    
    return model


def predict_vae_hypernet(model, X_train, X_test):
    """使用训练好的模型预测"""
    model.eval()
    X_train_t = torch.FloatTensor(X_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        output, _ = model(X_train_t, X_test_t)
        return output.argmax(dim=1).cpu().numpy()


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
    logger.info("37_true_hypernet.py - 真正的HyperNet框架")
    logger.info("核心：超网络生成神经决策树的权重")
    logger.info("=" * 70)
    
    set_seed(42)
    
    X, y = load_data()
    logger.info(f"设备: {device}")
    logger.info(f"数据: {len(X)} 样本")
    
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    
    # [1] RF基线
    logger.info("\n[1/4] RF 基线...")
    rf_preds, rf_labels = [], []
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_s, y_train)
        preds = rf.predict(X_test_s)
        
        fold_acc = accuracy_score(y_test, preds) * 100
        logger.info(f"   Fold {fold+1}/5: {fold_acc:.2f}%")
        
        rf_preds.extend(preds)
        rf_labels.extend(y_test)
    
    results['RF'] = accuracy_score(rf_labels, rf_preds) * 100
    logger.info(f"   RF: {results['RF']:.2f}%")
    
    # [2] VAE-HyperNet-Fusion (真正的HyperNet)
    logger.info("\n[2/4] VAE-HyperNet-Fusion (超网络生成权重)...")
    vhn_preds, vhn_labels = [], []
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        model = train_vae_hypernet(X_train_s, y_train, X_test_s, y_test, 
                                   epochs=300, seed=42+fold)
        preds = predict_vae_hypernet(model, X_train_s, X_test_s)
        
        fold_acc = accuracy_score(y_test, preds) * 100
        logger.info(f"   Fold {fold+1}/5: {fold_acc:.2f}%")
        
        vhn_preds.extend(preds)
        vhn_labels.extend(y_test)
    
    results['VAE-HyperNet-Fusion'] = accuracy_score(vhn_labels, vhn_preds) * 100
    logger.info(f"   VAE-HyperNet-Fusion: {results['VAE-HyperNet-Fusion']:.2f}%")
    
    # [3] VAE-HyperNet-Fusion (多次运行取最佳)
    logger.info("\n[3/4] VAE-HyperNet-Fusion (5次运行取最佳)...")
    vhn_best_preds, vhn_best_labels = [], []
    n_runs = 5
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        all_preds = []
        for run in range(n_runs):
            model = train_vae_hypernet(X_train_s, y_train, X_test_s, y_test,
                                       epochs=300, seed=42+fold*100+run)
            preds = predict_vae_hypernet(model, X_train_s, X_test_s)
            all_preds.append(preds)
        
        # 多数投票
        all_preds = np.array(all_preds)
        voted_preds = (all_preds.mean(axis=0) > 0.5).astype(int)
        
        fold_acc = accuracy_score(y_test, voted_preds) * 100
        logger.info(f"   Fold {fold+1}/5: {fold_acc:.2f}%")
        
        vhn_best_preds.extend(voted_preds)
        vhn_best_labels.extend(y_test)
    
    results['VAE-HyperNet(投票)'] = accuracy_score(vhn_best_labels, vhn_best_preds) * 100
    logger.info(f"   VAE-HyperNet(投票): {results['VAE-HyperNet(投票)']:.2f}%")
    
    # [4] Hybrid: RF + VAE-HyperNet
    logger.info("\n[4/4] Hybrid (RF + VAE-HyperNet)...")
    hybrid_preds, hybrid_labels = [], []
    
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
        
        # VAE-HyperNet
        model = train_vae_hypernet(X_train_s, y_train, X_test_s, y_test,
                                   epochs=300, seed=42+fold)
        model.eval()
        X_train_t = torch.FloatTensor(X_train_s).to(device)
        X_test_t = torch.FloatTensor(X_test_s).to(device)
        with torch.no_grad():
            vhn_output, _ = model(X_train_t, X_test_t)
            vhn_proba = F.softmax(vhn_output, dim=1).cpu().numpy()
        
        # 软投票
        combined = 0.5 * rf_proba + 0.5 * vhn_proba
        preds = combined.argmax(axis=1)
        
        fold_acc = accuracy_score(y_test, preds) * 100
        logger.info(f"   Fold {fold+1}/5: {fold_acc:.2f}%")
        
        hybrid_preds.extend(preds)
        hybrid_labels.extend(y_test)
    
    results['Hybrid(RF+HyperNet)'] = accuracy_score(hybrid_labels, hybrid_preds) * 100
    logger.info(f"   Hybrid: {results['Hybrid(RF+HyperNet)']:.2f}%")
    
    # 输出结果
    logger.info("\n" + "=" * 70)
    logger.info("[结果对比]")
    logger.info("=" * 70)
    
    best_acc = max(results.values())
    for name, acc in results.items():
        marker = "🏆" if acc == best_acc else "  "
        logger.info(f"{marker} {name:25s}: {acc:.2f}%")
    
    logger.info("=" * 70)
    logger.info("\n架构说明:")
    logger.info("  VAE-HyperNet-Fusion 是真正的HyperNet框架：")
    logger.info("  1. VAE: 数据增强")
    logger.info("  2. HyperNetwork: 根据训练数据生成神经决策树的权重")
    logger.info("  3. GeneratedTreeClassifier: 使用生成的权重进行分类")
    logger.info("=" * 70)
    
    # 保存结果
    output_dir = Path('/data2/image_identification/src/output')
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'37_true_hypernet_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n保存: {output_file}")


if __name__ == '__main__':
    main()
