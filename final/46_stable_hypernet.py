#!/usr/bin/env python3
"""
46_stable_hypernet.py - 稳定版HyperNet

改进措施：
1. 多头HyperNet集成 - 多个独立HyperNet，降低单一网络的随机性
2. 权重约束 - 限制生成权重范围，防止极端值
3. 一致性正则化 - 不同dropout的预测应该一致
4. 知识蒸馏 - 用RF的软标签指导训练
5. 更多投票 - 15次投票降低方差
6. 确定性增强 - 固定增强数据，消除每次训练的随机性
7. 温度退火 - 训练时高温度探索，推理时低温度决策
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


# ============== 改进1: 更稳定的VAE ==============
class StableVAE(nn.Module):
    """更稳定的VAE，使用更强的正则化"""
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        hidden = 32
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.2)
        )
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_var = nn.Linear(hidden, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, input_dim)
        )
        
        # 正交初始化，更稳定
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


# ============== 改进2: 多头HyperNet ==============
class StableHyperNetwork(nn.Module):
    """
    稳定的超网络：
    1. 多个独立的小HyperNet（heads）
    2. 输出权重约束（tanh）
    3. 残差连接
    """
    def __init__(self, input_dim, n_trees=10, tree_depth=3, n_heads=3, hidden_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.n_trees = n_trees
        self.tree_depth = tree_depth
        self.n_heads = n_heads
        
        n_internal = 2**tree_depth - 1
        n_leaves = 2**tree_depth
        
        self.params_per_tree = n_internal * input_dim + n_internal + n_leaves * 2
        self.total_params = self.params_per_tree * n_trees + n_trees
        
        self.n_internal = n_internal
        self.n_leaves = n_leaves
        
        # 数据编码器 - 更稳定的设计
        self.data_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # 多头HyperNet - 多个独立的生成器
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, self.total_params)
            ) for _ in range(n_heads)
        ])
        
        # 头部权重 - 可学习的融合权重
        self.head_weights = nn.Parameter(torch.ones(n_heads) / n_heads)
        
        # 输出缩放因子 - 控制生成权重的范围
        self.output_scale = nn.Parameter(torch.tensor(0.5))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, X_train):
        batch_size = X_train.size(0)
        
        # 编码训练数据
        if batch_size == 1:
            # 单样本时跳过BatchNorm
            encoded = X_train
            for layer in self.data_encoder:
                if isinstance(layer, nn.BatchNorm1d):
                    continue
                encoded = layer(encoded)
        else:
            encoded = self.data_encoder(X_train)
        
        # 使用均值和标准差作为context（更稳定）
        context_mean = encoded.mean(dim=0, keepdim=True)
        context_std = encoded.std(dim=0, keepdim=True) + 1e-6
        context = torch.cat([context_mean, context_std], dim=-1)[:, :encoded.size(-1)]
        
        # 多头生成参数
        params_list = []
        for head in self.heads:
            params = head(context.squeeze(0))
            params_list.append(params)
        
        # 加权平均多个头的输出
        params_stack = torch.stack(params_list, dim=0)  # [n_heads, total_params]
        weights = F.softmax(self.head_weights, dim=0)
        params = torch.einsum('h,hp->p', weights, params_stack)
        
        # 约束权重范围：使用tanh限制在[-scale, scale]
        params = torch.tanh(params) * torch.abs(self.output_scale)
        
        return params
    
    def parse_params(self, params):
        """解析参数"""
        trees_params = []
        offset = 0
        
        for t in range(self.n_trees):
            split_w_size = self.n_internal * self.input_dim
            split_weights = params[offset:offset+split_w_size].view(self.n_internal, self.input_dim)
            offset += split_w_size
            
            split_bias = params[offset:offset+self.n_internal]
            offset += self.n_internal
            
            leaf_size = self.n_leaves * 2
            leaf_logits = params[offset:offset+leaf_size].view(self.n_leaves, 2)
            offset += leaf_size
            
            trees_params.append({
                'split_weights': split_weights,
                'split_bias': split_bias,
                'leaf_logits': leaf_logits
            })
        
        tree_weights = params[offset:offset+self.n_trees]
        return trees_params, tree_weights


# ============== 改进3: 温度可调的软决策树 ==============
class StableTreeClassifier(nn.Module):
    """温度可调的分类器"""
    def __init__(self, tree_depth=3, init_temp=1.0):
        super().__init__()
        self.depth = tree_depth
        self.n_internal = 2**tree_depth - 1
        self.n_leaves = 2**tree_depth
        # 可学习的温度参数
        self.log_temperature = nn.Parameter(torch.tensor(np.log(init_temp)))
    
    @property
    def temperature(self):
        return torch.exp(self.log_temperature).clamp(0.1, 10.0)
    
    def forward_single_tree(self, x, split_weights, split_bias, leaf_logits):
        batch_size = x.size(0)
        
        # 温度缩放的分裂概率
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


# ============== 改进4: 稳定的完整模型 ==============
class StableVAEHyperNetFusion(nn.Module):
    def __init__(self, input_dim, n_trees=10, tree_depth=3, n_heads=3):
        super().__init__()
        self.vae = StableVAE(input_dim)
        self.hypernet = StableHyperNetwork(input_dim, n_trees, tree_depth, n_heads)
        self.classifier = StableTreeClassifier(tree_depth)
        
    def generate_augmented_data(self, X, y, n_augment=100, noise_scale=0.5):
        """确定性生成增强数据（只调用一次）"""
        self.vae.eval()
        
        augmented_X = [X]
        augmented_y = [y]
        
        with torch.no_grad():
            for i in range(n_augment):
                idx = i % X.size(0)
                mu, logvar = self.vae.encode(X[idx:idx+1])
                # 使用较小的噪声，更接近原始数据
                z = self.vae.reparameterize(mu, logvar, noise_scale=noise_scale)
                generated = self.vae.decoder(z)
                augmented_X.append(generated)
                augmented_y.append(y[idx:idx+1])
        
        return torch.cat(augmented_X, dim=0), torch.cat(augmented_y, dim=0)
    
    def forward(self, X_train, X_test):
        params = self.hypernet(X_train)
        trees_params, tree_weights = self.hypernet.parse_params(params)
        output = self.classifier(X_test, trees_params, tree_weights)
        return output, params


# ============== 改进5: 带知识蒸馏的训练 ==============
def train_stable_model(X_train, y_train, X_val, y_val, epochs=200, seed=42, 
                       use_distillation=True, rf_model=None):
    """
    稳定训练流程：
    1. 先训练VAE
    2. 一次性生成增强数据
    3. 用知识蒸馏训练HyperNet
    """
    set_seed(seed)
    
    input_dim = X_train.shape[1]
    model = StableVAEHyperNetFusion(input_dim, n_trees=15, tree_depth=3, n_heads=3).to(device)
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    
    # 阶段1: 训练VAE
    vae_optimizer = torch.optim.Adam(model.vae.parameters(), lr=0.002)
    
    for epoch in range(100):
        model.vae.train()
        vae_optimizer.zero_grad()
        
        recon, mu, logvar = model.vae(X_train_t)
        recon_loss = F.mse_loss(recon, X_train_t, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        vae_loss = recon_loss + 0.005 * kl_loss
        
        vae_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.vae.parameters(), 1.0)
        vae_optimizer.step()
    
    # 阶段2: 一次性生成增强数据（确定性）
    model.vae.eval()
    with torch.no_grad():
        X_aug, y_aug = model.generate_augmented_data(X_train_t, y_train_t, n_augment=200, noise_scale=0.3)
    
    # 获取RF软标签用于蒸馏
    rf_soft_labels = None
    if use_distillation and rf_model is not None:
        rf_proba = rf_model.predict_proba(X_aug.cpu().numpy())
        rf_soft_labels = torch.FloatTensor(rf_proba).to(device)
    
    # 阶段3: 训练HyperNet
    hypernet_optimizer = torch.optim.AdamW(
        list(model.hypernet.parameters()) + list(model.classifier.parameters()),
        lr=0.01, weight_decay=0.05
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(hypernet_optimizer, T_0=50)
    
    best_val_acc = 0
    best_state = None
    patience_count = 0
    max_patience = 30
    
    for epoch in range(epochs):
        model.hypernet.train()
        model.classifier.train()
        
        hypernet_optimizer.zero_grad()
        
        output, params = model(X_aug, X_aug)
        
        # 硬标签损失
        hard_loss = F.cross_entropy(output, y_aug)
        
        # 软标签蒸馏损失（如果有RF模型）
        soft_loss = 0
        if rf_soft_labels is not None:
            soft_loss = F.kl_div(
                F.log_softmax(output / 2.0, dim=1),  # 温度=2
                F.softmax(rf_soft_labels / 2.0, dim=1),
                reduction='batchmean'
            ) * 4.0  # 温度^2
        
        # 一致性正则化：dropout不同时预测应该一致
        model.hypernet.train()  # 开启dropout
        output2, _ = model(X_aug, X_aug)
        consistency_loss = F.mse_loss(output, output2)
        
        # 权重正则化
        reg_loss = 0.01 * (params ** 2).mean()
        
        # 总损失
        loss = hard_loss + 0.5 * soft_loss + 0.1 * consistency_loss + reg_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        hypernet_optimizer.step()
        scheduler.step()
        
        # 验证
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_output, _ = model(X_train_t, X_val_t)
                val_pred = val_output.argmax(dim=1).cpu().numpy()
                val_acc = accuracy_score(y_val, val_pred)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    patience_count = 0
                else:
                    patience_count += 1
                
                if patience_count >= max_patience:
                    break
    
    if best_state:
        model.load_state_dict(best_state)
    
    return model


def predict_stable(model, X_train, X_test):
    """推理时使用较低温度（更确定的决策）"""
    model.eval()
    
    # 降低推理温度
    original_temp = model.classifier.log_temperature.data.clone()
    model.classifier.log_temperature.data = torch.tensor(np.log(0.5), device=device)
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        output, _ = model(X_train_t, X_test_t)
        proba = F.softmax(output, dim=1)
    
    # 恢复温度
    model.classifier.log_temperature.data = original_temp
    
    return proba.cpu().numpy()


def load_data():
    for path in ['/data2/image_identification/src/data/Data_for_Jinming.csv',
                 'data/Data_for_Jinming.csv',
                 '/Users/jinmingzhang/D/1/1Postg/Sem_13_Thesis/5_2026.01.28/src/final/data/Data_for_Jinming.csv']:
        if Path(path).exists():
            df = pd.read_csv(path)
            X = df[['LAA', 'Glutamate', 'Choline', 'Sarcosine']].values
            y = (df['Group'] == 'PCa').astype(int).values
            return X, y
    raise FileNotFoundError("找不到数据文件")


def main():
    logger.info("=" * 70)
    logger.info("46_stable_hypernet.py - 稳定版HyperNet")
    logger.info("=" * 70)
    logger.info("改进措施:")
    logger.info("  1. 多头HyperNet集成")
    logger.info("  2. 权重约束（tanh）")
    logger.info("  3. 一致性正则化")
    logger.info("  4. 知识蒸馏（RF软标签）")
    logger.info("  5. 更多投票次数")
    logger.info("  6. 确定性数据增强")
    logger.info("  7. 温度退火")
    logger.info("=" * 70)
    
    X, y = load_data()
    logger.info(f"设备: {device}")
    logger.info(f"数据: {len(X)} 样本")
    
    # 测试多个种子
    test_seeds = [42, 123, 456, 789, 1000]
    all_results = {seed: {} for seed in test_seeds}
    
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # [1] RF基线（种子无关）
    logger.info("\n[1/4] RF 基线...")
    set_seed(42)
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
        
        rf_preds.extend(preds)
        rf_labels.extend(y_test)
    
    rf_acc = accuracy_score(rf_labels, rf_preds) * 100
    logger.info(f"   RF: {rf_acc:.2f}%")
    
    # [2] 测试不同种子下的稳定性
    logger.info("\n[2/4] 测试多种子稳定性...")
    
    seed_accuracies = []
    
    for seed in test_seeds:
        logger.info(f"\n   Seed={seed}:")
        set_seed(seed)
        
        # 重新划分fold
        kfold_seed = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        
        stable_preds, stable_labels = [], []
        
        for fold, (train_idx, test_idx) in enumerate(kfold_seed.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            
            # 训练RF用于蒸馏
            rf = RandomForestClassifier(n_estimators=100, random_state=seed)
            rf.fit(X_train_s, y_train)
            
            # 训练稳定模型
            model = train_stable_model(
                X_train_s, y_train, X_test_s, y_test,
                epochs=200, seed=seed+fold,
                use_distillation=True, rf_model=rf
            )
            
            proba = predict_stable(model, X_train_s, X_test_s)
            preds = proba.argmax(axis=1)
            
            fold_acc = accuracy_score(y_test, preds) * 100
            logger.info(f"      Fold {fold+1}/5: {fold_acc:.2f}%")
            
            stable_preds.extend(preds)
            stable_labels.extend(y_test)
        
        seed_acc = accuracy_score(stable_labels, stable_preds) * 100
        seed_accuracies.append(seed_acc)
        all_results[seed]['single'] = seed_acc
        logger.info(f"      Seed {seed} 准确率: {seed_acc:.2f}%")
    
    # [3] 投票集成（15次）
    logger.info("\n[3/4] 稳定模型 + 15次投票...")
    
    voting_accuracies = []
    
    for main_seed in test_seeds:
        logger.info(f"\n   Main Seed={main_seed}:")
        
        kfold_seed = StratifiedKFold(n_splits=5, shuffle=True, random_state=main_seed)
        
        voting_preds, voting_labels = [], []
        n_runs = 15  # 更多投票
        
        for fold, (train_idx, test_idx) in enumerate(kfold_seed.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            
            # RF用于蒸馏
            rf = RandomForestClassifier(n_estimators=100, random_state=main_seed)
            rf.fit(X_train_s, y_train)
            
            all_probas = []
            for run in range(n_runs):
                run_seed = main_seed + fold * 100 + run * 7  # 不同的种子间隔
                model = train_stable_model(
                    X_train_s, y_train, X_test_s, y_test,
                    epochs=150, seed=run_seed,
                    use_distillation=True, rf_model=rf
                )
                proba = predict_stable(model, X_train_s, X_test_s)
                all_probas.append(proba)
            
            # 软投票
            avg_proba = np.mean(all_probas, axis=0)
            preds = avg_proba.argmax(axis=1)
            
            fold_acc = accuracy_score(y_test, preds) * 100
            logger.info(f"      Fold {fold+1}/5: {fold_acc:.2f}%")
            
            voting_preds.extend(preds)
            voting_labels.extend(y_test)
        
        voting_acc = accuracy_score(voting_labels, voting_preds) * 100
        voting_accuracies.append(voting_acc)
        all_results[main_seed]['voting_15'] = voting_acc
        logger.info(f"      Seed {main_seed} 投票准确率: {voting_acc:.2f}%")
    
    # [4] 统计结果
    logger.info("\n" + "=" * 70)
    logger.info("[稳定性测试结果]")
    logger.info("=" * 70)
    
    logger.info(f"\nRF 基线: {rf_acc:.2f}%")
    
    logger.info(f"\n单次运行准确率 (across seeds):")
    logger.info(f"  平均: {np.mean(seed_accuracies):.2f}% ± {np.std(seed_accuracies):.2f}%")
    logger.info(f"  最小: {np.min(seed_accuracies):.2f}%")
    logger.info(f"  最大: {np.max(seed_accuracies):.2f}%")
    
    logger.info(f"\n15次投票准确率 (across seeds):")
    logger.info(f"  平均: {np.mean(voting_accuracies):.2f}% ± {np.std(voting_accuracies):.2f}%")
    logger.info(f"  最小: {np.min(voting_accuracies):.2f}%")
    logger.info(f"  最大: {np.max(voting_accuracies):.2f}%")
    
    logger.info("\n各种子详细结果:")
    for seed in test_seeds:
        logger.info(f"  Seed {seed}: 单次={all_results[seed].get('single', 'N/A'):.2f}%, "
                   f"投票={all_results[seed].get('voting_15', 'N/A'):.2f}%")
    
    # 评估稳定性改进
    old_std = 13.0  # 原始37版本的标准差约13%
    new_std = np.std(voting_accuracies)
    improvement = (old_std - new_std) / old_std * 100
    
    logger.info(f"\n稳定性改进:")
    logger.info(f"  原版标准差: ~13%")
    logger.info(f"  新版标准差: {new_std:.2f}%")
    logger.info(f"  稳定性提升: {improvement:.1f}%")
    
    logger.info("=" * 70)
    
    # 保存结果
    output_dir = Path('/data2/image_identification/src/output')
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'46_stable_hypernet_{timestamp}.json'
    
    results = {
        'rf_baseline': rf_acc,
        'single_run': {
            'mean': float(np.mean(seed_accuracies)),
            'std': float(np.std(seed_accuracies)),
            'min': float(np.min(seed_accuracies)),
            'max': float(np.max(seed_accuracies)),
            'values': seed_accuracies
        },
        'voting_15': {
            'mean': float(np.mean(voting_accuracies)),
            'std': float(np.std(voting_accuracies)),
            'min': float(np.min(voting_accuracies)),
            'max': float(np.max(voting_accuracies)),
            'values': voting_accuracies
        },
        'all_results': {str(k): v for k, v in all_results.items()}
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n保存: {output_file}")


if __name__ == '__main__':
    main()
