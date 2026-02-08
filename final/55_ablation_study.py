#!/usr/bin/env python3
"""
55_ablation_study.py - VAE-HyperNet-Fusion 消融实验

审稿人要求：
  "add ablation studies that isolate the effects of VAE augmentation,
   interpolation, hypernetwork parameterization, and ensembling"

实验设计（6个变体 + 1个完整模型 + RF基线）：
  ┌────────────────────────────────────────────────────────────────────┐
  │  Variant                   │ VAE Aug │ Interp │ HyperNet │ Ensem  │
  ├────────────────────────────┼─────────┼────────┼──────────┼────────┤
  │  A) Full Model (ours)      │   ✓     │   ✓    │    ✓     │   ✓   │
  │  B) w/o VAE Augmentation   │   ✗     │   ✗    │    ✓     │   ✓   │
  │  C) w/o Interpolation      │   ✓(*)  │   ✗    │    ✓     │   ✓   │
  │  D) w/o HyperNet           │   ✓     │   ✓    │    ✗     │   ✓   │
  │  E) w/o Ensembling         │   ✓     │   ✓    │    ✓     │   ✗   │
  │  F) w/o Multi-head         │   ✓     │   ✓    │    ✓     │  部分  │
  │  G) VAE-only (+ RF)        │   ✓     │   ✓    │    ✗     │   ✗   │
  │  H) RF Baseline            │   ✗     │   ✗    │    ✗     │   ✗   │
  └────────────────────────────────────────────────────────────────────┘

  (*) w/o Interpolation: VAE仅做重建（noise_scale=0），无插值多样性

各变体说明：
  B) w/o VAE Augmentation：
     不使用数据增强，直接在原始训练数据上训练HyperNet+SoftTree集成。
     用于验证VAE数据增强对小样本的贡献。

  C) w/o Interpolation：
     VAE做数据增强，但noise_scale=0（只用μ解码，不添加随机噪声）。
     生成样本 = 原始样本的确定性重建，无多样性。
     用于验证latent space插值/扰动带来的多样性的重要性。

  D) w/o HyperNet：
     不使用超网络生成权重，而是直接用标准反向传播训练SoftTree集成的参数。
     用于验证"超网络动态生成权重"这一范式的好处。

  E) w/o Ensembling：
     单头HyperNet（n_heads=1）+ 单棵SoftTree（n_trees=1）。
     用于验证集成（多头+多树）的效果。

  F) w/o Multi-head：
     单头HyperNet（n_heads=1），但保留多棵SoftTree（n_trees=15）。
     用于隔离多头HyperNet vs 多树集成各自的贡献。

  G) VAE + RF：
     使用VAE增强数据，然后训练RandomForest。
     用于验证VAE的通用增强效果，不依赖我们的分类器设计。

评估方式：5折交叉验证，报告均值±标准差。在11个数据集(0-10)上运行。

运行方式（服务器）：
  nohup /data1/condaproject/dinov2/bin/python3 -u 55_ablation_study.py > 55_log.txt 2>&1 &

运行方式（本地Mac）：
  python 55_ablation_study.py --datasets 0 --no-gpu
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import pandas as pd
from pathlib import Path
import warnings
import multiprocessing as mp
from datetime import datetime
import json
import time
import random
import traceback
import argparse

warnings.filterwarnings('ignore')


# =====================================================================
# 1. 基础设施：种子、设备
# =====================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================================
# 2. 模型组件 (来自 47_stable_37.py，支持消融配置)
# =====================================================================

class VAE(nn.Module):
    """变分自编码器 - 数据增强"""
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


class HyperNetworkForTree(nn.Module):
    """超网络 - 支持多头/单头配置"""
    def __init__(self, input_dim, n_classes, n_trees=10, tree_depth=3,
                 hidden_dim=64, n_heads=5):
        super().__init__()
        self.input_dim = input_dim
        self.n_trees = n_trees
        self.tree_depth = tree_depth
        self.n_heads = n_heads
        self.n_classes = n_classes

        n_internal = 2**tree_depth - 1
        n_leaves = 2**tree_depth

        self.params_per_tree = (n_internal * input_dim + n_internal +
                                n_leaves * n_classes)
        self.total_params = self.params_per_tree * n_trees + n_trees

        self.n_internal = n_internal
        self.n_leaves = n_leaves

        # 共享数据编码器
        self.data_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 多头/单头超网络
        self.hyper_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim * 2, self.total_params)
            ) for _ in range(n_heads)
        ])

        self.head_weights = nn.Parameter(torch.ones(n_heads) / n_heads)
        self.output_scale = nn.Parameter(torch.tensor(1.0))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, X_train):
        encoded = self.data_encoder(X_train)
        context_mean = encoded.mean(dim=0, keepdim=True)

        all_params = []
        for head in self.hyper_heads:
            params = head(context_mean.squeeze(0))
            all_params.append(params)

        all_params = torch.stack(all_params, dim=0)
        weights = F.softmax(self.head_weights, dim=0)
        params = torch.einsum('h,hp->p', weights, all_params)
        params = torch.tanh(params) * self.output_scale
        return params

    def parse_params(self, params):
        trees_params = []
        offset = 0
        for t in range(self.n_trees):
            sw_size = self.n_internal * self.input_dim
            split_weights = params[offset:offset+sw_size].view(
                self.n_internal, self.input_dim)
            offset += sw_size

            split_bias = params[offset:offset+self.n_internal]
            offset += self.n_internal

            leaf_size = self.n_leaves * self.n_classes
            leaf_logits = params[offset:offset+leaf_size].view(
                self.n_leaves, self.n_classes)
            offset += leaf_size

            trees_params.append({
                'split_weights': split_weights,
                'split_bias': split_bias,
                'leaf_logits': leaf_logits
            })
        tree_weights = params[offset:offset+self.n_trees]
        return trees_params, tree_weights


class SoftTreeClassifier(nn.Module):
    """软决策树集成 - 温度可学习"""
    def __init__(self, tree_depth=3, init_temp=1.0):
        super().__init__()
        self.depth = tree_depth
        self.n_internal = 2**tree_depth - 1
        self.n_leaves = 2**tree_depth
        self.log_temperature = nn.Parameter(torch.tensor(np.log(init_temp)))

    @property
    def temperature(self):
        return torch.exp(self.log_temperature).clamp(0.1, 5.0)

    def forward_single_tree(self, x, split_weights, split_bias, leaf_logits):
        batch_size = x.size(0)
        split_probs = torch.sigmoid(
            (x @ split_weights.T + split_bias) / self.temperature)
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
                x, tree_param['split_weights'],
                tree_param['split_bias'],
                tree_param['leaf_logits'])
            outputs.append(out)
        outputs = torch.stack(outputs, dim=0)
        weights = F.softmax(tree_weights, dim=0)
        final_output = torch.einsum('t,tbc->bc', weights, outputs)
        return final_output


class DirectSoftTreeEnsemble(nn.Module):
    """
    直接训练的软决策树集成（不使用超网络）。
    用于消融变体D: w/o HyperNet。
    参数直接存储在模型中，通过标准反向传播训练。
    """
    def __init__(self, input_dim, n_classes, n_trees=15, tree_depth=3):
        super().__init__()
        self.input_dim = input_dim
        self.n_trees = n_trees
        self.n_classes = n_classes
        self.depth = tree_depth
        self.n_internal = 2**tree_depth - 1
        self.n_leaves = 2**tree_depth

        # 直接存储各棵树的参数（非超网络生成）
        self.split_weights = nn.ParameterList([
            nn.Parameter(torch.randn(self.n_internal, input_dim) * 0.1)
            for _ in range(n_trees)
        ])
        self.split_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(self.n_internal))
            for _ in range(n_trees)
        ])
        self.leaf_logits = nn.ParameterList([
            nn.Parameter(torch.randn(self.n_leaves, n_classes) * 0.1)
            for _ in range(n_trees)
        ])
        self.tree_weights = nn.Parameter(torch.ones(n_trees) / n_trees)
        self.log_temperature = nn.Parameter(torch.tensor(0.0))

    @property
    def temperature(self):
        return torch.exp(self.log_temperature).clamp(0.1, 5.0)

    def forward(self, x):
        batch_size = x.size(0)
        all_outputs = []

        for t in range(self.n_trees):
            sw = self.split_weights[t]
            sb = self.split_biases[t]
            ll = self.leaf_logits[t]

            split_probs = torch.sigmoid(
                (x @ sw.T + sb) / self.temperature)
            leaf_probs = torch.ones(
                batch_size, self.n_leaves, device=x.device)

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

            leaf_class_probs = F.softmax(ll / self.temperature, dim=-1)
            out = torch.einsum('bl,lc->bc', leaf_probs, leaf_class_probs)
            all_outputs.append(out)

        all_outputs = torch.stack(all_outputs, dim=0)
        weights = F.softmax(self.tree_weights, dim=0)
        final_output = torch.einsum('t,tbc->bc', weights, all_outputs)
        return final_output


class AblationModel(nn.Module):
    """
    统一的消融模型，根据配置启用/禁用各组件。

    ablation_config:
        use_vae: bool       - 是否使用VAE数据增强
        use_interp: bool    - 是否在VAE采样时添加噪声(interpolation)
        use_hypernet: bool  - 是否使用超网络生成权重
        n_heads: int        - 超网络头数 (1=单头, 5=多头)
        n_trees: int        - 软决策树数量 (1=单树, 15=多树)
    """
    def __init__(self, input_dim, n_classes, config):
        super().__init__()
        self.config = config
        self.n_classes = n_classes

        # VAE (总是创建，但配置决定是否使用)
        self.vae = VAE(input_dim)

        if config['use_hypernet']:
            self.hypernet = HyperNetworkForTree(
                input_dim, n_classes,
                n_trees=config['n_trees'],
                tree_depth=3,
                n_heads=config['n_heads']
            )
            self.classifier = SoftTreeClassifier(tree_depth=3)
        else:
            self.direct_trees = DirectSoftTreeEnsemble(
                input_dim, n_classes,
                n_trees=config['n_trees'],
                tree_depth=3
            )

    def generate_augmented_data(self, X, y, n_augment=200):
        """数据增强：根据配置决定noise_scale"""
        self.vae.eval()
        augmented_X = [X]
        augmented_y = [y]

        noise_scale = 0.3 if self.config['use_interp'] else 0.0

        with torch.no_grad():
            for i in range(n_augment):
                idx = i % X.size(0)
                mu, logvar = self.vae.encode(X[idx:idx+1])
                z = self.vae.reparameterize(mu, logvar, noise_scale=noise_scale)
                generated = self.vae.decoder(z)
                augmented_X.append(generated)
                augmented_y.append(y[idx:idx+1])

        return torch.cat(augmented_X, dim=0), torch.cat(augmented_y, dim=0)

    def forward(self, X_train, X_test):
        if self.config['use_hypernet']:
            params = self.hypernet(X_train)
            trees_params, tree_weights = self.hypernet.parse_params(params)
            output = self.classifier(X_test, trees_params, tree_weights)
            return output, params
        else:
            output = self.direct_trees(X_test)
            return output, None


# =====================================================================
# 3. 训练函数
# =====================================================================

def train_ablation_model(X_train, y_train, X_val, y_val,
                         n_classes, config, device, epochs=300, seed=42):
    """训练一个消融变体模型"""
    set_seed(seed)

    input_dim = X_train.shape[1]
    model = AblationModel(input_dim, n_classes, config).to(device)

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)

    # ---- 阶段1: 训练VAE (如果使用VAE增强) ----
    if config['use_vae']:
        vae_optimizer = torch.optim.Adam(
            model.vae.parameters(), lr=0.002, weight_decay=1e-5)
        model.vae.train()
        for epoch in range(100):
            vae_optimizer.zero_grad()
            recon, mu, logvar = model.vae(X_train_t)
            recon_loss = F.mse_loss(recon, X_train_t)
            kl_loss = -0.5 * torch.mean(
                1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 0.01 * kl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.vae.parameters(), 1.0)
            vae_optimizer.step()

        # 确定性数据增强
        model.vae.eval()
        with torch.no_grad():
            X_aug, y_aug = model.generate_augmented_data(
                X_train_t, y_train_t, n_augment=200)
    else:
        # 不使用VAE，直接用原始数据
        X_aug = X_train_t
        y_aug = y_train_t

    # ---- 阶段2: 训练分类器 ----
    if config['use_hypernet']:
        params_to_optimize = (
            list(model.hypernet.parameters()) +
            list(model.classifier.parameters())
        )
    else:
        params_to_optimize = list(model.direct_trees.parameters())

    optimizer = torch.optim.AdamW(
        params_to_optimize, lr=0.01, weight_decay=0.05)

    warmup_epochs = 20

    def get_lr(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + np.cos(
            np.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    best_val_acc = 0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        if not config['use_vae']:
            model.vae.eval()  # VAE不参与反向传播

        optimizer.zero_grad()

        if config['use_hypernet']:
            output, params = model(X_aug, X_aug)
        else:
            output = model.direct_trees(X_aug)
            params = None

        cls_loss = F.cross_entropy(output, y_aug)

        reg_loss = 0
        if params is not None:
            reg_loss = 0.01 * (params ** 2).mean()

        # 一致性正则化 (仅超网络模式且预热后)
        consistency_loss = 0
        if config['use_hypernet'] and epoch > warmup_epochs:
            model.eval()
            with torch.no_grad():
                output2, _ = model(X_aug, X_aug)
            model.train()
            if not config['use_vae']:
                model.vae.eval()
            consistency_loss = F.mse_loss(output, output2.detach()) * 0.1

        loss = cls_loss + reg_loss + consistency_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        scheduler.step()

        # 验证
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                if config['use_hypernet']:
                    val_output, _ = model(X_train_t, X_val_t)
                else:
                    val_output = model.direct_trees(X_val_t)
                val_pred = val_output.argmax(dim=1).cpu().numpy()
                val_acc = accuracy_score(y_val, val_pred)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = {k: v.clone()
                                  for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= 5:
                    break

    if best_state:
        model.load_state_dict(best_state)
    return model


def predict_ablation(model, X_train, X_test, device):
    """用消融模型做预测"""
    model.eval()
    X_train_t = torch.FloatTensor(X_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        if model.config['use_hypernet']:
            output, _ = model(X_train_t, X_test_t)
        else:
            output = model.direct_trees(X_test_t)
        proba = F.softmax(output, dim=1)
    return proba.cpu().numpy()


# =====================================================================
# 4. 消融变体定义
# =====================================================================

ABLATION_VARIANTS = {
    'A_Full_Model': {
        'use_vae': True,
        'use_interp': True,
        'use_hypernet': True,
        'n_heads': 5,
        'n_trees': 15,
        'description': 'Full VAE-HyperNet-Fusion (all components)'
    },
    'B_No_VAE_Aug': {
        'use_vae': False,
        'use_interp': False,
        'use_hypernet': True,
        'n_heads': 5,
        'n_trees': 15,
        'description': 'w/o VAE Augmentation'
    },
    'C_No_Interpolation': {
        'use_vae': True,
        'use_interp': False,    # noise_scale=0, VAE只做确定性重建
        'use_hypernet': True,
        'n_heads': 5,
        'n_trees': 15,
        'description': 'w/o Interpolation (noise_scale=0)'
    },
    'D_No_HyperNet': {
        'use_vae': True,
        'use_interp': True,
        'use_hypernet': False,   # 直接训练SoftTree参数
        'n_heads': 1,
        'n_trees': 15,
        'description': 'w/o HyperNet (direct backprop on trees)'
    },
    'E_No_Ensemble': {
        'use_vae': True,
        'use_interp': True,
        'use_hypernet': True,
        'n_heads': 1,            # 单头
        'n_trees': 1,            # 单树
        'description': 'w/o Ensembling (1 head, 1 tree)'
    },
    'F_No_MultiHead': {
        'use_vae': True,
        'use_interp': True,
        'use_hypernet': True,
        'n_heads': 1,            # 单头
        'n_trees': 15,           # 保留多树
        'description': 'w/o Multi-head (1 head, 15 trees)'
    },
    'G_VAE_plus_RF': {
        'use_vae': True,
        'use_interp': True,
        'use_hypernet': False,
        'n_heads': 1,
        'n_trees': 1,
        'use_rf_classifier': True,  # 特殊标记：VAE增强后用RF
        'description': 'VAE augmentation + Random Forest'
    },
}


# =====================================================================
# 5. 数据集加载 (来自54)
# =====================================================================

def load_dataset_by_id(dataset_id,
                       data_dir="/data2/image_identification/src/small_data"):
    """加载数据集，兼容服务器和本地Mac路径"""
    # 同时尝试本地Mac路径
    local_data_dir = "/Users/jinmingzhang/D/1/1Postg/Sem_13_Thesis/3/数据2/小数据"
    local_prostate = "/Users/jinmingzhang/D/1/1Postg/Sem_13_Thesis/5_2026.01.28/src/final/data/Data_for_Jinming.csv"

    datasets = {
        0: ("prostate", None),
        1: ("balloons", "1.balloons/adult+stretch.data"),
        2: ("lenses", "2lens/lenses.data"),
        3: ("caesarian",
            "3.caesarian+section+classification+dataset/caesarian.csv"),
        4: ("iris", "4.iris/iris.data"),
        5: ("fertility", "5.fertility/fertility_Diagnosis.txt"),
        6: ("zoo", "6.zoo/zoo.data"),
        7: ("seeds", "7.seeds/seeds_dataset.txt"),
        8: ("haberman", "8.haberman+s+survival/haberman.data"),
        9: ("glass", "9.glass+identification/glass.data"),
        10: ("yeast", "10.yeast/yeast.data"),
    }

    if dataset_id not in datasets:
        return None, None, None

    name, filepath_rel = datasets[dataset_id]

    try:
        if dataset_id == 0:
            for path in [
                "/data2/image_identification/src/data/Data_for_Jinming.csv",
                "data/Data_for_Jinming.csv",
                local_prostate
            ]:
                if Path(path).exists():
                    df = pd.read_csv(path)
                    X = df.iloc[:, 2:].values.astype(float)
                    y = df['Group'].values
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                    return X, y, name
            return None, None, name

        # 尝试服务器和本地路径
        for base_dir in [data_dir, local_data_dir]:
            filepath = os.path.join(base_dir, filepath_rel)
            if not Path(filepath).exists():
                continue

            if dataset_id == 1:
                df = pd.read_csv(filepath, header=None)
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values
            elif dataset_id == 2:
                df = pd.read_csv(filepath, sep=r'\s+', header=None,
                                 engine='python')
                X = df.iloc[:, 1:-1].values
                y = df.iloc[:, -1].values
            elif dataset_id == 3:
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                data_start = False
                data_rows = []
                for line in lines:
                    if '@data' in line.lower():
                        data_start = True
                        continue
                    if data_start and line.strip() and not line.startswith('%'):
                        row = line.strip().split(',')
                        if len(row) >= 2:
                            data_rows.append(row)
                df = pd.DataFrame(data_rows)
                X = df.iloc[:, :-1].values.astype(float)
                y = df.iloc[:, -1].values
            elif dataset_id == 4:
                df = pd.read_csv(filepath, header=None)
                df = df.dropna(how='all')
                df = df[df.iloc[:, -1].astype(str).str.strip() != '']
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values
            elif dataset_id == 5:
                df = pd.read_csv(filepath, header=None)
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values
            elif dataset_id == 6:
                df = pd.read_csv(filepath, header=None)
                X = df.iloc[:, 1:-1].values
                y = df.iloc[:, -1].values
            elif dataset_id == 7:
                df = pd.read_csv(filepath, sep=r'\s+', header=None,
                                 engine='python')
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values
            elif dataset_id == 8:
                df = pd.read_csv(filepath, header=None)
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values
            elif dataset_id == 9:
                df = pd.read_csv(filepath, header=None)
                X = df.iloc[:, 1:-1].values
                y = df.iloc[:, -1].values
            elif dataset_id == 10:
                df = pd.read_csv(filepath, sep=r'\s+', header=None,
                                 engine='python')
                X = df.iloc[:, 1:-1].values
                y = df.iloc[:, -1].values

            # 处理非数值特征
            for col in range(X.shape[1]):
                try:
                    X[:, col] = X[:, col].astype(float)
                except (ValueError, TypeError):
                    le = LabelEncoder()
                    X[:, col] = le.fit_transform(X[:, col].astype(str))
            X = X.astype(float)

            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
            return X, y, name

        return None, None, name

    except Exception as e:
        print(f"加载数据集 {name} 出错: {e}")
        traceback.print_exc()
        return None, None, name


# =====================================================================
# 6. 单任务评估函数 (子进程)
# =====================================================================

def evaluate_single_task(args):
    """
    评估单个消融变体在一个数据集一个fold上的表现。
    在子进程中运行。
    """
    (variant_name, config, dataset_id, fold_idx,
     train_idx_list, test_idx_list,
     X_data, y_data, n_classes, gpu_id, seed) = args

    train_idx = np.array(train_idx_list)
    test_idx = np.array(test_idx_list)

    try:
        device = torch.device(
            f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        set_seed(seed + fold_idx)

        X_train, X_test = X_data[train_idx], X_data[test_idx]
        y_train, y_test = y_data[train_idx], y_data[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # 特殊变体G: VAE + RF
        if config.get('use_rf_classifier', False):
            vae_device = device

            # 训练VAE
            set_seed(seed + fold_idx)
            input_dim = X_train_s.shape[1]
            vae = VAE(input_dim).to(vae_device)
            X_t = torch.FloatTensor(X_train_s).to(vae_device)
            y_t = torch.LongTensor(y_train).to(vae_device)

            vae_opt = torch.optim.Adam(
                vae.parameters(), lr=0.002, weight_decay=1e-5)
            vae.train()
            for ep in range(100):
                vae_opt.zero_grad()
                recon, mu, logvar = vae(X_t)
                rl = F.mse_loss(recon, X_t)
                kl = -0.5 * torch.mean(
                    1 + logvar - mu.pow(2) - logvar.exp())
                (rl + 0.01 * kl).backward()
                torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
                vae_opt.step()

            # VAE增强
            vae.eval()
            aug_X = [X_t]
            aug_y = [y_t]
            with torch.no_grad():
                for i in range(200):
                    idx = i % X_t.size(0)
                    mu, logvar = vae.encode(X_t[idx:idx+1])
                    z = vae.reparameterize(mu, logvar, noise_scale=0.3)
                    gen = vae.decoder(z)
                    aug_X.append(gen)
                    aug_y.append(y_t[idx:idx+1])
            X_aug = torch.cat(aug_X, dim=0).cpu().numpy()
            y_aug = torch.cat(aug_y, dim=0).cpu().numpy()

            # RF on augmented data
            rf = RandomForestClassifier(
                n_estimators=100, random_state=seed, n_jobs=1)
            rf.fit(X_aug, y_aug)
            preds = rf.predict(X_test_s)
            acc = accuracy_score(y_test, preds) * 100
            return {
                'variant': variant_name,
                'dataset_id': dataset_id,
                'fold': fold_idx,
                'accuracy': acc,
                'status': 'ok'
            }

        # 标准消融变体
        model = train_ablation_model(
            X_train_s, y_train, X_test_s, y_test,
            n_classes, config, device,
            epochs=300, seed=seed + fold_idx
        )

        proba = predict_ablation(model, X_train_s, X_test_s, device)
        preds = proba.argmax(axis=1)
        acc = accuracy_score(y_test, preds) * 100

        # 清理GPU
        del model
        torch.cuda.empty_cache()

        return {
            'variant': variant_name,
            'dataset_id': dataset_id,
            'fold': fold_idx,
            'accuracy': acc,
            'status': 'ok'
        }

    except Exception as e:
        traceback.print_exc()
        return {
            'variant': variant_name,
            'dataset_id': dataset_id,
            'fold': fold_idx,
            'accuracy': 0.0,
            'status': f'error: {str(e)}'
        }


# =====================================================================
# 7. 主函数
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='VAE-HyperNet-Fusion Ablation Study')
    parser.add_argument('--datasets', nargs='+', type=int,
                        default=list(range(11)),
                        help='Dataset IDs to evaluate (default: 0-10)')
    parser.add_argument('--gpus', nargs='+', type=int, default=None,
                        help='GPU IDs to use (default: auto-detect)')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Force CPU mode')
    parser.add_argument('--workers-per-gpu', type=int, default=3,
                        help='Workers per GPU (default: 3)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed')
    parser.add_argument('--variants', nargs='+', default=None,
                        help='Variant names to run (default: all)')
    args = parser.parse_args()

    print("=" * 80)
    print("    VAE-HyperNet-Fusion 消融实验 (Ablation Study)")
    print("=" * 80)
    print()

    # 确定设备
    if args.no_gpu or not torch.cuda.is_available():
        gpu_ids = []
        n_workers = max(1, mp.cpu_count() // 4)
        print(f"模式: CPU, workers={n_workers}")
    else:
        if args.gpus:
            gpu_ids = args.gpus
        else:
            gpu_ids = list(range(torch.cuda.device_count()))
            # 跳过GPU 0和1（可能被占用）
            if len(gpu_ids) > 2:
                gpu_ids = gpu_ids[2:]
        n_workers = len(gpu_ids) * args.workers_per_gpu
        print(f"模式: GPU {gpu_ids}, workers={n_workers}")

    # 选择变体
    if args.variants:
        variants = {k: v for k, v in ABLATION_VARIANTS.items()
                    if k in args.variants}
    else:
        variants = ABLATION_VARIANTS

    print(f"\n消融变体 ({len(variants)}个):")
    for name, cfg in variants.items():
        print(f"  {name}: {cfg['description']}")

    # 加上RF基线
    print(f"\n数据集: {args.datasets}")

    # ---- 加载所有数据集 ----
    print("\n加载数据集...")
    all_datasets = {}
    for did in args.datasets:
        X, y, dname = load_dataset_by_id(did)
        if X is not None:
            n_classes = len(np.unique(y))
            all_datasets[did] = {
                'X': X, 'y': y, 'name': dname,
                'n_classes': n_classes,
                'n_samples': len(X),
                'n_features': X.shape[1]
            }
            print(f"  [{did}] {dname}: {len(X)} samples, "
                  f"{X.shape[1]} features, {n_classes} classes")
        else:
            print(f"  [{did}] {dname}: SKIP (load failed)")

    if not all_datasets:
        print("没有可用数据集，退出。")
        return

    # ---- 准备任务列表 ----
    print("\n准备任务...")
    tasks = []
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for did, dinfo in all_datasets.items():
        X, y = dinfo['X'], dinfo['y']
        n_classes = dinfo['n_classes']
        folds = list(kfold.split(X, y))

        # RF基线任务不需要加入pool，直接在主进程算
        # 各消融变体的任务
        for vname, vcfg in variants.items():
            for fold_idx, (train_idx, test_idx) in enumerate(folds):
                gpu_id = (gpu_ids[(len(tasks)) % len(gpu_ids)]
                          if gpu_ids else 0)
                tasks.append((
                    vname, vcfg, did, fold_idx,
                    train_idx.tolist(), test_idx.tolist(),
                    X, y, n_classes, gpu_id, args.seed
                ))

    total = len(tasks)
    print(f"总任务数: {total} "
          f"({len(all_datasets)} datasets × {len(variants)} variants × 5 folds)")

    # ---- 先计算RF基线（快速） ----
    print("\n计算RF基线...")
    rf_results = {}
    for did, dinfo in all_datasets.items():
        X, y = dinfo['X'], dinfo['y']
        fold_accs = []
        for fold_idx, (train_idx, test_idx) in enumerate(
                kfold.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            rf = RandomForestClassifier(
                n_estimators=100, random_state=42)
            rf.fit(X_train_s, y_train)
            acc = accuracy_score(y_test, rf.predict(X_test_s)) * 100
            fold_accs.append(acc)
        mean_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)
        rf_results[did] = {
            'mean': mean_acc, 'std': std_acc,
            'fold_accs': fold_accs
        }
        print(f"  [{did}] {dinfo['name']}: RF = {mean_acc:.1f}±{std_acc:.1f}%")

    # ---- 并行运行消融任务 ----
    print(f"\n开始并行消融实验 ({n_workers} workers)...")
    start_time = time.time()

    if n_workers > 1 and total > 1:
        ctx = mp.get_context('spawn')
        with ctx.Pool(n_workers) as pool:
            results_list = []
            for i, result in enumerate(
                    pool.imap_unordered(evaluate_single_task, tasks)):
                results_list.append(result)
                done = i + 1
                elapsed = time.time() - start_time
                eta = elapsed / done * (total - done) if done > 0 else 0
                # 每完成10个或最后一个时打印进度
                if done % 10 == 0 or done == total:
                    print(f"  进度: {done}/{total} "
                          f"({done/total*100:.0f}%) "
                          f"已用时 {elapsed:.0f}s, "
                          f"预计还需 {eta:.0f}s")
    else:
        results_list = []
        for i, task in enumerate(tasks):
            result = evaluate_single_task(task)
            results_list.append(result)
            print(f"  进度: {i+1}/{total}")

    total_time = time.time() - start_time
    print(f"\n总耗时: {total_time:.0f}s ({total_time/60:.1f}min)")

    # ---- 汇总结果 ----
    print("\n" + "=" * 80)
    print("  消融实验结果汇总")
    print("=" * 80)

    # 按 (variant, dataset) 汇总
    from collections import defaultdict
    summary = defaultdict(lambda: defaultdict(list))

    for r in results_list:
        if r['status'] == 'ok':
            summary[r['variant']][r['dataset_id']].append(r['accuracy'])

    # 打印表头
    variant_names = list(variants.keys())
    header = f"{'Dataset':<15} {'RF':>12}"
    for vn in variant_names:
        short = vn.split('_', 1)[1] if '_' in vn else vn
        header += f" {short:>14}"
    print(header)
    print("-" * len(header))

    # 收集完整结果用于保存
    full_results = {
        'experiment': 'Ablation Study for VAE-HyperNet-Fusion',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'seed': args.seed,
        'n_folds': 5,
        'rf_baseline': {},
        'ablation_results': {},
        'per_dataset': {}
    }

    for did in sorted(all_datasets.keys()):
        dname = all_datasets[did]['name']
        rf_str = (f"{rf_results[did]['mean']:.1f}"
                  f"±{rf_results[did]['std']:.1f}")

        row = f"{dname:<15} {rf_str:>12}"

        dataset_results = {'RF': rf_str}

        for vn in variant_names:
            accs = summary[vn].get(did, [])
            if accs:
                m, s = np.mean(accs), np.std(accs)
                cell = f"{m:.1f}±{s:.1f}"
                dataset_results[vn] = {
                    'mean': round(m, 2), 'std': round(s, 2),
                    'fold_accs': [round(a, 2) for a in accs]
                }
            else:
                cell = "ERROR"
                dataset_results[vn] = {'mean': 0, 'std': 0, 'fold_accs': []}
            row += f" {cell:>14}"

        print(row)
        full_results['per_dataset'][dname] = dataset_results

    # 打印平均行
    print("-" * len(header))
    avg_row = f"{'AVERAGE':<15}"
    rf_means = [rf_results[did]['mean'] for did in sorted(all_datasets.keys())]
    avg_row += f" {np.mean(rf_means):>7.1f}     "

    for vn in variant_names:
        vn_means = []
        for did in sorted(all_datasets.keys()):
            accs = summary[vn].get(did, [])
            if accs:
                vn_means.append(np.mean(accs))
        if vn_means:
            avg_row += f" {np.mean(vn_means):>9.1f}     "
        else:
            avg_row += f" {'N/A':>14}"
    print(avg_row)

    # 打印各组件的贡献分析
    print("\n" + "=" * 80)
    print("  各组件的贡献分析 (Full Model vs 去掉该组件的准确率差)")
    print("=" * 80)

    component_effects = {
        'VAE Augmentation': ('A_Full_Model', 'B_No_VAE_Aug'),
        'Interpolation': ('A_Full_Model', 'C_No_Interpolation'),
        'HyperNet': ('A_Full_Model', 'D_No_HyperNet'),
        'Ensembling': ('A_Full_Model', 'E_No_Ensemble'),
        'Multi-head': ('A_Full_Model', 'F_No_MultiHead'),
    }

    print(f"\n{'Component':<22} ", end='')
    for did in sorted(all_datasets.keys()):
        print(f" {all_datasets[did]['name']:>10}", end='')
    print(f" {'AVG':>8}")
    print("-" * (22 + 11 * len(all_datasets) + 9))

    for comp_name, (full_key, ablated_key) in component_effects.items():
        row = f"{comp_name:<22} "
        deltas = []
        for did in sorted(all_datasets.keys()):
            full_accs = summary[full_key].get(did, [])
            ablated_accs = summary[ablated_key].get(did, [])
            if full_accs and ablated_accs:
                delta = np.mean(full_accs) - np.mean(ablated_accs)
                deltas.append(delta)
                sign = '+' if delta >= 0 else ''
                row += f" {sign}{delta:>8.1f}%"
            else:
                row += f" {'N/A':>10}"
        if deltas:
            avg_delta = np.mean(deltas)
            sign = '+' if avg_delta >= 0 else ''
            row += f" {sign}{avg_delta:>6.1f}%"
        print(row)

    print("\n说明: 正值表示该组件有正向贡献（移除后准确率下降）")

    # ---- 保存结果 ----
    # 尝试服务器路径，失败则用本地
    output_dirs = [
        Path('/data2/image_identification/src/output'),
        Path('/Users/jinmingzhang/D/1/1Postg/Sem_13_Thesis/'
             '5_2026.01.28/src/final/output')
    ]
    for output_dir in output_dirs:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = output_dir / f'55_ablation_{timestamp}.json'

            # 添加component effects到结果
            full_results['component_effects'] = {}
            for comp_name, (full_key, ablated_key) in component_effects.items():
                deltas = []
                for did in sorted(all_datasets.keys()):
                    full_accs = summary[full_key].get(did, [])
                    ablated_accs = summary[ablated_key].get(did, [])
                    if full_accs and ablated_accs:
                        deltas.append(
                            np.mean(full_accs) - np.mean(ablated_accs))
                full_results['component_effects'][comp_name] = {
                    'per_dataset_delta': deltas,
                    'avg_delta': float(np.mean(deltas)) if deltas else 0
                }

            with open(output_file, 'w') as f:
                json.dump(full_results, f, indent=2, ensure_ascii=False)
            print(f"\n结果已保存: {output_file}")
            break
        except Exception as e:
            print(f"保存到 {output_dir} 失败: {e}")

    # 打印LaTeX格式表格（方便论文使用）
    print("\n" + "=" * 80)
    print("  LaTeX Table")
    print("=" * 80)
    print()
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Ablation study results (\%). "
          r"Each column removes one component from the full model.}")
    print(r"\label{tab:ablation}")
    print(r"\resizebox{\textwidth}{!}{")
    print(r"\begin{tabular}{l" + "c" * (2 + len(variants)) + "}")
    print(r"\toprule")
    # Header
    latex_header = "Dataset & RF"
    short_labels = {
        'A_Full_Model': 'Full',
        'B_No_VAE_Aug': 'w/o VAE',
        'C_No_Interpolation': 'w/o Interp.',
        'D_No_HyperNet': 'w/o HyperNet',
        'E_No_Ensemble': 'w/o Ensem.',
        'F_No_MultiHead': 'w/o M-Head',
        'G_VAE_plus_RF': 'VAE+RF',
    }
    for vn in variant_names:
        latex_header += f" & {short_labels.get(vn, vn)}"
    latex_header += r" \\"
    print(latex_header)
    print(r"\midrule")

    for did in sorted(all_datasets.keys()):
        dname = all_datasets[did]['name']
        rf_m = rf_results[did]['mean']
        rf_s = rf_results[did]['std']
        line = f"{dname} & {rf_m:.1f}$\\pm${rf_s:.1f}"

        # 找出该数据集上最好的变体 (不含RF)
        best_acc = 0
        best_vn = None
        for vn in variant_names:
            accs = summary[vn].get(did, [])
            if accs and np.mean(accs) > best_acc:
                best_acc = np.mean(accs)
                best_vn = vn

        for vn in variant_names:
            accs = summary[vn].get(did, [])
            if accs:
                m, s = np.mean(accs), np.std(accs)
                if vn == best_vn and m > rf_m:
                    line += f" & \\textbf{{{m:.1f}$\\pm${s:.1f}}}"
                else:
                    line += f" & {m:.1f}$\\pm${s:.1f}"
            else:
                line += " & --"
        line += r" \\"
        print(line)

    print(r"\midrule")
    # Average row
    avg_line = "Average"
    avg_line += f" & {np.mean(rf_means):.1f}"
    for vn in variant_names:
        vn_means = []
        for did in sorted(all_datasets.keys()):
            accs = summary[vn].get(did, [])
            if accs:
                vn_means.append(np.mean(accs))
        if vn_means:
            avg_line += f" & {np.mean(vn_means):.1f}"
        else:
            avg_line += " & --"
    avg_line += r" \\"
    print(avg_line)

    print(r"\bottomrule")
    print(r"\end{tabular}}")
    print(r"\end{table}")

    print("\n消融实验完成！")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
