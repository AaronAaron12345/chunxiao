#!/usr/bin/env python3
"""
62_ablation_v2.py - 消融实验 v2 (使用Optuna优化后的每数据集最优参数)

== 与 55_ablation_study 的区别 ==
  55版本: 所有数据集使用相同的默认超参数 (n_trees=15, n_heads=5, epochs=300, ...)
  62版本: 每个数据集使用59/61 Optuna搜索到的最优超参数

== 消融变体 (8个, 同55) ==
  A  Full Model        — 所有组件: VAE增强 + 插值 + 超网络 + 多头集成
  B  w/o VAE Aug       — 去掉VAE增强, 直接用原始数据
  C  w/o Interpolation — VAE增强但noise_scale=0(纯重建)
  D  w/o HyperNet      — 去掉超网络, 直接反向传播训练SoftTree参数
  E  w/o Ensemble      — 单头+单树 (n_heads=1, n_trees=1)
  F  w/o Multi-head    — 单头+多树 (n_heads=1, 保持优化的n_trees)
  G  VAE + RF          — VAE增强数据 + 随机森林分类器
  H  RF Baseline       — 纯随机森林 (无增强)

== 执行 ==
  nohup /data1/condaproject/dinov2/bin/python3 -u 62_ablation_v2.py \
        --gpus 1 2 3 4 5 > 62_log.txt 2>&1 &
"""

import os, sys, time, json, random, traceback, argparse
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
from datetime import datetime
from collections import defaultdict
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')

# ====================================================================
# 1. SEED
# ====================================================================
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ====================================================================
# 2. MODEL DEFINITIONS (与59_bayesian_tuning.py完全一致)
# ====================================================================
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        h = 32
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h), nn.LayerNorm(h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU())
        self.fc_mu  = nn.Linear(h, latent_dim)
        self.fc_var = nn.Linear(h, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, input_dim))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None: nn.init.zeros_(m.bias)
    def encode(self, x):
        h = self.encoder(x); return self.fc_mu(h), self.fc_var(h)
    def reparameterize(self, mu, lv, ns=1.0):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * lv) * ns
    def forward(self, x, ns=1.0):
        mu, lv = self.encode(x)
        return self.decoder(self.reparameterize(mu, lv, ns=ns)), mu, lv


class HyperNet(nn.Module):
    def __init__(self, input_dim, n_classes, n_trees=15, tree_depth=3,
                 hidden_dim=64, n_heads=5, dropout=0.15):
        super().__init__()
        self.input_dim, self.n_trees = input_dim, n_trees
        self.tree_depth, self.n_heads, self.n_classes = tree_depth, n_heads, n_classes
        ni = 2**tree_depth - 1
        nl = 2**tree_depth
        self.n_internal, self.n_leaves = ni, nl
        self.params_per_tree = ni * input_dim + ni + nl * n_classes
        self.total_params = self.params_per_tree * n_trees + n_trees
        self.data_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout * 0.67),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.hyper_heads = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.ReLU(),
            nn.Dropout(dropout * 0.67),
            nn.Linear(hidden_dim * 2, self.total_params))
            for _ in range(n_heads)])
        self.head_weights = nn.Parameter(torch.ones(n_heads) / n_heads)
        self.output_scale = nn.Parameter(torch.tensor(1.0))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, X):
        ctx = self.data_encoder(X).mean(dim=0)
        all_p = torch.stack([h(ctx) for h in self.hyper_heads])
        w = F.softmax(self.head_weights, dim=0)
        return torch.tanh(torch.einsum('h,hp->p', w, all_p)) * self.output_scale

    def parse_params(self, params):
        trees, off = [], 0
        for _ in range(self.n_trees):
            sw_sz = self.n_internal * self.input_dim
            sw = params[off:off+sw_sz].view(self.n_internal, self.input_dim); off += sw_sz
            sb = params[off:off+self.n_internal]; off += self.n_internal
            ll_sz = self.n_leaves * self.n_classes
            ll = params[off:off+ll_sz].view(self.n_leaves, self.n_classes); off += ll_sz
            trees.append({'sw': sw, 'sb': sb, 'll': ll})
        return trees, params[off:off+self.n_trees]


class TreeClassifier(nn.Module):
    def __init__(self, tree_depth=3, init_temp=1.0):
        super().__init__()
        self.depth = tree_depth
        self.n_internal = 2**tree_depth - 1
        self.n_leaves = 2**tree_depth
        self.log_temp = nn.Parameter(torch.tensor(float(np.log(init_temp))))
    @property
    def temperature(self):
        return torch.exp(self.log_temp).clamp(0.1, 5.0)
    def forward_tree(self, x, sw, sb, ll):
        B = x.size(0)
        sp = torch.sigmoid((x @ sw.T + sb) / self.temperature)
        lp = torch.ones(B, self.n_leaves, device=x.device)
        for li in range(self.n_leaves):
            pp = torch.ones(B, device=x.device)
            ni = li + self.n_internal
            for _ in range(self.depth):
                pi = (ni - 1) // 2
                pp = pp * (sp[:, pi] if ni % 2 == 0 else 1 - sp[:, pi])
                ni = pi
            lp[:, li] = pp
        return torch.einsum('bl,lc->bc', lp, F.softmax(ll / self.temperature, dim=-1))
    def forward(self, x, trees, tw):
        outs = torch.stack([self.forward_tree(x, t['sw'], t['sb'], t['ll'])
                            for t in trees])
        return torch.einsum('t,tbc->bc', F.softmax(tw, dim=0), outs)


class DirectSoftTreeEnsemble(nn.Module):
    """直接训练SoftTree (无超网络) — 用于消融D"""
    def __init__(self, input_dim, n_classes, n_trees=15, tree_depth=3):
        super().__init__()
        self.input_dim = input_dim
        self.n_trees = n_trees
        self.n_classes = n_classes
        self.depth = tree_depth
        self.n_internal = 2**tree_depth - 1
        self.n_leaves = 2**tree_depth
        self.split_weights = nn.ParameterList([
            nn.Parameter(torch.randn(self.n_internal, input_dim) * 0.1)
            for _ in range(n_trees)])
        self.split_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(self.n_internal))
            for _ in range(n_trees)])
        self.leaf_logits = nn.ParameterList([
            nn.Parameter(torch.randn(self.n_leaves, n_classes) * 0.1)
            for _ in range(n_trees)])
        self.tree_weights = nn.Parameter(torch.ones(n_trees) / n_trees)
        self.log_temperature = nn.Parameter(torch.tensor(0.0))

    @property
    def temperature(self):
        return torch.exp(self.log_temperature).clamp(0.1, 5.0)

    def forward(self, x):
        B = x.size(0)
        all_out = []
        for t in range(self.n_trees):
            sw, sb, ll = self.split_weights[t], self.split_biases[t], self.leaf_logits[t]
            sp = torch.sigmoid((x @ sw.T + sb) / self.temperature)
            lp = torch.ones(B, self.n_leaves, device=x.device)
            for li in range(self.n_leaves):
                pp = torch.ones(B, device=x.device)
                ni = li + self.n_internal
                for _ in range(self.depth):
                    pi = (ni - 1) // 2
                    pp = pp * (sp[:, pi] if ni % 2 == 0 else 1 - sp[:, pi])
                    ni = pi
                lp[:, li] = pp
            all_out.append(torch.einsum('bl,lc->bc', lp,
                                        F.softmax(ll / self.temperature, dim=-1)))
        outs = torch.stack(all_out)
        return torch.einsum('t,tbc->bc', F.softmax(self.tree_weights, dim=0), outs)


# ====================================================================
# 3. ABLATION MODEL (统一封装, 支持各消融配置)
# ====================================================================
class AblationModel(nn.Module):
    """
    消融模型: 通过 ablation_config 控制各组件:
      use_vae:      是否使用VAE数据增强
      use_interp:   是否在VAE采样时加噪声(interpolation)
      use_hypernet: 是否使用超网络生成参数
      n_heads:      超网络头数
      n_trees:      SoftTree棵数
      tree_depth:   SoftTree深度
      hidden_dim:   超网络隐藏维度
      latent_dim:   VAE隐空间维度
      dropout:      Dropout率
    """
    def __init__(self, input_dim, n_classes, config):
        super().__init__()
        self.config = config
        self.n_classes = n_classes

        latent_dim = config.get('latent_dim', 8)
        self.vae = VAE(input_dim, latent_dim=latent_dim)

        tree_depth = config.get('tree_depth', 3)
        hidden_dim = config.get('hidden_dim', 64)
        dropout = config.get('dropout', 0.15)

        if config['use_hypernet']:
            self.hypernet = HyperNet(
                input_dim, n_classes,
                n_trees=config['n_trees'],
                tree_depth=tree_depth,
                hidden_dim=hidden_dim,
                n_heads=config['n_heads'],
                dropout=dropout)
            self.classifier = TreeClassifier(tree_depth=tree_depth)
        else:
            self.direct_trees = DirectSoftTreeEnsemble(
                input_dim, n_classes,
                n_trees=config['n_trees'],
                tree_depth=tree_depth)

    def augment(self, X, y, n_augment=200, noise_scale=0.3):
        self.vae.eval()
        aX, aY = [X], [y]
        ns = noise_scale if self.config['use_interp'] else 0.0
        with torch.no_grad():
            for i in range(n_augment):
                idx = i % X.size(0)
                mu, lv = self.vae.encode(X[idx:idx+1])
                aX.append(self.vae.decoder(self.vae.reparameterize(mu, lv, ns=ns)))
                aY.append(y[idx:idx+1])
        return torch.cat(aX), torch.cat(aY)

    def forward(self, X_train, X_test):
        if self.config['use_hypernet']:
            p = self.hypernet(X_train)
            tp, tw = self.hypernet.parse_params(p)
            return self.classifier(X_test, tp, tw), p
        else:
            return self.direct_trees(X_test), None


# ====================================================================
# 4. 训练函数 (全参数化, 从Optuna最优参数读取)
# ====================================================================
def train_ablation_model(X_train, y_train, X_val, y_val,
                         n_classes, config, hp, device, seed=42):
    """
    用指定超参数hp训练一个消融变体模型。
    hp来自optuna_d{i}.json中的best_params。
    """
    set_seed(seed)
    input_dim = X_train.shape[1]
    model = AblationModel(input_dim, n_classes, config).to(device)

    Xt = torch.FloatTensor(X_train).to(device)
    yt = torch.LongTensor(y_train).to(device)
    Xv = torch.FloatTensor(X_val).to(device)

    # ---- Phase 1: 训练VAE (如果使用VAE增强) ----
    if config['use_vae']:
        vae_lr = hp.get('vae_lr', 0.002)
        vae_epochs = hp.get('vae_epochs', 100)
        kl_weight = hp.get('kl_weight', 0.01)

        vopt = torch.optim.Adam(model.vae.parameters(), lr=vae_lr, weight_decay=1e-5)
        model.vae.train()
        for _ in range(vae_epochs):
            vopt.zero_grad()
            rec, mu, lv = model.vae(Xt)
            loss = F.mse_loss(rec, Xt) + kl_weight * (
                -0.5 * torch.mean(1 + lv - mu ** 2 - lv.exp()))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.vae.parameters(), 1.0)
            vopt.step()

        # 数据增强
        n_augment = hp.get('n_augment', 200)
        noise_scale = hp.get('noise_scale', 0.3)
        with torch.no_grad():
            Xa, ya = model.augment(Xt, yt, n_augment, noise_scale)
    else:
        Xa, ya = Xt, yt

    # ---- Phase 2: 训练分类器 ----
    if config['use_hypernet']:
        prms = list(model.hypernet.parameters()) + list(model.classifier.parameters())
    else:
        prms = list(model.direct_trees.parameters())

    lr = hp.get('lr', 0.01)
    weight_decay = hp.get('weight_decay', 0.05)
    epochs = hp.get('epochs', 300)
    warmup = hp.get('warmup_epochs', 20)
    reg_weight = hp.get('reg_weight', 0.01)

    opt = torch.optim.AdamW(prms, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e:
        (e + 1) / warmup if e < warmup else
        0.5 * (1 + np.cos(np.pi * (e - warmup) / max(epochs - warmup, 1))))

    best_acc, best_st, noimpr = 0, None, 0

    for ep in range(epochs):
        model.train()
        if not config['use_vae']:
            model.vae.eval()

        opt.zero_grad()

        if config['use_hypernet']:
            out, p = model(Xa, Xa)
            loss = F.cross_entropy(out, ya) + reg_weight * (p ** 2).mean()
        else:
            out = model.direct_trees(Xa)
            p = None
            loss = F.cross_entropy(out, ya)

        # 一致性正则 (仅超网络+预热后)
        if config['use_hypernet'] and ep > warmup:
            model.eval()
            with torch.no_grad():
                o2, _ = model(Xa, Xa)
            model.train()
            if not config['use_vae']:
                model.vae.eval()
            loss = loss + 0.1 * F.mse_loss(out, o2.detach())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step(); sched.step()

        # 验证 (每10轮)
        if (ep + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                if config['use_hypernet']:
                    vo, _ = model(Xt, Xv)
                else:
                    vo = model.direct_trees(Xv)
                va = accuracy_score(y_val, vo.argmax(1).cpu().numpy())
            if va > best_acc:
                best_acc = va
                best_st = {k: v.clone() for k, v in model.state_dict().items()}
                noimpr = 0
            else:
                noimpr += 1
            if noimpr >= 5:
                break

    if best_st:
        model.load_state_dict(best_st)
    return model


def predict_ablation(model, X_train, X_test, device):
    model.eval()
    Xt = torch.FloatTensor(X_train).to(device)
    Xe = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        if model.config['use_hypernet']:
            out, _ = model(Xt, Xe)
        else:
            out = model.direct_trees(Xe)
    return F.softmax(out, dim=1).cpu().numpy()


# ====================================================================
# 5. 消融变体定义 (从Optuna最优参数构建)
# ====================================================================

def build_ablation_variants(hp):
    """
    根据Optuna最优超参数hp, 构建8个消融变体的配置。
    每个变体继承hp中的模型结构参数, 但去除特定组件。
    """
    n_trees = hp.get('n_trees', 15)
    tree_depth = hp.get('tree_depth', 3)
    n_heads = hp.get('n_heads', 5)
    hidden_dim = hp.get('hidden_dim', 64)
    latent_dim = hp.get('latent_dim', 8)
    dropout = hp.get('dropout', 0.15)

    base = {
        'n_trees': n_trees, 'tree_depth': tree_depth,
        'hidden_dim': hidden_dim, 'latent_dim': latent_dim,
        'dropout': dropout,
    }

    variants = {
        'A_Full_Model': {
            **base,
            'use_vae': True, 'use_interp': True,
            'use_hypernet': True, 'n_heads': n_heads,
            'description': 'Full VAE-HyperNet-Fusion (all components)'
        },
        'B_No_VAE_Aug': {
            **base,
            'use_vae': False, 'use_interp': False,
            'use_hypernet': True, 'n_heads': n_heads,
            'description': 'w/o VAE Augmentation'
        },
        'C_No_Interpolation': {
            **base,
            'use_vae': True, 'use_interp': False,
            'use_hypernet': True, 'n_heads': n_heads,
            'description': 'w/o Interpolation (noise_scale=0)'
        },
        'D_No_HyperNet': {
            **base,
            'use_vae': True, 'use_interp': True,
            'use_hypernet': False, 'n_heads': 1,
            'description': 'w/o HyperNet (direct backprop on trees)'
        },
        'E_No_Ensemble': {
            **base,
            'use_vae': True, 'use_interp': True,
            'use_hypernet': True,
            'n_heads': 1, 'n_trees': 1,
            'description': 'w/o Ensembling (1 head, 1 tree)'
        },
        'F_No_MultiHead': {
            **base,
            'use_vae': True, 'use_interp': True,
            'use_hypernet': True,
            'n_heads': 1,  # 单头, 保留优化后的n_trees
            'description': f'w/o Multi-head (1 head, {n_trees} trees)'
        },
        'G_VAE_plus_RF': {
            **base,
            'use_vae': True, 'use_interp': True,
            'use_hypernet': False,
            'n_heads': 1, 'n_trees': 1,
            'use_rf_classifier': True,
            'description': 'VAE augmentation + Random Forest'
        },
    }
    return variants


# ====================================================================
# 6. 加载Optuna最优参数
# ====================================================================

DEFAULT_HP = {
    'n_trees': 15, 'tree_depth': 3, 'n_heads': 5,
    'hidden_dim': 64, 'latent_dim': 8, 'dropout': 0.15,
    'n_augment': 200, 'noise_scale': 0.3,
    'vae_lr': 0.002, 'vae_epochs': 100, 'kl_weight': 0.01,
    'lr': 0.01, 'weight_decay': 0.05, 'epochs': 300,
    'warmup_epochs': 20, 'reg_weight': 0.01,
}


def load_optuna_params(dataset_id, output_dir):
    """从optuna_d{i}.json加载最优超参数, 不存在则用默认值"""
    f = Path(output_dir) / f'optuna_d{dataset_id}.json'
    if f.exists():
        with open(f) as fp:
            data = json.load(fp)
        hp = data.get('best_params', DEFAULT_HP)
        # 确保所有必填字段存在
        for k, v in DEFAULT_HP.items():
            if k not in hp:
                hp[k] = v
        return hp, data
    else:
        print(f"  警告: {f} 不存在, 使用默认参数")
        return DEFAULT_HP.copy(), None


# ====================================================================
# 7. 数据集加载 (同59)
# ====================================================================
def load_dataset_by_id(did, data_dir="/data2/image_identification/src/small_data"):
    local_data_dir = "/Users/jinmingzhang/D/1/1Postg/Sem_13_Thesis/3/数据2/小数据"
    local_prostate = "/Users/jinmingzhang/D/1/1Postg/Sem_13_Thesis/5_2026.01.28/src/final/data/Data_for_Jinming.csv"

    datasets = {
        0: ("prostate", None),
        1: ("balloons", "1.balloons/adult+stretch.data"),
        2: ("lenses", "2lens/lenses.data"),
        3: ("caesarian", "3.caesarian+section+classification+dataset/caesarian.csv"),
        4: ("iris", "4.iris/iris.data"),
        5: ("fertility", "5.fertility/fertility_Diagnosis.txt"),
        6: ("zoo", "6.zoo/zoo.data"),
        7: ("seeds", "7.seeds/seeds_dataset.txt"),
        8: ("haberman", "8.haberman+s+survival/haberman.data"),
        9: ("glass", "9.glass+identification/glass.data"),
        10: ("yeast", "10.yeast/yeast.data"),
    }
    if did not in datasets:
        return None, None, None
    name, frel = datasets[did]
    try:
        if did == 0:
            for p in ["/data2/image_identification/src/data/Data_for_Jinming.csv",
                      "data/Data_for_Jinming.csv", local_prostate]:
                if Path(p).exists():
                    df = pd.read_csv(p)
                    X = df.iloc[:, 2:].values.astype(float)
                    y = LabelEncoder().fit_transform(df['Group'].values)
                    return X, y, name
            return None, None, name

        for base in [data_dir, local_data_dir]:
            fp = os.path.join(base, frel)
            if not Path(fp).exists():
                continue
            if did == 1:
                df = pd.read_csv(fp, header=None)
                X = df.iloc[:, :-1].values; y = df.iloc[:, -1].values
            elif did == 2:
                df = pd.read_csv(fp, sep=r'\s+', header=None, engine='python')
                X = df.iloc[:, 1:-1].values; y = df.iloc[:, -1].values
            elif did == 3:
                rows = []
                with open(fp) as f:
                    started = False
                    for ln in f:
                        if '@data' in ln.lower():
                            started = True; continue
                        if started and ln.strip() and not ln.startswith('%'):
                            r = ln.strip().split(',')
                            if len(r) >= 2:
                                rows.append(r)
                df = pd.DataFrame(rows)
                X = df.iloc[:, :-1].values.astype(float)
                y = df.iloc[:, -1].values
            elif did == 4:
                df = pd.read_csv(fp, header=None).dropna(how='all')
                df = df[df.iloc[:, -1].astype(str).str.strip() != '']
                X = df.iloc[:, :-1].values; y = df.iloc[:, -1].values
            elif did in (5, 8):
                df = pd.read_csv(fp, header=None)
                X = df.iloc[:, :-1].values; y = df.iloc[:, -1].values
            elif did == 6:
                df = pd.read_csv(fp, header=None)
                X = df.iloc[:, 1:-1].values; y = df.iloc[:, -1].values
            elif did == 7:
                df = pd.read_csv(fp, sep=r'\s+', header=None, engine='python')
                X = df.iloc[:, :-1].values; y = df.iloc[:, -1].values
            elif did == 9:
                df = pd.read_csv(fp, header=None)
                X = df.iloc[:, 1:-1].values; y = df.iloc[:, -1].values
            elif did == 10:
                df = pd.read_csv(fp, sep=r'\s+', header=None, engine='python')
                X = df.iloc[:, 1:-1].values; y = df.iloc[:, -1].values
            else:
                continue
            for c in range(X.shape[1]):
                try:
                    X[:, c] = X[:, c].astype(float)
                except:
                    X[:, c] = LabelEncoder().fit_transform(X[:, c].astype(str))
            X = X.astype(float)
            y = LabelEncoder().fit_transform(y.astype(str))
            return X, y, name
        return None, None, name
    except Exception as e:
        print(f"  加载 d{did} ({name}) 出错: {e}")
        traceback.print_exc()
        return None, None, name


# ====================================================================
# 8. 子进程评估函数
# ====================================================================
def evaluate_single_task(args):
    """评估单个 (变体, 数据集, fold) 组合"""
    (variant_name, config, dataset_id, fold_idx,
     train_idx_list, test_idx_list,
     X_data, y_data, n_classes, hp, gpu_id, seed) = args

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

        # ------ 特殊变体G: VAE + RF ------
        if config.get('use_rf_classifier', False):
            set_seed(seed + fold_idx)
            input_dim = X_train_s.shape[1]
            latent_dim = config.get('latent_dim', 8)
            vae = VAE(input_dim, latent_dim=latent_dim).to(device)

            Xt = torch.FloatTensor(X_train_s).to(device)
            yt = torch.LongTensor(y_train).to(device)

            # 训练VAE (用优化后的参数)
            vae_lr = hp.get('vae_lr', 0.002)
            vae_epochs = hp.get('vae_epochs', 100)
            kl_weight = hp.get('kl_weight', 0.01)
            vopt = torch.optim.Adam(vae.parameters(), lr=vae_lr, weight_decay=1e-5)
            vae.train()
            for _ in range(vae_epochs):
                vopt.zero_grad()
                rec, mu, lv = vae(Xt)
                loss = F.mse_loss(rec, Xt) + kl_weight * (
                    -0.5 * torch.mean(1 + lv - mu ** 2 - lv.exp()))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
                vopt.step()

            # VAE增强
            n_augment = hp.get('n_augment', 200)
            noise_scale = hp.get('noise_scale', 0.3)
            vae.eval()
            aX, aY = [Xt], [yt]
            with torch.no_grad():
                for i in range(n_augment):
                    idx = i % Xt.size(0)
                    mu, lv = vae.encode(Xt[idx:idx+1])
                    z = vae.reparameterize(mu, lv, ns=noise_scale)
                    aX.append(vae.decoder(z))
                    aY.append(yt[idx:idx+1])
            X_aug = torch.cat(aX).cpu().numpy()
            y_aug = torch.cat(aY).cpu().numpy()

            rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)
            rf.fit(X_aug, y_aug)
            preds = rf.predict(X_test_s)
            acc = accuracy_score(y_test, preds) * 100

            del vae; torch.cuda.empty_cache()
            return {
                'variant': variant_name, 'dataset_id': dataset_id,
                'fold': fold_idx, 'accuracy': acc, 'status': 'ok'
            }

        # ------ 标准消融变体 (A-F) ------
        model = train_ablation_model(
            X_train_s, y_train, X_test_s, y_test,
            n_classes, config, hp, device,
            seed=seed + fold_idx)

        proba = predict_ablation(model, X_train_s, X_test_s, device)
        preds = proba.argmax(axis=1)
        acc = accuracy_score(y_test, preds) * 100

        del model; torch.cuda.empty_cache()
        return {
            'variant': variant_name, 'dataset_id': dataset_id,
            'fold': fold_idx, 'accuracy': acc, 'status': 'ok'
        }

    except Exception as e:
        traceback.print_exc()
        return {
            'variant': variant_name, 'dataset_id': dataset_id,
            'fold': fold_idx, 'accuracy': 0.0, 'status': f'error: {str(e)}'
        }


# ====================================================================
# 9. 主函数
# ====================================================================
def main():
    parser = argparse.ArgumentParser(
        description='VAE-HyperNet-Fusion Ablation Study v2 (Optuna params)')
    parser.add_argument('--datasets', nargs='+', type=int,
                        default=list(range(11)),
                        help='Dataset IDs (default: 0-10)')
    parser.add_argument('--gpus', nargs='+', type=int, default=None,
                        help='GPU IDs (default: auto)')
    parser.add_argument('--no-gpu', action='store_true')
    parser.add_argument('--workers-per-gpu', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--variants', nargs='+', default=None,
                        help='Subset of variant names (default: all)')
    parser.add_argument('--output-dir', type=str,
                        default='/data2/image_identification/src/output',
                        help='Path with optuna_d*.json files + save results')
    args = parser.parse_args()

    print("=" * 80)
    print("  VAE-HyperNet-Fusion 消融实验 v2 (Optuna优化参数)")
    print("=" * 80)
    print(f"  与55版区别: 每个数据集使用Optuna搜索到的最优超参数")
    print()

    # ---- 设备 ----
    if args.no_gpu or not torch.cuda.is_available():
        gpu_ids = []
        n_workers = max(1, mp.cpu_count() // 4)
        print(f"模式: CPU, workers={n_workers}")
    else:
        if args.gpus:
            gpu_ids = args.gpus
        else:
            gpu_ids = list(range(torch.cuda.device_count()))
            if len(gpu_ids) > 2:
                gpu_ids = gpu_ids[2:]
        n_workers = len(gpu_ids) * args.workers_per_gpu
        print(f"模式: GPU {gpu_ids}, workers={n_workers}")

    output_dir = args.output_dir

    # ---- 加载每个数据集的Optuna最优参数 ----
    print("\n加载Optuna最优参数...")
    all_hp = {}  # dataset_id → best_params dict
    for did in args.datasets:
        hp, optuna_data = load_optuna_params(did, output_dir)
        all_hp[did] = hp
        if optuna_data:
            acc_key = 'final_mean_acc' if 'final_mean_acc' in optuna_data else 'final_acc_mean'
            acc = optuna_data.get(acc_key, 'N/A')
            print(f"  d{did}: Optuna best acc={acc}%  "
                  f"trees={hp.get('n_trees')}, depth={hp.get('tree_depth')}, "
                  f"heads={hp.get('n_heads')}, aug={hp.get('n_augment')}, "
                  f"epochs={hp.get('epochs')}")
        else:
            print(f"  d{did}: 使用默认参数 (无Optuna结果)")

    # ---- 加载数据集 ----
    print("\n加载数据集...")
    all_datasets = {}
    for did in args.datasets:
        X, y, dname = load_dataset_by_id(did)
        if X is not None:
            nc = len(np.unique(y))
            all_datasets[did] = {
                'X': X, 'y': y, 'name': dname,
                'n_classes': nc, 'n_samples': len(X),
                'n_features': X.shape[1]
            }
            print(f"  [{did}] {dname}: {len(X)} samples, "
                  f"{X.shape[1]} features, {nc} classes")
        else:
            print(f"  [{did}] {dname}: SKIP")

    if not all_datasets:
        print("没有可用数据集"); return

    # ---- 准备任务 ----
    print("\n准备任务...")
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tasks = []

    for did, dinfo in all_datasets.items():
        X, y = dinfo['X'], dinfo['y']
        nc = dinfo['n_classes']
        folds = list(kfold.split(X, y))
        hp = all_hp[did]

        # 为此数据集构建消融变体 (使用此数据集的优化参数)
        variants = build_ablation_variants(hp)

        # 过滤 (如果指定了子集)
        if args.variants:
            variants = {k: v for k, v in variants.items() if k in args.variants}

        for vname, vcfg in variants.items():
            for fold_idx, (train_idx, test_idx) in enumerate(folds):
                gpu_id = gpu_ids[len(tasks) % len(gpu_ids)] if gpu_ids else 0
                tasks.append((
                    vname, vcfg, did, fold_idx,
                    train_idx.tolist(), test_idx.tolist(),
                    X, y, nc, hp, gpu_id, args.seed
                ))

    total = len(tasks)
    n_variants = 7  # A-G
    if args.variants:
        n_variants = len(args.variants)
    print(f"总任务数: {total} "
          f"({len(all_datasets)} datasets × {n_variants} variants × 5 folds)")

    # ---- RF基线 ----
    print("\n计算RF基线...")
    rf_results = {}
    for did, dinfo in all_datasets.items():
        X, y = dinfo['X'], dinfo['y']
        fold_accs = []
        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
            sc = StandardScaler()
            Xtr = sc.fit_transform(X[train_idx])
            Xte = sc.transform(X[test_idx])
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(Xtr, y[train_idx])
            acc = accuracy_score(y[test_idx], rf.predict(Xte)) * 100
            fold_accs.append(acc)
        m, s = np.mean(fold_accs), np.std(fold_accs)
        rf_results[did] = {'mean': m, 'std': s, 'fold_accs': fold_accs}
        print(f"  [{did}] {dinfo['name']}: RF = {m:.1f}±{s:.1f}%")

    # ---- 并行运行 ----
    print(f"\n开始并行消融实验 ({n_workers} workers)...")
    t0 = time.time()

    if n_workers > 1 and total > 1:
        ctx = mp.get_context('spawn')
        with ctx.Pool(n_workers) as pool:
            results_list = []
            for i, result in enumerate(
                    pool.imap_unordered(evaluate_single_task, tasks)):
                results_list.append(result)
                done = i + 1
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done) if done > 0 else 0
                if done % 10 == 0 or done == total:
                    # 打印状态: 成功/失败统计
                    ok = sum(1 for r in results_list if r['status'] == 'ok')
                    err = done - ok
                    print(f"  进度: {done}/{total} ({done/total*100:.0f}%) "
                          f"ok={ok} err={err} "
                          f"已用{elapsed:.0f}s 预计还需{eta:.0f}s",
                          flush=True)
    else:
        results_list = []
        for i, task in enumerate(tasks):
            result = evaluate_single_task(task)
            results_list.append(result)
            print(f"  进度: {i+1}/{total}")

    total_time = time.time() - t0
    print(f"\n总耗时: {total_time:.0f}s ({total_time/60:.1f}min)")

    # ---- 汇总 ----
    print("\n" + "=" * 80)
    print("  消融实验 v2 结果汇总 (Optuna优化参数)")
    print("=" * 80)

    summary = defaultdict(lambda: defaultdict(list))
    for r in results_list:
        if r['status'] == 'ok':
            summary[r['variant']][r['dataset_id']].append(r['accuracy'])

    # 确定打印的变体顺序
    variant_order = ['A_Full_Model', 'B_No_VAE_Aug', 'C_No_Interpolation',
                     'D_No_HyperNet', 'E_No_Ensemble', 'F_No_MultiHead',
                     'G_VAE_plus_RF']
    if args.variants:
        variant_order = [v for v in variant_order if v in args.variants]

    # 打印表格
    header = f"{'Dataset':<15} {'RF':>12}"
    for vn in variant_order:
        short = vn.split('_', 1)[1] if '_' in vn else vn
        header += f" {short:>14}"
    print(header)
    print("-" * len(header))

    full_results = {
        'experiment': 'Ablation Study v2 (Optuna optimized params)',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'seed': args.seed,
        'n_folds': 5,
        'optuna_params_per_dataset': {str(k): v for k, v in all_hp.items()},
        'rf_baseline': {},
        'ablation_results': {},
        'per_dataset': {}
    }

    for did in sorted(all_datasets.keys()):
        dname = all_datasets[did]['name']
        rf_str = f"{rf_results[did]['mean']:.1f}±{rf_results[did]['std']:.1f}"
        row = f"{dname:<15} {rf_str:>12}"

        dataset_results = {
            'RF': {'mean': round(rf_results[did]['mean'], 2),
                   'std': round(rf_results[did]['std'], 2),
                   'fold_accs': [round(a, 2) for a in rf_results[did]['fold_accs']]},
            'optuna_params': all_hp[did]
        }

        for vn in variant_order:
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

    # 平均行
    print("-" * len(header))
    avg_row = f"{'AVERAGE':<15}"
    rf_means = [rf_results[did]['mean'] for did in sorted(all_datasets.keys())]
    avg_row += f" {np.mean(rf_means):>7.1f}     "
    for vn in variant_order:
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

    # ---- 组件贡献分析 ----
    print("\n" + "=" * 80)
    print("  各组件贡献分析 (Full Model - 去掉该组件 = Δ)")
    print("=" * 80)

    component_effects = {
        'VAE Augmentation': ('A_Full_Model', 'B_No_VAE_Aug'),
        'Interpolation':    ('A_Full_Model', 'C_No_Interpolation'),
        'HyperNet':         ('A_Full_Model', 'D_No_HyperNet'),
        'Ensembling':       ('A_Full_Model', 'E_No_Ensemble'),
        'Multi-head':       ('A_Full_Model', 'F_No_MultiHead'),
    }

    print(f"\n{'Component':<22} ", end='')
    for did in sorted(all_datasets.keys()):
        print(f" {all_datasets[did]['name']:>10}", end='')
    print(f" {'AVG':>8}")
    print("-" * (22 + 11 * len(all_datasets) + 9))

    full_results['component_effects'] = {}
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

        full_results['component_effects'][comp_name] = {
            'per_dataset_delta': [round(d, 2) for d in deltas],
            'avg_delta': round(float(np.mean(deltas)), 2) if deltas else 0
        }

    print("\n说明: 正值=该组件有正向贡献(去掉后准确率下降)")

    # ---- 与55版(默认参数)对比 ----
    print("\n" + "=" * 80)
    print("  v2 vs v1 (默认参数) — Full Model A对比")
    print("=" * 80)

    # 尝试加载55版结果
    v1_files = sorted(Path(output_dir).glob('55_ablation_*.json'))
    if v1_files:
        with open(v1_files[-1]) as f:
            v1_data = json.load(f)
        print(f"\nv1 results from: {v1_files[-1].name}")
        print(f"{'Dataset':<15} {'v1 Full':>12} {'v2 Full':>12} {'Δ':>8}")
        print("-" * 49)
        for did in sorted(all_datasets.keys()):
            dname = all_datasets[did]['name']
            v2_accs = summary['A_Full_Model'].get(did, [])
            v2_m = np.mean(v2_accs) if v2_accs else 0

            v1_m = 0
            if dname in v1_data.get('per_dataset', {}):
                v1_entry = v1_data['per_dataset'][dname].get('A_Full_Model', {})
                v1_m = v1_entry.get('mean', 0) if isinstance(v1_entry, dict) else 0

            delta = v2_m - v1_m
            sign = '+' if delta >= 0 else ''
            print(f"{dname:<15} {v1_m:>10.1f}% {v2_m:>10.1f}% {sign}{delta:>6.1f}%")
    else:
        print("\n  (未找到55版结果文件, 跳过v1/v2对比)")

    # ---- 保存结果 ----
    output_dirs = [
        Path(output_dir),
        Path('/Users/jinmingzhang/D/1/1Postg/Sem_13_Thesis/'
             '5_2026.01.28/src/final/output')
    ]
    for od in output_dirs:
        try:
            od.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            outf = od / f'62_ablation_v2_{timestamp}.json'
            with open(outf, 'w') as f:
                json.dump(full_results, f, indent=2, ensure_ascii=False)
            print(f"\n结果已保存: {outf}")
            break
        except Exception as e:
            print(f"保存到 {od} 失败: {e}")

    # ---- LaTeX 表格 ----
    print("\n" + "=" * 80)
    print("  LaTeX Table (Ablation Study v2)")
    print("=" * 80)
    print()
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Ablation study results (\%) with Optuna-optimized hyperparameters. "
          r"Each column removes one component from the full model.}")
    print(r"\label{tab:ablation}")
    print(r"\resizebox{\textwidth}{!}{")
    print(r"\begin{tabular}{l" + "c" * (2 + len(variant_order)) + "}")
    print(r"\toprule")

    short_labels = {
        'A_Full_Model': 'Full',
        'B_No_VAE_Aug': 'w/o VAE',
        'C_No_Interpolation': 'w/o Interp.',
        'D_No_HyperNet': 'w/o HyperNet',
        'E_No_Ensemble': 'w/o Ensem.',
        'F_No_MultiHead': 'w/o M-Head',
        'G_VAE_plus_RF': 'VAE+RF',
    }

    latex_header = "Dataset & RF"
    for vn in variant_order:
        latex_header += f" & {short_labels.get(vn, vn)}"
    latex_header += r" \\"
    print(latex_header)
    print(r"\midrule")

    for did in sorted(all_datasets.keys()):
        dname = all_datasets[did]['name']
        rf_m, rf_s = rf_results[did]['mean'], rf_results[did]['std']
        line = f"{dname} & {rf_m:.1f}$\\pm${rf_s:.1f}"

        # 在所有变体+RF中找最大
        best_acc, best_vn = rf_m, 'RF'
        for vn in variant_order:
            accs = summary[vn].get(did, [])
            if accs and np.mean(accs) > best_acc:
                best_acc = np.mean(accs)
                best_vn = vn

        if best_vn == 'RF':
            line = f"{dname} & \\textbf{{{rf_m:.1f}$\\pm${rf_s:.1f}}}"

        for vn in variant_order:
            accs = summary[vn].get(did, [])
            if accs:
                m, s = np.mean(accs), np.std(accs)
                if vn == best_vn:
                    line += f" & \\textbf{{{m:.1f}$\\pm${s:.1f}}}"
                else:
                    line += f" & {m:.1f}$\\pm${s:.1f}"
            else:
                line += " & --"
        line += r" \\"
        print(line)

    print(r"\midrule")
    avg_line = "Average"
    avg_line += f" & {np.mean(rf_means):.1f}"
    for vn in variant_order:
        vn_means = [np.mean(summary[vn].get(did, [])) for did in sorted(all_datasets.keys())
                    if summary[vn].get(did, [])]
        avg_line += f" & {np.mean(vn_means):.1f}" if vn_means else " & --"
    avg_line += r" \\"
    print(avg_line)

    print(r"\bottomrule")
    print(r"\end{tabular}}")
    print(r"\end{table}")

    # ---- 组件贡献 LaTeX 表 ----
    print()
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Contribution of each component (accuracy difference \%).}")
    print(r"\label{tab:component_contribution}")
    print(r"\begin{tabular}{l" + "c" * (len(all_datasets) + 1) + "}")
    print(r"\toprule")
    ch = "Component"
    for did in sorted(all_datasets.keys()):
        ch += f" & {all_datasets[did]['name']}"
    ch += r" & Avg. \\"
    print(ch)
    print(r"\midrule")

    for comp_name, (full_key, ablated_key) in component_effects.items():
        cl = comp_name
        deltas = []
        for did in sorted(all_datasets.keys()):
            fa = summary[full_key].get(did, [])
            aa = summary[ablated_key].get(did, [])
            if fa and aa:
                d = np.mean(fa) - np.mean(aa)
                deltas.append(d)
                sign = '+' if d >= 0 else ''
                cl += f" & {sign}{d:.1f}"
            else:
                cl += " & --"
        if deltas:
            ad = np.mean(deltas)
            sign = '+' if ad >= 0 else ''
            cl += f" & {sign}{ad:.1f}"
        else:
            cl += " & --"
        cl += r" \\"
        print(cl)

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    print("\n消融实验 v2 完成！")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
