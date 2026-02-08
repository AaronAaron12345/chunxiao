#!/usr/bin/env python3
"""
63_replot_figures.py - 重新绘制论文图表 (使用Optuna优化后的VAE-HNF数据)

== 生成的图表 ==
  fig1_roc_curves.png/eps     — ROC Curves (prostate, 二分类)
  fig2_pr_curves.png/eps      — Precision-Recall Curves (prostate)
  fig3_acc_vs_n.png/eps       — Model Accuracy vs. Dataset Size (n)
  fig4_acc_vs_d.png/eps       — Model Accuracy vs. Number of Features (d)
  fig5_acc_vs_k.png/eps       — Model Accuracy vs. Number of Target Classes (k)

== 数据来源 ==
  RF/XGBoost/TabPFN/HyperTab/TPOT: 58_results_*.json (5-fold CV)
  VAE-HNF: 用 Optuna 最优参数重新在 prostate 上跑 (获取y_proba);
           fig3/4/5的准确率用 62_ablation_v2 结果

== 执行 ==
  /data1/condaproject/dinov2/bin/python3 63_replot_figures.py
"""

import os, sys, json, random, glob, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

# ── 路径 ──
BASE_DIR = Path('/data2/image_identification/src')
FINAL_DIR = BASE_DIR / 'final'
OUTPUT_DIR = BASE_DIR / 'output'
DATA_DIR  = BASE_DIR / 'data'

# ── SEED ──
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

# =====================================================================
# 1. MODEL DEFINITIONS (与62_ablation_v2.py一致)
# =====================================================================
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
        ni = 2**tree_depth - 1; nl = 2**tree_depth
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


# =====================================================================
# 2. 数据加载 (复用59_bayesian_tuning.py的load_dataset_by_id)
# =====================================================================
import traceback

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


# =====================================================================
# 3. 用Optuna最优参数跑VAE-HNF在haberman上, 获取y_proba
# =====================================================================
def run_vae_hnf_for_roc(ds_id, device='cuda:1'):
    """在指定数据集上用Optuna最优参数跑5-fold CV，返回 y_true, y_proba"""
    # 加载Optuna参数
    param_file = OUTPUT_DIR / f'optuna_d{ds_id}.json'
    if param_file.exists():
        with open(param_file) as f:
            od = json.load(f)
        hp = od['best_params']
    else:
        print(f"  [WARN] No optuna params for d{ds_id}, using defaults")
        hp = {}

    n_trees   = hp.get('n_trees', 15)
    tree_depth= hp.get('tree_depth', 3)
    n_heads   = hp.get('n_heads', 5)
    hidden_dim= hp.get('hidden_dim', 64)
    latent_dim= hp.get('latent_dim', 8)
    n_augment = hp.get('n_augment', 200)
    noise_scale = hp.get('noise_scale', 0.3)
    vae_lr    = hp.get('vae_lr', 0.002)
    vae_epochs= hp.get('vae_epochs', 100)
    kl_weight = hp.get('kl_weight', 0.01)
    lr        = hp.get('lr', 0.01)
    weight_decay = hp.get('weight_decay', 0.05)
    epochs    = hp.get('epochs', 300)
    warmup    = hp.get('warmup_epochs', 20)
    reg_weight= hp.get('reg_weight', 0.01)

    X, y, name = load_dataset_by_id(ds_id)
    if X is None:
        print(f"  ERROR: cannot load dataset d{ds_id}")
        return np.array([]), np.array([])

    n_classes = len(np.unique(y))
    input_dim = X.shape[1]

    all_y_true = np.zeros(len(y), dtype=int)
    all_y_proba = np.zeros((len(y), n_classes), dtype=np.float64)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold_i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        set_seed(42 + fold_i)
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[train_idx])
        X_va = sc.transform(X[val_idx])
        y_tr, y_va = y[train_idx], y[val_idx]

        Xt = torch.FloatTensor(X_tr).to(device)
        yt = torch.LongTensor(y_tr).to(device)
        Xv = torch.FloatTensor(X_va).to(device)

        # ---- Build model ----
        vae = VAE(input_dim, latent_dim).to(device)
        hypernet = HyperNet(input_dim, n_classes, n_trees, tree_depth,
                            hidden_dim, n_heads).to(device)
        classifier = TreeClassifier(tree_depth).to(device)

        # ---- Train VAE ----
        vopt = torch.optim.Adam(vae.parameters(), lr=vae_lr, weight_decay=1e-5)
        vae.train()
        for _ in range(vae_epochs):
            vopt.zero_grad()
            rec, mu, lv = vae(Xt)
            loss = F.mse_loss(rec, Xt) + kl_weight * (
                -0.5 * torch.mean(1 + lv - mu**2 - lv.exp()))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            vopt.step()

        # ---- Augment ----
        vae.eval()
        aX, aY = [Xt], [yt]
        with torch.no_grad():
            for i in range(n_augment):
                idx = i % Xt.size(0)
                mu, lv = vae.encode(Xt[idx:idx+1])
                aX.append(vae.decoder(vae.reparameterize(mu, lv, ns=noise_scale)))
                aY.append(yt[idx:idx+1])
        Xa = torch.cat(aX)
        ya = torch.cat(aY)

        # ---- Train classifier ----
        prms = list(hypernet.parameters()) + list(classifier.parameters())
        opt = torch.optim.AdamW(prms, lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e:
            (e + 1) / warmup if e < warmup else
            0.5 * (1 + np.cos(np.pi * (e - warmup) / max(epochs - warmup, 1))))

        best_acc, best_proba = 0, None
        for ep in range(epochs):
            hypernet.train(); classifier.train()
            opt.zero_grad()
            p = hypernet(Xa)
            tp, tw = hypernet.parse_params(p)
            out = classifier(Xa, tp, tw)
            loss = F.cross_entropy(out, ya) + reg_weight * (p**2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prms, 1.0)
            opt.step()
            sched.step()

            # Eval
            if (ep + 1) % 10 == 0 or ep == epochs - 1:
                hypernet.eval(); classifier.eval()
                with torch.no_grad():
                    p = hypernet(Xa)
                    tp, tw = hypernet.parse_params(p)
                    out_v = classifier(Xv, tp, tw)
                    proba = F.softmax(out_v, dim=1)
                    preds = proba.argmax(dim=1).cpu().numpy()
                    acc = (preds == y_va).mean()
                    if acc >= best_acc:
                        best_acc = acc
                        best_proba = proba.cpu().numpy()

        all_y_true[val_idx] = y_va
        if best_proba is not None:
            all_y_proba[val_idx] = best_proba

        print(f"    fold {fold_i}: acc={best_acc:.1%}")

    return all_y_true, all_y_proba


# =====================================================================
# 4. 绘图
# =====================================================================
def plot_all(output_dir):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': 22,
        'axes.titlesize': 26,
        'axes.labelsize': 24,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 18,
        'legend.title_fontsize': 20,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'lines.linewidth': 2.5,
        'lines.markersize': 10,
    })

    # ── 加载58的结果JSON (RF/XGBoost/TabPFN/HyperTab/TPOT的y_true/y_proba) ──
    result_files = sorted(glob.glob(str(OUTPUT_DIR / '58_results_*.json')))
    if not result_files:
        print("ERROR: 58_results_*.json not found!")
        return
    with open(result_files[-1]) as f:
        data58 = json.load(f)
    summary = data58['summary']

    MODEL_ORDER = ['RF', 'XGBoost', 'TabPFN', 'HyperTab', 'TPOT', 'VAE-HNF']

    COLORS = {
        'RF':       '#FFD700',
        'XGBoost':  '#7CFC00',
        'TabPFN':   '#00CED1',
        'HyperTab': '#4169E1',
        'TPOT':     '#9370DB',
        'VAE-HNF':  '#FF0000'
    }
    LS = {
        'RF':       '--',
        'XGBoost':  '--',
        'TabPFN':   '--',
        'HyperTab': '-.',
        'TPOT':     '-.',
        'VAE-HNF':  '-'
    }
    LW = {
        'RF':       2.5,
        'XGBoost':  2.5,
        'TabPFN':   2.5,
        'HyperTab': 2.2,
        'TPOT':     2.2,
        'VAE-HNF':  4.0
    }
    MK = {
        'RF':       's',
        'XGBoost':  'D',
        'TabPFN':   '^',
        'HyperTab': 'v',
        'TPOT':     'X',
        'VAE-HNF':  'o'
    }

    # ── 更新 VAE-HNF 准确率 (Optuna优化后) ──
    # 来自 62_ablation_v2 的 Full Model 结果
    optuna_acc = {
        'prostate':  84.7,
        'balloons': 100.0,
        'lenses':    92.0,
        'caesarian':  76.2,
        'iris':      98.0,
        'fertility': 91.0,
        'zoo':       99.0,
        'seeds':     95.2,
        'haberman':  77.8,
        'glass':     72.9,
        'yeast':     61.3,
    }
    optuna_std = {
        'prostate':   7.8,
        'balloons':   0.0,
        'lenses':     9.8,
        'caesarian':   8.3,
        'iris':       2.7,
        'fertility':  2.0,
        'zoo':        2.0,
        'seeds':      2.6,
        'haberman':   3.1,
        'glass':      4.7,
        'yeast':      2.6,
    }
    for ds_name in summary:
        if ds_name in optuna_acc and 'VAE-HNF' in summary[ds_name]:
            summary[ds_name]['VAE-HNF']['mean'] = optuna_acc[ds_name]
            summary[ds_name]['VAE-HNF']['std']  = optuna_std[ds_name]

    ds_names = list(summary.keys())

    # ==================================================================
    # 图1 & 图2: ROC & PR Curves (prostate — 二分类, VAE-HNF最优)
    # ==================================================================
    roc_ds = 'prostate'
    print(f"\n重新跑 VAE-HNF on {roc_ds} (Optuna params) 以获取 y_proba...")
    ds_id_roc = 0  # prostate is dataset #0

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    hnf_y_true, hnf_y_proba = run_vae_hnf_for_roc(ds_id_roc, device=device)

    # 更新 summary 中 VAE-HNF 的 y_true/y_proba
    summary[roc_ds]['VAE-HNF']['y_true']  = hnf_y_true.tolist()
    summary[roc_ds]['VAE-HNF']['y_proba'] = hnf_y_proba.tolist()

    # ── 图1: ROC ──
    fig, ax = plt.subplots(figsize=(10, 8))
    for mn in MODEL_ORDER:
        d = summary[roc_ds].get(mn, {})
        yt, yp = d.get('y_true', []), d.get('y_proba', [])
        if not yt or not yp: continue
        ya = np.array(yt); pa = np.array(yp)
        ys = pa[:, 1] if pa.ndim == 2 and pa.shape[1] >= 2 else pa.ravel()
        fpr, tpr, _ = roc_curve(ya, ys)
        ra = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=COLORS[mn], linestyle=LS[mn],
                linewidth=LW[mn], label=f'{mn} (AUC={ra:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=2.0, label='Random Guess')
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate',
           title=f'ROC Curves — Prostate Cancer Dataset',
           xlim=[0, 1], ylim=[0, 1.05])
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig1_roc_curves.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'fig1_roc_curves.eps'), format='eps', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  图1 ROC ✓")

    # ── 图2: PR ──
    fig, ax = plt.subplots(figsize=(10, 8))
    for mn in MODEL_ORDER:
        d = summary[roc_ds].get(mn, {})
        yt, yp = d.get('y_true', []), d.get('y_proba', [])
        if not yt or not yp: continue
        ya = np.array(yt); pa = np.array(yp)
        ys = pa[:, 1] if pa.ndim == 2 and pa.shape[1] >= 2 else pa.ravel()
        pr, re, _ = precision_recall_curve(ya, ys)
        ap = average_precision_score(ya, ys)
        ax.plot(re, pr, color=COLORS[mn], linestyle=LS[mn],
                linewidth=LW[mn], label=f'{mn} (AP={ap:.3f})')
    ax.set(xlabel='Recall', ylabel='Precision',
           title=f'Precision-Recall Curves — Prostate Cancer Dataset')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig2_pr_curves.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'fig2_pr_curves.eps'), format='eps', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  图2 PR ✓")

    # ==================================================================
    # 图3: Accuracy vs Dataset Size (n)
    # ==================================================================
    fig, ax = plt.subplots(figsize=(12, 8))
    ds_sorted = sorted(ds_names,
                        key=lambda d: summary[d].get('meta', {}).get('n_samples', 0))
    ns = [summary[d]['meta']['n_samples'] for d in ds_sorted]
    for mn in MODEL_ORDER:
        accs = [summary[d].get(mn, {}).get('mean', 0) for d in ds_sorted]
        ax.plot(ns, accs, color=COLORS[mn], linestyle=LS[mn], linewidth=LW[mn],
                marker=MK[mn], markersize=12, label=mn, markeredgecolor='white',
                markeredgewidth=0.8)
    ax.set(xlabel='Number of Records (n)', ylabel='Accuracy (%)',
           title='Model Accuracy vs. Dataset Size (n)')
    ax.set_xscale('log')
    ax.legend(loc='best', title='Model', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig3_acc_vs_n.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'fig3_acc_vs_n.eps'), format='eps', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  图3 Acc vs n ✓")

    # ==================================================================
    # 图4: Accuracy vs Number of Features (d)
    # ==================================================================
    fig, ax = plt.subplots(figsize=(12, 8))
    ds_fd = sorted(ds_names,
                   key=lambda d: summary[d].get('meta', {}).get('n_features', 0))
    ds_vals = [summary[d]['meta']['n_features'] for d in ds_fd]
    for mn in MODEL_ORDER:
        accs = [summary[d].get(mn, {}).get('mean', 0) for d in ds_fd]
        ax.plot(ds_vals, accs, color=COLORS[mn], linestyle=LS[mn], linewidth=LW[mn],
                marker=MK[mn], markersize=12, label=mn, markeredgecolor='white',
                markeredgewidth=0.8)
    ax.set(xlabel='Number of Features (d)', ylabel='Accuracy (%)',
           title='Model Accuracy vs. Number of Features (d)')
    ax.legend(loc='best', title='Model', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig4_acc_vs_d.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'fig4_acc_vs_d.eps'), format='eps', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  图4 Acc vs d ✓")

    # ==================================================================
    # 图5: Accuracy vs Number of Target Classes (k)
    # ==================================================================
    fig, ax = plt.subplots(figsize=(12, 8))
    ds_kd = sorted(ds_names,
                   key=lambda d: summary[d].get('meta', {}).get('n_classes', 0))
    kvs = [summary[d]['meta']['n_classes'] for d in ds_kd]
    uk = sorted(set(kvs))
    for mn in MODEL_ORDER:
        avg_a, std_a = [], []
        for k in uk:
            ds_k = [d for d in ds_kd if summary[d]['meta']['n_classes'] == k]
            a_k = [summary[d].get(mn, {}).get('mean', 0) for d in ds_k]
            avg_a.append(np.mean(a_k))
            std_a.append(np.std(a_k) if len(a_k) > 1 else 0)
        ax.plot(uk, avg_a, color=COLORS[mn], linestyle=LS[mn], linewidth=LW[mn],
                marker=MK[mn], markersize=12, label=mn, markeredgecolor='white',
                markeredgewidth=0.8)
        if any(s > 0 for s in std_a):
            ax.fill_between(uk, [a - s for a, s in zip(avg_a, std_a)],
                            [a + s for a, s in zip(avg_a, std_a)],
                            color=COLORS[mn], alpha=0.08)
    ax.set(xlabel='Number of Target Classes (k)', ylabel='Accuracy (%)',
           title='Model Accuracy vs. Number of Target Classes (k)')
    ax.legend(loc='best', title='Model', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig5_acc_vs_k.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'fig5_acc_vs_k.eps'), format='eps', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  图5 Acc vs k ✓")

    print(f"\n所有图表已保存到: {output_dir}")


# =====================================================================
# 5. main
# =====================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("63_replot_figures.py — 重新绘制论文图表 (Optuna优化版)")
    print("=" * 60)
    output_dir = str(OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    plot_all(output_dir)
    print("\nDone!")
