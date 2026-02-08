#!/usr/bin/env python3
"""
59_bayesian_tuning.py - 贝叶斯优化(Optuna TPE)调参 + 6模型新版对比

== 执行流程 ==
Phase 1: Optuna 贝叶斯搜索 VAE-HNF 最优超参数
         每个数据集一个独立subprocess → 多GPU并行 → 每个GPU上多进程并行
Phase 2: 用最优参数做 VAE-HNF 5折CV最终评估 (收集预测结果用于画图)
Phase 3: 合并58版其他5模型结果 → 新表格 + 新图片 (保存为 v2, 不覆盖旧版)

== 服务器运行 ==
  nohup /data1/condaproject/dinov2/bin/python3 -u 59_bayesian_tuning.py \
        --gpus 1 2 3 4 5 --n_trials 50 > 59_log.txt 2>&1 &

== Worker 模式 (内部自动调用) ==
  python 59_bayesian_tuning.py --worker --gpu_id 1 --datasets 0
"""

import os, sys, time, json, random, traceback, argparse, logging, glob, subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, roc_curve, auc,
                             precision_recall_curve, average_precision_score)
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ---- Optuna ----
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'optuna', '-q'])
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner

optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# ====================================================================
# 1. SEED
# ====================================================================
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ====================================================================
# 2. MODEL DEFINITIONS (from 47_stable_37.py, 全参数化)
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
        return self.decoder(self.reparameterize(mu, lv, ns)), mu, lv


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


class VAEHyperNetFusion(nn.Module):
    """VAE-HyperNet-Fusion 全参数化版本"""
    def __init__(self, input_dim, n_classes, n_trees=15, tree_depth=3,
                 n_heads=5, hidden_dim=64, latent_dim=8, dropout=0.15):
        super().__init__()
        self.vae = VAE(input_dim, latent_dim=latent_dim)
        self.hypernet = HyperNet(input_dim, n_classes, n_trees, tree_depth,
                                 hidden_dim, n_heads, dropout)
        self.classifier = TreeClassifier(tree_depth)

    def augment(self, X, y, n_aug=200, ns=0.3):
        self.vae.eval()
        aX, aY = [X], [y]
        with torch.no_grad():
            for i in range(n_aug):
                idx = i % X.size(0)
                mu, lv = self.vae.encode(X[idx:idx+1])
                aX.append(self.vae.decoder(self.vae.reparameterize(mu, lv, ns=ns)))
                aY.append(y[idx:idx+1])
        return torch.cat(aX), torch.cat(aY)

    def forward(self, X_train, X_test):
        p = self.hypernet(X_train)
        tp, tw = self.hypernet.parse_params(p)
        return self.classifier(X_test, tp, tw), p


# ====================================================================
# 3. DATASET LOADING (from 58_full_comparison_v3.py)
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
# 4. TRAIN & PREDICT (全参数化)
# ====================================================================
def train_model(X_tr, y_tr, X_val, y_val, n_classes, device, hp, seed=42):
    """用指定超参数训练 VAE-HNF"""
    set_seed(seed)
    dim = X_tr.shape[1]
    model = VAEHyperNetFusion(
        dim, n_classes,
        n_trees=hp['n_trees'],
        tree_depth=hp['tree_depth'],
        n_heads=hp['n_heads'],
        hidden_dim=hp['hidden_dim'],
        latent_dim=hp['latent_dim'],
        dropout=hp['dropout']
    ).to(device)

    Xt = torch.FloatTensor(X_tr).to(device)
    yt = torch.LongTensor(y_tr).to(device)
    Xv = torch.FloatTensor(X_val).to(device)

    # Phase 1: VAE
    vopt = torch.optim.Adam(model.vae.parameters(), lr=hp['vae_lr'], weight_decay=1e-5)
    model.vae.train()
    for _ in range(hp['vae_epochs']):
        vopt.zero_grad()
        rec, mu, lv = model.vae(Xt)
        loss = F.mse_loss(rec, Xt) + hp['kl_weight'] * (
            -0.5 * torch.mean(1 + lv - mu ** 2 - lv.exp()))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.vae.parameters(), 1.0)
        vopt.step()

    # Augment
    model.vae.eval()
    with torch.no_grad():
        Xa, ya = model.augment(Xt, yt, hp['n_augment'], hp['noise_scale'])

    # Phase 2: HyperNet + Tree
    prms = list(model.hypernet.parameters()) + list(model.classifier.parameters())
    opt = torch.optim.AdamW(prms, lr=hp['lr'], weight_decay=hp['weight_decay'])
    wu = hp['warmup_epochs']
    epochs = hp['epochs']
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e:
        (e + 1) / wu if e < wu else
        0.5 * (1 + np.cos(np.pi * (e - wu) / max(epochs - wu, 1))))

    best_acc, best_st, noimpr = 0, None, 0
    for ep in range(epochs):
        model.hypernet.train(); model.classifier.train()
        opt.zero_grad()
        out, p = model(Xa, Xa)
        loss = F.cross_entropy(out, ya) + hp['reg_weight'] * (p ** 2).mean()
        if ep > wu:
            model.eval()
            with torch.no_grad():
                o2, _ = model(Xa, Xa)
            model.hypernet.train(); model.classifier.train()
            loss = loss + 0.1 * F.mse_loss(out, o2.detach())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step(); sched.step()

        if (ep + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                vo, _ = model(Xt, Xv)
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


def predict_model(model, X_tr, X_te, device):
    model.eval()
    with torch.no_grad():
        out, _ = model(torch.FloatTensor(X_tr).to(device),
                       torch.FloatTensor(X_te).to(device))
        return F.softmax(out, dim=1).cpu().numpy()


def ensure_proba_columns(proba, n_classes, classes=None):
    if proba is None:
        return None
    if proba.ndim == 1:
        proba = np.column_stack([1 - proba, proba])
    if proba.shape[1] < n_classes:
        fp = np.zeros((proba.shape[0], n_classes))
        if classes is not None:
            for i, c in enumerate(classes):
                if c < n_classes:
                    fp[:, c] = proba[:, i]
        else:
            fp[:, :proba.shape[1]] = proba
        proba = fp
    return proba


# ====================================================================
# 5. OPTUNA OBJECTIVE + DEFAULT PARAMS
# ====================================================================
DEFAULT_HP = {
    'n_trees': 15, 'tree_depth': 3, 'n_heads': 5,
    'hidden_dim': 64, 'latent_dim': 8, 'dropout': 0.15,
    'n_augment': 200, 'noise_scale': 0.3,
    'vae_lr': 0.002, 'vae_epochs': 100, 'kl_weight': 0.01,
    'lr': 0.01, 'weight_decay': 0.05, 'epochs': 300,
    'warmup_epochs': 20, 'reg_weight': 0.01,
}


def make_objective(X, y, n_classes, device):
    """创建 Optuna 目标函数 (5折CV平均准确率)"""
    def objective(trial):
        hp = {
            'n_trees':       trial.suggest_int('n_trees', 5, 30, step=5),
            'tree_depth':    trial.suggest_int('tree_depth', 2, 5),
            'n_heads':       trial.suggest_int('n_heads', 3, 9, step=2),
            'hidden_dim':    trial.suggest_categorical('hidden_dim', [32, 64, 128]),
            'latent_dim':    trial.suggest_categorical('latent_dim', [4, 8, 16]),
            'dropout':       trial.suggest_float('dropout', 0.05, 0.25),
            'n_augment':     trial.suggest_int('n_augment', 50, 800, step=50),
            'noise_scale':   trial.suggest_float('noise_scale', 0.1, 1.0),
            'vae_lr':        trial.suggest_float('vae_lr', 5e-4, 0.01, log=True),
            'vae_epochs':    trial.suggest_int('vae_epochs', 50, 200, step=25),
            'kl_weight':     trial.suggest_float('kl_weight', 1e-3, 0.05, log=True),
            'lr':            trial.suggest_float('lr', 1e-3, 0.05, log=True),
            'weight_decay':  trial.suggest_float('weight_decay', 0.01, 0.1, log=True),
            'epochs':        trial.suggest_categorical('epochs', [200, 300, 500]),
            'warmup_epochs': trial.suggest_int('warmup_epochs', 10, 40, step=5),
            'reg_weight':    trial.suggest_float('reg_weight', 1e-3, 0.05, log=True),
        }

        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accs = []
        for fi, (tri, tei) in enumerate(kfold.split(X, y)):
            sc = StandardScaler()
            Xtr = sc.fit_transform(X[tri])
            Xte = sc.transform(X[tei])

            mdl = train_model(Xtr, y[tri], Xte, y[tei], n_classes, device,
                              hp, seed=42 + fi)
            proba = predict_model(mdl, Xtr, Xte, device)
            acc = accuracy_score(y[tei], proba.argmax(1)) * 100
            accs.append(acc)

            # Optuna 剪枝: 如果中间结果太差就提前停止
            trial.report(np.mean(accs), fi)
            if trial.should_prune():
                raise optuna.TrialPruned()

            del mdl; torch.cuda.empty_cache()

        return np.mean(accs)
    return objective


# ====================================================================
# 6. WORKER (每个数据集一个独立进程, Optuna搜索 + 最终评估)
# ====================================================================
def run_worker(gpu_id, did, n_trials, out_dir):
    """单数据集Worker: Optuna搜索最优参数 → 最终5折CV评估 → 保存结果"""
    device = torch.device(f'cuda:{gpu_id}')
    X, y, name = load_dataset_by_id(did)
    if X is None:
        print(f"[GPU{gpu_id}] d{did}: SKIP (load failed)", flush=True)
        return

    nc = len(np.unique(y))
    print(f"[GPU{gpu_id}] d{did} ({name}): n={len(X)} d={X.shape[1]} k={nc} "
          f"-- {n_trials} trials starting", flush=True)

    # ---- Optuna 搜索 ----
    t0 = time.time()
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2))

    # 把默认超参数作为第一组评估 (确保至少不比默认差)
    study.enqueue_trial(DEFAULT_HP)

    obj = make_objective(X, y, nc, device)
    study.optimize(obj, n_trials=n_trials, n_jobs=1, show_progress_bar=False)

    search_time = time.time() - t0
    best = study.best_trial
    hp = best.params

    print(f"[GPU{gpu_id}] d{did} ({name}): Optuna BEST={best.value:.2f}% "
          f"({search_time:.0f}s, {len(study.trials)} trials, "
          f"{len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])} pruned)",
          flush=True)

    # ---- 最终评估 (用最优参数, 记录完整预测) ----
    print(f"[GPU{gpu_id}] d{did} ({name}): Final evaluation...", flush=True)
    t1 = time.time()

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    faccs = []
    yt_all, yp_all, yproba_all = [], [], []

    for fi, (tri, tei) in enumerate(kfold.split(X, y)):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tri])
        Xte = sc.transform(X[tei])

        mdl = train_model(Xtr, y[tri], Xte, y[tei], nc, device, hp, seed=42 + fi)
        proba = predict_model(mdl, Xtr, Xte, device)
        proba = ensure_proba_columns(proba, nc)
        preds = proba.argmax(1)
        acc = accuracy_score(y[tei], preds) * 100
        faccs.append(acc)

        yt_all.extend(y[tei].tolist())
        yp_all.extend(preds.tolist())
        yproba_all.extend(proba.tolist())

        del mdl; torch.cuda.empty_cache()

    eval_time = time.time() - t1

    # ---- 保存结果 ----
    result = {
        'dataset_id': did,
        'name': name,
        'n_samples': int(len(X)),
        'n_features': int(X.shape[1]),
        'n_classes': int(nc),
        'optuna_best_acc': float(best.value),
        'final_mean_acc': float(np.mean(faccs)),
        'final_std_acc': float(np.std(faccs)),
        'fold_accs': [float(round(a, 2)) for a in faccs],
        'best_params': hp,
        'default_params': DEFAULT_HP,
        'n_trials_completed': len([t for t in study.trials
                                   if t.state == optuna.trial.TrialState.COMPLETE]),
        'n_trials_pruned': len([t for t in study.trials
                                if t.state == optuna.trial.TrialState.PRUNED]),
        'search_time_sec': search_time,
        'eval_time_sec': eval_time,
        'gpu_id': gpu_id,
        # 完整预测结果 (用于画图)
        'y_true': yt_all,
        'y_pred': yp_all,
        'y_proba': yproba_all,
        # Top 10 trials
        'top_trials': sorted(
            [{'number': t.number, 'value': t.value, 'params': t.params}
             for t in study.trials
             if t.state == optuna.trial.TrialState.COMPLETE],
            key=lambda t: t['value'], reverse=True)[:10]
    }

    outf = os.path.join(out_dir, f'optuna_d{did}.json')
    with open(outf, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[GPU{gpu_id}] d{did} ({name}): FINAL={np.mean(faccs):.2f}±{np.std(faccs):.2f}% "
          f"(total {search_time + eval_time:.0f}s) → {outf}", flush=True)


# ====================================================================
# 7. PLOT GENERATION (v2 版本, 不覆盖旧图)
# ====================================================================
def generate_plots_v2(summary, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14,
                         'axes.labelsize': 12, 'legend.fontsize': 9,
                         'figure.dpi': 150})

    COLORS = {'RF': '#FFD700', 'XGBoost': '#7CFC00', 'TabPFN': '#00CED1',
              'HyperTab': '#4169E1', 'TPOT': '#9370DB', 'VAE-HNF': '#FF0000'}
    LS = {'RF': '--', 'XGBoost': '--', 'TabPFN': '--',
          'HyperTab': '-.', 'TPOT': '-.', 'VAE-HNF': '-'}
    LW = {'RF': 1.5, 'XGBoost': 1.5, 'TabPFN': 1.5,
          'HyperTab': 1.5, 'TPOT': 1.5, 'VAE-HNF': 3.0}
    MODEL_ORDER = ['RF', 'XGBoost', 'TabPFN', 'HyperTab', 'TPOT', 'VAE-HNF']

    available = [m for m in MODEL_ORDER if any(
        summary.get(d, {}).get(m, {}).get('mean', 0) > 0
        for d in summary if d != 'meta')]
    print(f"  Plot models: {available}")

    # ---- Fig 1: ROC ----
    binary_ds = [d for d in summary
                 if summary[d].get('meta', {}).get('n_classes') == 2]
    roc_ds = None
    for pref in ['haberman', 'caesarian', 'prostate', 'fertility', 'balloons']:
        if pref in binary_ds:
            roc_ds = pref; break
    if roc_ds is None and binary_ds:
        roc_ds = binary_ds[0]

    if roc_ds:
        fig, ax = plt.subplots(figsize=(8, 6))
        for mn in available:
            d = summary[roc_ds].get(mn, {})
            yt, yp = d.get('y_true', []), d.get('y_proba', [])
            if not yt or not yp:
                continue
            ya = np.array(yt)
            pa = np.array(yp)
            ys = pa[:, 1] if pa.ndim == 2 and pa.shape[1] >= 2 else pa.ravel()
            fpr, tpr, _ = roc_curve(ya, ys)
            ra = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=COLORS[mn], linestyle=LS[mn],
                    linewidth=LW[mn], label=f'{mn} (AUC={ra:.2f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=.5, label='Random Guess')
        ax.set(xlabel='FPR', ylabel='TPR',
               title=f'ROC Curves ({roc_ds}) [Optimized]',
               xlim=[0, 1], ylim=[0, 1.05])
        ax.legend(loc='lower right')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'fig1_v2_roc_curves.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  fig1_v2 ROC ({roc_ds}) ✓")

    # ---- Fig 2: PR ----
    if roc_ds:
        fig, ax = plt.subplots(figsize=(8, 6))
        for mn in available:
            d = summary[roc_ds].get(mn, {})
            yt, yp = d.get('y_true', []), d.get('y_proba', [])
            if not yt or not yp:
                continue
            ya = np.array(yt)
            pa = np.array(yp)
            ys = pa[:, 1] if pa.ndim == 2 and pa.shape[1] >= 2 else pa.ravel()
            pr, re, _ = precision_recall_curve(ya, ys)
            ap = average_precision_score(ya, ys)
            ax.plot(re, pr, color=COLORS[mn], linestyle=LS[mn],
                    linewidth=LW[mn], label=f'{mn} (AP={ap:.2f})')
        ax.set(xlabel='Recall', ylabel='Precision',
               title=f'Precision-Recall Curves ({roc_ds}) [Optimized]')
        ax.legend(loc='best')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'fig2_v2_pr_curves.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  fig2_v2 PR ({roc_ds}) ✓")

    # ---- Fig 3: Acc vs n ----
    fig, ax = plt.subplots(figsize=(8, 6))
    ds_sorted = sorted([d for d in summary],
                       key=lambda d: summary[d].get('meta', {}).get('n_samples', 0))
    ns = [summary[d]['meta']['n_samples'] for d in ds_sorted]
    for mn in available:
        accs = [summary[d].get(mn, {}).get('mean', 0) for d in ds_sorted]
        mk = 'o' if mn == 'VAE-HNF' else 's'
        ax.plot(ns, accs, color=COLORS[mn], linestyle=LS[mn], linewidth=LW[mn],
                marker=mk, markersize=5, label=mn)
    ax.set(xlabel='Number of Records (n)', ylabel='Accuracy (%)',
           title='Model Accuracy vs. Dataset Size [Optimized]')
    ax.set_xscale('log')
    ax.legend(loc='best', title='Model')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig3_v2_acc_vs_n.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  fig3_v2 Acc vs n ✓")

    # ---- Fig 4: Acc vs d ----
    fig, ax = plt.subplots(figsize=(8, 6))
    ds_fd = sorted([d for d in summary],
                   key=lambda d: summary[d].get('meta', {}).get('n_features', 0))
    ds_vals = [summary[d]['meta']['n_features'] for d in ds_fd]
    for mn in available:
        accs = [summary[d].get(mn, {}).get('mean', 0) for d in ds_fd]
        mk = 'o' if mn == 'VAE-HNF' else 's'
        ax.plot(ds_vals, accs, color=COLORS[mn], linestyle=LS[mn], linewidth=LW[mn],
                marker=mk, markersize=5, label=mn)
    ax.set(xlabel='Number of Features (d)', ylabel='Accuracy (%)',
           title='Model Accuracy vs. Number of Features [Optimized]')
    ax.legend(loc='best', title='Model')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig4_v2_acc_vs_d.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  fig4_v2 Acc vs d ✓")

    # ---- Fig 5: Acc vs k ----
    fig, ax = plt.subplots(figsize=(8, 6))
    ds_kd = sorted([d for d in summary],
                   key=lambda d: summary[d].get('meta', {}).get('n_classes', 0))
    kvs = [summary[d]['meta']['n_classes'] for d in ds_kd]
    uk = sorted(set(kvs))
    for mn in available:
        avg_a, std_a = [], []
        for k in uk:
            ds_k = [d for d in ds_kd if summary[d]['meta']['n_classes'] == k]
            a_k = [summary[d].get(mn, {}).get('mean', 0) for d in ds_k]
            avg_a.append(np.mean(a_k))
            std_a.append(np.std(a_k) if len(a_k) > 1 else 0)
        mk = 'o' if mn == 'VAE-HNF' else 's'
        ax.plot(uk, avg_a, color=COLORS[mn], linestyle=LS[mn], linewidth=LW[mn],
                marker=mk, markersize=5, label=mn)
        if any(s > 0 for s in std_a):
            ax.fill_between(uk,
                            [a - s for a, s in zip(avg_a, std_a)],
                            [a + s for a, s in zip(avg_a, std_a)],
                            color=COLORS[mn], alpha=0.1)
    ax.set(xlabel='Number of Target Classes (k)', ylabel='Accuracy (%)',
           title='Model Accuracy vs. Target Classes [Optimized]')
    ax.legend(loc='best', title='Model')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig5_v2_acc_vs_k.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  fig5_v2 Acc vs k ✓")


# ====================================================================
# 8. MAIN (协调器)
# ====================================================================
def main():
    parser = argparse.ArgumentParser(
        description='59 - Bayesian Hyperparameter Optimization for VAE-HNF')
    parser.add_argument('--gpus', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Optuna trials per dataset')
    parser.add_argument('--datasets', nargs='+', type=int, default=list(range(11)))
    parser.add_argument('--prev_results', type=str, default=None,
                        help='path to 58_results_*.json for other 5 models')
    parser.add_argument('--skip_optimize', action='store_true',
                        help='skip Phase 1, use existing optuna_d*.json')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    out_dir = Path('/data2/image_identification/src/output')
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    # ---- Worker 模式 (由协调器通过subprocess调用) ----
    if args.worker:
        for did in args.datasets:
            run_worker(args.gpu_id, did, args.n_trials, str(out_dir))
        return

    # =============== ORCHESTRATOR MODE ===============
    MODEL_ORDER = ['RF', 'XGBoost', 'TabPFN', 'HyperTab', 'TPOT', 'VAE-HNF']
    n_gpus = len(args.gpus)

    # ===== PHASE 1: 贝叶斯优化搜索 =====
    if not args.skip_optimize:
        print("=" * 70)
        print("59_bayesian_tuning.py - Phase 1: Optuna贝叶斯超参数优化")
        print("=" * 70)
        print(f"GPUs: {args.gpus}")
        print(f"Datasets: {args.datasets}")
        print(f"Trials/dataset: {args.n_trials}")
        print(f"策略: 每个数据集一个独立subprocess, 多进程共享GPU")
        print("=" * 70)

        # 为每个数据集启动一个独立subprocess (round-robin分配GPU)
        script = os.path.abspath(__file__)
        python = sys.executable

        procs = []
        gpu_assignment = {}
        for i, did in enumerate(args.datasets):
            gpu = args.gpus[i % n_gpus]
            gpu_assignment[did] = gpu
            cmd = [python, '-u', script, '--worker',
                   '--gpu_id', str(gpu),
                   '--n_trials', str(args.n_trials),
                   '--datasets', str(did)]
            logf = open(out_dir / f'59_worker_d{did}.log', 'w')
            p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
            procs.append((did, gpu, p, logf))
            print(f"  d{did} → GPU{gpu} (PID={p.pid})")

        print(f"\n  共 {len(procs)} 个 worker 已启动")
        print(f"  GPU分配: " + ", ".join(
            f"GPU{g}: d{[d for d, g2 in gpu_assignment.items() if g2==g]}"
            for g in args.gpus))

        # 监控
        print("\n监控中 (每30秒更新)...")
        while True:
            all_done = True
            parts = []
            for did, gpu, p, _ in procs:
                rc = p.poll()
                if rc is None:
                    all_done = False
                    if (out_dir / f'optuna_d{did}.json').exists():
                        parts.append(f"d{did}:eval")
                    else:
                        parts.append(f"d{did}:search")
                else:
                    parts.append(f"d{did}:done" if rc == 0 else f"d{did}:FAIL({rc})")
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] {' | '.join(parts)}",
                  flush=True)
            if all_done:
                break
            time.sleep(30)

        for _, _, _, lf in procs:
            lf.close()

        # 检查失败
        for did, gpu, p, _ in procs:
            if p.returncode != 0:
                print(f"  WARNING: d{did} worker failed (rc={p.returncode})")
                log_path = out_dir / f'59_worker_d{did}.log'
                if log_path.exists():
                    with open(log_path) as f:
                        lines = f.readlines()
                    print("  Last 5 lines:")
                    for ln in lines[-5:]:
                        print(f"    {ln.rstrip()}")

    # ===== 收集结果 =====
    print("\n" + "=" * 70)
    print("Phase 2: 收集最优超参数和评估结果")
    print("=" * 70)

    vhnf_results = {}
    best_params_all = {}
    for did in args.datasets:
        f = out_dir / f'optuna_d{did}.json'
        if not f.exists():
            print(f"  d{did}: NOT FOUND (skipped)")
            continue
        with open(f) as fp:
            data = json.load(fp)
        name = data['name']
        best_params_all[str(did)] = data
        vhnf_results[name] = {
            'mean': round(data['final_mean_acc'], 2),
            'std': round(data['final_std_acc'], 2),
            'fold_accs': data['fold_accs'],
            'y_true': data['y_true'],
            'y_pred': data['y_pred'],
            'y_proba': data['y_proba'],
            'best_params': data['best_params'],
        }
        default_acc_line = ""
        top_trials = data.get('top_trials', [])
        # 查默认参数的结果 (trial #0)
        for t in data.get('top_trials', []):
            if t['number'] == 0:
                default_acc_line = f"  (default: {t['value']:.2f}%)"
                break
        print(f"  d{did} ({name}): {data['final_mean_acc']:.2f}±{data['final_std_acc']:.2f}%"
              f"  (optuna: {data['optuna_best_acc']:.2f}%){default_acc_line}"
              f"  [{data['n_trials_completed']} trials, {data['search_time_sec']:.0f}s]")

    # 保存最优参数汇总
    bp_file = out_dir / f'59_best_params_{ts}.json'
    with open(bp_file, 'w') as f:
        json.dump(best_params_all, f, indent=2, ensure_ascii=False)
    print(f"\n  Best params → {bp_file}")

    # ===== PHASE 3: 合并 58 结果 + 生成表格 + 画图 =====
    print("\n" + "=" * 70)
    print("Phase 3: 合并结果 + 生成表格 + 画图")
    print("=" * 70)

    # 加载 58 的其他5模型结果
    prev_file = args.prev_results
    if not prev_file:
        candidates = sorted(glob.glob(str(out_dir / '58_results_*.json')))
        if candidates:
            prev_file = candidates[-1]

    prev_summary = {}
    if prev_file and Path(prev_file).exists():
        with open(prev_file) as f:
            prev = json.load(f)
        prev_summary = prev.get('summary', {})
        print(f"  加载58版结果: {prev_file}")
    else:
        print("  WARNING: 未找到58结果文件")

    # 构建合并后的 summary
    summary = {}
    for name, vr in vhnf_results.items():
        # 获取 meta 信息
        for did in args.datasets:
            f = out_dir / f'optuna_d{did}.json'
            if f.exists():
                with open(f) as fp:
                    d = json.load(fp)
                if d['name'] == name:
                    meta = {'n_samples': d['n_samples'],
                            'n_features': d['n_features'],
                            'n_classes': d['n_classes']}
                    break
        else:
            meta = {}

        summary[name] = {'meta': meta}
        # 其他5模型从58结果取
        for mn in MODEL_ORDER[:-1]:
            if name in prev_summary and mn in prev_summary[name]:
                summary[name][mn] = prev_summary[name][mn]
            else:
                summary[name][mn] = {'mean': 0, 'std': 0, 'fold_accs': [],
                                     'y_true': [], 'y_pred': [], 'y_proba': []}
        # VAE-HNF 用新的优化结果
        summary[name]['VAE-HNF'] = vr

    # 补充 prev 中有但 vhnf 中没有的数据集 (不应该发生)
    for name in prev_summary:
        if name not in summary and name != 'meta':
            summary[name] = prev_summary[name]

    # ---- 打印表格 ----
    print("\n" + "=" * 120)
    print("新版结果: VAE-HNF 贝叶斯优化后 (v2)")
    print("=" * 120)
    hdr = f"{'Dataset':<15}"
    for m in MODEL_ORDER:
        hdr += f"  {m:>12}"
    print(hdr)
    print("-" * 120)

    mavg = {m: [] for m in MODEL_ORDER}
    mwin = {m: 0 for m in MODEL_ORDER}

    for dn in sorted(summary,
                     key=lambda d: summary[d].get('meta', {}).get('n_samples', 0)):
        if dn == 'meta':
            continue
        row = f"{dn:<15}"
        best_a, best_m = -1, None
        for m in MODEL_ORDER:
            d = summary[dn].get(m, {})
            me = d.get('mean', 0)
            st = d.get('std', 0)
            if me > 0:
                row += f"  {me:>5.1f}±{st:<4.1f}"
            else:
                row += f"  {'N/A':>10}"
            mavg[m].append(me)
            if me > best_a:
                best_a = me; best_m = m
        if best_m:
            mwin[best_m] += 1
        print(row)

    print("-" * 120)
    ar = f"{'Average':<15}"
    for m in MODEL_ORDER:
        v = np.mean(mavg[m]) if mavg[m] else 0
        ar += f"  {v:>5.1f}     "
    print(ar)
    wr = f"{'Wins':<15}"
    for m in MODEL_ORDER:
        wr += f"  {mwin[m]:>5d}     "
    print(wr)
    print("=" * 120)

    # ---- 保存合并结果 ----
    result_file = out_dir / f'59_results_{ts}.json'
    save_summary = {}
    for dn, dd in summary.items():
        save_summary[dn] = {}
        for k, v in dd.items():
            if k == 'meta':
                save_summary[dn][k] = v
            else:
                save_summary[dn][k] = {
                    'mean': v.get('mean', 0),
                    'std': v.get('std', 0),
                    'fold_accs': v.get('fold_accs', []),
                    'y_true': v.get('y_true', []),
                    'y_pred': v.get('y_pred', []),
                    'y_proba': v.get('y_proba', []),
                }

    with open(result_file, 'w') as f:
        json.dump({
            'timestamp': ts,
            'models': MODEL_ORDER,
            'note': 'VAE-HNF with Bayesian optimized hyperparameters (Optuna TPE)',
            'n_trials': args.n_trials,
            'best_params_per_dataset': {
                k: v.get('best_params', {}) for k, v in best_params_all.items()
            },
            'summary': save_summary
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  结果: {result_file}")

    # ---- 画图 ----
    print("\n生成 v2 图表...")
    generate_plots_v2(save_summary, str(out_dir))

    # ---- 与旧版对比 ----
    print("\n" + "=" * 70)
    print("v1 (默认参数) vs v2 (贝叶斯优化) 对比")
    print("=" * 70)
    print(f"{'Dataset':<15}  {'v1 (default)':>12}  {'v2 (optimized)':>14}  {'提升':>8}")
    print("-" * 55)
    improvements = []
    for dn in sorted(summary,
                     key=lambda d: summary[d].get('meta', {}).get('n_samples', 0)):
        if dn == 'meta':
            continue
        v2 = summary[dn].get('VAE-HNF', {}).get('mean', 0)
        v1 = prev_summary.get(dn, {}).get('VAE-HNF', {}).get('mean', 0)
        diff = v2 - v1
        improvements.append(diff)
        arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "=")
        print(f"{dn:<15}  {v1:>10.1f}%  {v2:>12.1f}%  {arrow}{abs(diff):>5.1f}pp")
    print("-" * 55)
    avg_imp = np.mean(improvements)
    print(f"{'平均提升':<15}  {'':>10}  {'':>12}  {'+' if avg_imp>0 else ''}{avg_imp:.1f}pp")
    print("=" * 70)

    print("\n全部完成!")
    print(f"  结果文件: {result_file}")
    print(f"  参数文件: {bp_file}")
    print(f"  图表: fig1_v2 ~ fig5_v2 in {out_dir}")


if __name__ == '__main__':
    main()
