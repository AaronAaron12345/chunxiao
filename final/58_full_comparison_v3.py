#!/usr/bin/env python3
"""
58_full_comparison_v3.py  –  6 模型全面对比实验 (稳定顺序版)

完全不使用 multiprocessing，所有任务顺序执行，100% 可靠。
GPU 模型自动轮询使用指定的多块 GPU。

对比模型：
  1. RF (Random Forest)           - sklearn, CPU
  2. XGBoost                      - xgboost, CPU
  3. TabPFN                       - tabpfn, GPU
  4. HyperTab                     - hypertab, GPU
  5. TPOT (AutoML)                - tpot, CPU
  6. VAE-HNF (Our work)           - 本文模型, GPU

11 数据集 (0-10), 5-fold Stratified CV

运行方式（服务器）：
  nohup /data1/condaproject/dinov2/bin/python3 -u 58_full_comparison_v3.py \
        --gpus 1 2 3 4 5 > 58_log.txt 2>&1 &
"""

import os, sys, time, json, random, traceback, argparse, logging, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, roc_curve, auc,
                             precision_recall_curve, average_precision_score)
import xgboost as xgb
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# =====================================================================
# 1. 种子
# =====================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =====================================================================
# 2. VAE-HNF 模型  (来自 47_stable_37.py)
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
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None: nn.init.zeros_(m.bias)
    def encode(self, x):
        h = self.encoder(x); return self.fc_mu(h), self.fc_var(h)
    def reparameterize(self, mu, logvar, ns=1.0):
        std = torch.exp(0.5*logvar); return mu + torch.randn_like(std)*std*ns
    def forward(self, x, ns=1.0):
        mu, lv = self.encode(x); z = self.reparameterize(mu, lv, ns)
        return self.decoder(z), mu, lv

class StableHyperNetworkForTree(nn.Module):
    def __init__(self, input_dim, n_classes, n_trees=15, tree_depth=3,
                 hidden_dim=64, n_heads=5):
        super().__init__()
        self.input_dim, self.n_trees = input_dim, n_trees
        self.tree_depth, self.n_heads, self.n_classes = tree_depth, n_heads, n_classes
        n_int  = 2**tree_depth - 1
        n_leaf = 2**tree_depth
        self.n_internal, self.n_leaves = n_int, n_leaf
        self.params_per_tree = n_int*input_dim + n_int + n_leaf*n_classes
        self.total_params = self.params_per_tree*n_trees + n_trees
        self.data_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.hyper_heads = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2), nn.LayerNorm(hidden_dim*2),
            nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(hidden_dim*2, hidden_dim*2), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim*2, self.total_params)) for _ in range(n_heads)])
        self.head_weights = nn.Parameter(torch.ones(n_heads)/n_heads)
        self.output_scale = nn.Parameter(torch.tensor(1.0))
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, X_train):
        enc = self.data_encoder(X_train)
        ctx = enc.mean(dim=0, keepdim=True)
        all_p = torch.stack([h(ctx.squeeze(0)) for h in self.hyper_heads], dim=0)
        w = F.softmax(self.head_weights, dim=0)
        return torch.tanh(torch.einsum('h,hp->p', w, all_p)) * self.output_scale
    def parse_params(self, params):
        trees, off = [], 0
        for _ in range(self.n_trees):
            sw_sz = self.n_internal*self.input_dim
            sw = params[off:off+sw_sz].view(self.n_internal, self.input_dim); off += sw_sz
            sb = params[off:off+self.n_internal]; off += self.n_internal
            ll_sz = self.n_leaves*self.n_classes
            ll = params[off:off+ll_sz].view(self.n_leaves, self.n_classes); off += ll_sz
            trees.append({'split_weights': sw, 'split_bias': sb, 'leaf_logits': ll})
        tw = params[off:off+self.n_trees]
        return trees, tw

class StableTreeClassifier(nn.Module):
    def __init__(self, tree_depth=3, init_temp=1.0):
        super().__init__()
        self.depth = tree_depth
        self.n_internal = 2**tree_depth - 1
        self.n_leaves  = 2**tree_depth
        self.log_temperature = nn.Parameter(torch.tensor(np.log(init_temp)))
    @property
    def temperature(self):
        return torch.exp(self.log_temperature).clamp(0.1, 5.0)
    def forward_single_tree(self, x, sw, sb, ll):
        B = x.size(0)
        sp = torch.sigmoid((x @ sw.T + sb) / self.temperature)
        lp = torch.ones(B, self.n_leaves, device=x.device)
        for li in range(self.n_leaves):
            pp = torch.ones(B, device=x.device); ni = li + self.n_internal
            for _ in range(self.depth):
                pi = (ni-1)//2
                pp = pp * (sp[:, pi] if ni%2==0 else 1-sp[:, pi])
                ni = pi
            lp[:, li] = pp
        return torch.einsum('bl,lc->bc', lp, F.softmax(ll/self.temperature, dim=-1))
    def forward(self, x, trees_params, tree_weights):
        outs = torch.stack([self.forward_single_tree(
            x, tp['split_weights'], tp['split_bias'], tp['leaf_logits'])
            for tp in trees_params], dim=0)
        return torch.einsum('t,tbc->bc', F.softmax(tree_weights, dim=0), outs)

class StableVAEHyperNetFusion(nn.Module):
    def __init__(self, input_dim, n_classes, n_trees=15, tree_depth=3, n_heads=5):
        super().__init__()
        self.vae = VAE(input_dim)
        self.hypernet = StableHyperNetworkForTree(
            input_dim, n_classes, n_trees, tree_depth, n_heads=n_heads)
        self.classifier = StableTreeClassifier(tree_depth)
    def generate_augmented_data_deterministic(self, X, y, n_aug=200, ns=0.3):
        self.vae.eval(); aX, aY = [X], [y]
        with torch.no_grad():
            for i in range(n_aug):
                idx = i % X.size(0); mu, lv = self.vae.encode(X[idx:idx+1])
                z = self.vae.reparameterize(mu, lv, ns=ns)
                aX.append(self.vae.decoder(z)); aY.append(y[idx:idx+1])
        return torch.cat(aX, 0), torch.cat(aY, 0)
    def forward(self, X_train, X_test):
        p = self.hypernet(X_train); tp, tw = self.hypernet.parse_params(p)
        return self.classifier(X_test, tp, tw), p

def train_vae_hnf(X_train, y_train, X_val, y_val, n_classes, device,
                  epochs=300, seed=42):
    set_seed(seed)
    dim = X_train.shape[1]
    model = StableVAEHyperNetFusion(dim, n_classes, 15, 3, 5).to(device)
    Xt = torch.FloatTensor(X_train).to(device)
    yt = torch.LongTensor(y_train).to(device)
    Xv = torch.FloatTensor(X_val).to(device)
    # 阶段 1 — VAE
    vopt = torch.optim.Adam(model.vae.parameters(), lr=0.002, weight_decay=1e-5)
    model.vae.train()
    for ep in range(100):
        vopt.zero_grad()
        rec, mu, lv = model.vae(Xt)
        loss = F.mse_loss(rec, Xt) + 0.01*(-0.5*torch.mean(1+lv-mu**2-lv.exp()))
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.vae.parameters(), 1.0)
        vopt.step()
    # 增强
    model.vae.eval()
    with torch.no_grad():
        Xa, ya = model.generate_augmented_data_deterministic(Xt, yt, 200, 0.3)
    # 阶段 2 — HyperNet + Tree
    prms = list(model.hypernet.parameters()) + list(model.classifier.parameters())
    opt  = torch.optim.AdamW(prms, lr=0.01, weight_decay=0.05)
    wu = 20
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e:
        (e+1)/wu if e < wu else 0.5*(1+np.cos(np.pi*(e-wu)/(epochs-wu))))
    best_acc, best_st, noimpr = 0, None, 0
    for ep in range(epochs):
        model.hypernet.train(); model.classifier.train(); opt.zero_grad()
        out, p = model(Xa, Xa)
        loss = F.cross_entropy(out, ya) + 0.01*(p**2).mean()
        if ep > wu:
            model.eval()
            with torch.no_grad(): o2, _ = model(Xa, Xa)
            model.hypernet.train(); model.classifier.train()
            loss = loss + 0.1*F.mse_loss(out, o2.detach())
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step(); sched.step()
        if (ep+1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                vo, _ = model(Xt, Xv)
                va = accuracy_score(y_val, vo.argmax(1).cpu().numpy())
            if va > best_acc:
                best_acc = va; best_st = {k: v.clone() for k, v in model.state_dict().items()}; noimpr = 0
            else:
                noimpr += 1
            if noimpr >= 5: break
    if best_st: model.load_state_dict(best_st)
    return model

def predict_vae_hnf(model, X_train, X_test, device):
    model.eval()
    with torch.no_grad():
        out, _ = model(torch.FloatTensor(X_train).to(device),
                       torch.FloatTensor(X_test).to(device))
        return F.softmax(out, dim=1).cpu().numpy()

# =====================================================================
# 3. 数据集加载
# =====================================================================
def load_dataset_by_id(dataset_id,
                       data_dir="/data2/image_identification/src/small_data"):
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
    if dataset_id not in datasets:
        return None, None, None
    name, frel = datasets[dataset_id]
    try:
        if dataset_id == 0:
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
            if not Path(fp).exists(): continue
            if dataset_id == 1:
                df = pd.read_csv(fp, header=None); X = df.iloc[:,:-1].values; y = df.iloc[:,-1].values
            elif dataset_id == 2:
                df = pd.read_csv(fp, sep=r'\s+', header=None, engine='python')
                X = df.iloc[:,1:-1].values; y = df.iloc[:,-1].values
            elif dataset_id == 3:
                rows = []
                with open(fp) as f:
                    started = False
                    for ln in f:
                        if '@data' in ln.lower(): started = True; continue
                        if started and ln.strip() and not ln.startswith('%'):
                            r = ln.strip().split(',');
                            if len(r)>=2: rows.append(r)
                df = pd.DataFrame(rows); X = df.iloc[:,:-1].values.astype(float); y = df.iloc[:,-1].values
            elif dataset_id == 4:
                df = pd.read_csv(fp, header=None).dropna(how='all')
                df = df[df.iloc[:,-1].astype(str).str.strip() != '']
                X = df.iloc[:,:-1].values; y = df.iloc[:,-1].values
            elif dataset_id in (5, 8):
                df = pd.read_csv(fp, header=None); X = df.iloc[:,:-1].values; y = df.iloc[:,-1].values
            elif dataset_id == 6:
                df = pd.read_csv(fp, header=None); X = df.iloc[:,1:-1].values; y = df.iloc[:,-1].values
            elif dataset_id in (7,):
                df = pd.read_csv(fp, sep=r'\s+', header=None, engine='python')
                X = df.iloc[:,:-1].values; y = df.iloc[:,-1].values
            elif dataset_id == 9:
                df = pd.read_csv(fp, header=None); X = df.iloc[:,1:-1].values; y = df.iloc[:,-1].values
            elif dataset_id == 10:
                df = pd.read_csv(fp, sep=r'\s+', header=None, engine='python')
                X = df.iloc[:,1:-1].values; y = df.iloc[:,-1].values
            else:
                continue
            for c in range(X.shape[1]):
                try: X[:,c] = X[:,c].astype(float)
                except: X[:,c] = LabelEncoder().fit_transform(X[:,c].astype(str))
            X = X.astype(float); y = LabelEncoder().fit_transform(y.astype(str))
            return X, y, name
        return None, None, name
    except Exception as e:
        print(f"  加载数据集 {name} 出错: {e}"); traceback.print_exc()
        return None, None, name

# =====================================================================
# 4. 概率矩阵修正
# =====================================================================
def ensure_proba_columns(proba, n_classes, classes=None):
    if proba is None: return None
    if proba.ndim == 1: proba = np.column_stack([1-proba, proba])
    if proba.shape[1] < n_classes:
        fp = np.zeros((proba.shape[0], n_classes))
        if classes is not None:
            for i, c in enumerate(classes):
                if c < n_classes: fp[:, c] = proba[:, i]
        else:
            fp[:, :proba.shape[1]] = proba
        proba = fp
    return proba

# =====================================================================
# 5. 单模型单折评估函数 (全在主进程顺序调用)
# =====================================================================
def evaluate_fold(model_name, X_train_s, y_train, X_test_s, y_test,
                  n_classes, gpu_id, seed, fold_idx):
    """
    对单个 (model, fold) 进行评估，返回 (preds, proba, acc)。
    全部在主进程执行，无子进程/daemon 问题。
    """
    device_str = f'cuda:{gpu_id}' if torch.cuda.is_available() and gpu_id >= 0 else 'cpu'

    if model_name == 'RF':
        clf = RandomForestClassifier(n_estimators=100, random_state=seed)
        clf.fit(X_train_s, y_train)
        preds = clf.predict(X_test_s)
        proba = ensure_proba_columns(clf.predict_proba(X_test_s), n_classes, clf.classes_)

    elif model_name == 'XGBoost':
        clf = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            use_label_encoder=False,
            eval_metric='mlogloss' if n_classes > 2 else 'logloss',
            random_state=seed, verbosity=0)
        clf.fit(X_train_s, y_train)
        preds = clf.predict(X_test_s)
        proba = ensure_proba_columns(clf.predict_proba(X_test_s), n_classes, clf.classes_)

    elif model_name == 'TabPFN':
        from tabpfn import TabPFNClassifier
        clf = TabPFNClassifier(device=device_str, N_ensemble_configurations=16)
        Xtr = X_train_s[:1000]; ytr = y_train[:1000]
        Xts = X_test_s
        if Xtr.shape[1] > 100: Xtr = Xtr[:,:100]; Xts = Xts[:,:100]
        clf.fit(Xtr, ytr)
        preds = clf.predict(Xts)
        proba = ensure_proba_columns(clf.predict_proba(Xts), n_classes)

    elif model_name == 'HyperTab':
        from hypertab import HyperTabClassifier
        clf = HyperTabClassifier(device=device_str, test_nodes=100,
                                  epochs=50, hidden_dims=64)
        clf.fit(X_train_s, y_train)
        preds = clf.predict(X_test_s)
        proba = ensure_proba_columns(np.array(clf.predict_proba(X_test_s)), n_classes)

    elif model_name == 'TPOT':
        from tpot import TPOTClassifier
        clf = TPOTClassifier(cv=3, random_state=seed, verbose=0,
                             max_time_mins=1, max_eval_time_mins=0.5,
                             n_jobs=1)
        clf.fit(X_train_s, y_train)
        preds = clf.predict(X_test_s)
        try:
            proba = ensure_proba_columns(clf.predict_proba(X_test_s), n_classes)
        except Exception:
            proba = np.zeros((len(X_test_s), n_classes))
            for i, p in enumerate(preds): proba[i, p] = 1.0

    elif model_name == 'VAE-HNF':
        device = torch.device(device_str)
        mdl = train_vae_hnf(X_train_s, y_train, X_test_s, y_test,
                            n_classes, device, epochs=300, seed=seed+fold_idx)
        proba = predict_vae_hnf(mdl, X_train_s, X_test_s, device)
        preds = proba.argmax(axis=1)
        del mdl; torch.cuda.empty_cache()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    acc = accuracy_score(y_test, preds) * 100
    return preds, proba, acc


# =====================================================================
# 6. 主运行逻辑 (全顺序)
# =====================================================================
def run_comparison(gpu_ids, dataset_ids, model_names, seed=42):
    all_results = []
    dataset_meta = {}

    # —— 加载全部数据集 ——
    datasets = {}
    for did in dataset_ids:
        X, y, name = load_dataset_by_id(did)
        if X is None:
            print(f"  跳过 d{did}: {name} (加载失败)"); continue
        datasets[did] = (X, y, name)
        nc = len(np.unique(y))
        dataset_meta[str(did)] = dict(name=name, n_samples=int(len(X)),
                                       n_features=int(X.shape[1]),
                                       n_classes=int(nc))
        print(f"  d{did}: {name}  n={len(X)}  d={X.shape[1]}  k={nc}")

    n_gpus = len(gpu_ids) if gpu_ids else 1
    gpu_counter = 0  # 轮询计数器

    total_tasks = len(model_names) * sum(5 for _ in datasets)
    done = 0
    t0 = time.time()

    # —— 逐模型 → 逐数据集 → 逐折 ——
    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"  模型: {model_name}")
        print(f"{'='*60}")
        model_t0 = time.time()

        for did, (X, y, name) in datasets.items():
            nc = len(np.unique(y))
            kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            for fi, (tri, tei) in enumerate(kfold.split(X, y)):
                # GPU 轮询
                gpu_id = gpu_ids[gpu_counter % n_gpus] if gpu_ids else -1
                gpu_counter += 1

                set_seed(seed + fi)
                X_tr, X_te = X[tri], X[tei]
                y_tr, y_te = y[tri], y[tei]
                sc = StandardScaler()
                X_tr_s = sc.fit_transform(X_tr)
                X_te_s = sc.transform(X_te)

                fold_t0 = time.time()
                try:
                    preds, proba, acc = evaluate_fold(
                        model_name, X_tr_s, y_tr, X_te_s, y_te,
                        nc, gpu_id, seed, fi)
                    status = 'ok'
                except Exception as e:
                    print(f"    ✗ {model_name}|d{did}|f{fi}: {e}")
                    traceback.print_exc()
                    preds = np.zeros(len(y_te), dtype=int)
                    proba = None
                    acc = 0.0
                    status = f'error: {str(e)}'

                elapsed_fold = time.time() - fold_t0
                done += 1
                elapsed_total = time.time() - t0

                result = dict(
                    model=model_name, dataset_id=did, fold=fi,
                    acc=acc, status=status,
                    y_true=y_te.tolist(),
                    y_pred=preds.tolist(),
                    y_proba=proba.tolist() if proba is not None else None)
                all_results.append(result)

                gpu_str = f"GPU{gpu_id}" if gpu_id >= 0 else "CPU"
                print(f"    [{done:>3d}/{total_tasks}] "
                      f"{model_name:>8s}|{name:<12s}|f{fi} "
                      f"=> {acc:6.2f}%  ({elapsed_fold:.1f}s)  "
                      f"[{gpu_str}]  "
                      f"total {elapsed_total:.0f}s", flush=True)

        model_elapsed = time.time() - model_t0
        print(f"  ── {model_name} 完成, 耗时 {model_elapsed:.1f}s "
              f"({model_elapsed/60:.1f}min)")

    total_elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"全部完成! 总耗时: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    print(f"{'='*60}")

    return all_results, dataset_meta


# =====================================================================
# 7. 结果聚合
# =====================================================================
MODEL_ORDER = ['RF', 'XGBoost', 'TabPFN', 'HyperTab', 'TPOT', 'VAE-HNF']

def aggregate_results(results_list, dataset_meta):
    grouped = {}
    for r in results_list:
        if r['status'] != 'ok': continue
        key = (r['model'], r['dataset_id'])
        grouped.setdefault(key, dict(accs=[], y_true_all=[], y_pred_all=[], y_proba_all=[]))
        grouped[key]['accs'].append(r['acc'])
        grouped[key]['y_true_all'].extend(r['y_true'])
        grouped[key]['y_pred_all'].extend(r['y_pred'])
        if r.get('y_proba'): grouped[key]['y_proba_all'].extend(r['y_proba'])

    summary = {}
    for did_str, meta in dataset_meta.items():
        did = int(did_str); dn = meta['name']; summary[dn] = {'meta': meta}
        for mn in MODEL_ORDER:
            key = (mn, did)
            if key in grouped:
                g = grouped[key]
                summary[dn][mn] = dict(
                    mean=round(np.mean(g['accs']), 2),
                    std=round(np.std(g['accs']), 2),
                    fold_accs=[round(a,2) for a in g['accs']],
                    y_true=g['y_true_all'], y_pred=g['y_pred_all'],
                    y_proba=g['y_proba_all'])
            else:
                summary[dn][mn] = dict(mean=0, std=0, fold_accs=[],
                                        y_true=[], y_pred=[], y_proba=[])
    return summary


# =====================================================================
# 8. 绘图
# =====================================================================
def generate_plots(summary, output_dir):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':12, 'axes.titlesize':14,
                         'axes.labelsize':12, 'legend.fontsize':9,
                         'figure.dpi':150})

    COLORS = {'RF':'#FFD700','XGBoost':'#7CFC00','TabPFN':'#00CED1',
              'HyperTab':'#4169E1','TPOT':'#9370DB','VAE-HNF':'#FF0000'}
    LS   = {'RF':'--','XGBoost':'--','TabPFN':'--',
            'HyperTab':'-.','TPOT':'-.','VAE-HNF':'-'}
    LW   = {'RF':1.5,'XGBoost':1.5,'TabPFN':1.5,
            'HyperTab':1.5,'TPOT':1.5,'VAE-HNF':3.0}

    available = [m for m in MODEL_ORDER if any(
        summary.get(d,{}).get(m,{}).get('mean',0)>0
        for d in summary if d != 'meta')]
    print(f"  可用模型: {available}")

    # —— 图 1: ROC ——
    binary_ds = [d for d in summary if summary[d].get('meta',{}).get('n_classes')==2]
    roc_ds = None
    for pref in ['haberman','caesarian','prostate','fertility','balloons']:
        if pref in binary_ds: roc_ds = pref; break
    if roc_ds is None and binary_ds: roc_ds = binary_ds[0]

    if roc_ds:
        fig, ax = plt.subplots(figsize=(8,6))
        for mn in available:
            d = summary[roc_ds].get(mn, {})
            yt, yp = d.get('y_true',[]), d.get('y_proba',[])
            if not yt or not yp: continue
            ya = np.array(yt); pa = np.array(yp)
            ys = pa[:,1] if pa.ndim==2 and pa.shape[1]>=2 else pa.ravel()
            fpr, tpr, _ = roc_curve(ya, ys); ra = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=COLORS[mn], linestyle=LS[mn],
                    linewidth=LW[mn], label=f'{mn} (AUC={ra:.2f})')
        ax.plot([0,1],[0,1],'k--',alpha=.5,label='Random Guess')
        ax.set(xlabel='FPR', ylabel='TPR', title='ROC Curves', xlim=[0,1], ylim=[0,1.05])
        ax.legend(loc='lower right'); fig.tight_layout()
        fig.savefig(os.path.join(output_dir,'fig1_roc_curves.png'),dpi=150,bbox_inches='tight')
        plt.close(fig); print(f"  图1 ROC ({roc_ds}) ✓")

    # —— 图 2: PR ——
    if roc_ds:
        fig, ax = plt.subplots(figsize=(8,6))
        for mn in available:
            d = summary[roc_ds].get(mn, {})
            yt, yp = d.get('y_true',[]), d.get('y_proba',[])
            if not yt or not yp: continue
            ya = np.array(yt); pa = np.array(yp)
            ys = pa[:,1] if pa.ndim==2 and pa.shape[1]>=2 else pa.ravel()
            pr, re, _ = precision_recall_curve(ya, ys)
            ap = average_precision_score(ya, ys)
            ax.plot(re, pr, color=COLORS[mn], linestyle=LS[mn],
                    linewidth=LW[mn], label=f'{mn} (AP={ap:.2f})')
        ax.set(xlabel='Recall', ylabel='Precision',
               title='Precision-Recall Curves')
        ax.legend(loc='best'); fig.tight_layout()
        fig.savefig(os.path.join(output_dir,'fig2_pr_curves.png'),dpi=150,bbox_inches='tight')
        plt.close(fig); print(f"  图2 PR  ({roc_ds}) ✓")

    # —— 图 3: Acc vs n ——
    fig, ax = plt.subplots(figsize=(8,6))
    ds_sorted = sorted([d for d in summary], key=lambda d: summary[d].get('meta',{}).get('n_samples',0))
    ns = [summary[d]['meta']['n_samples'] for d in ds_sorted]
    for mn in available:
        accs = [summary[d].get(mn,{}).get('mean',0) for d in ds_sorted]
        mk = 'o' if mn=='VAE-HNF' else 's'
        ax.plot(ns, accs, color=COLORS[mn], linestyle=LS[mn], linewidth=LW[mn],
                marker=mk, markersize=5, label=mn)
    ax.set(xlabel='Number of Records (n)', ylabel='Accuracy (%)',
           title='Model Accuracy vs. Dataset Size (n)')
    ax.set_xscale('log'); ax.legend(loc='best',title='Model'); fig.tight_layout()
    fig.savefig(os.path.join(output_dir,'fig3_acc_vs_n.png'),dpi=150,bbox_inches='tight')
    plt.close(fig); print("  图3 Acc vs n ✓")

    # —— 图 4: Acc vs d ——
    fig, ax = plt.subplots(figsize=(8,6))
    ds_fd = sorted([d for d in summary], key=lambda d: summary[d].get('meta',{}).get('n_features',0))
    ds_vals = [summary[d]['meta']['n_features'] for d in ds_fd]
    for mn in available:
        accs = [summary[d].get(mn,{}).get('mean',0) for d in ds_fd]
        mk = 'o' if mn=='VAE-HNF' else 's'
        ax.plot(ds_vals, accs, color=COLORS[mn], linestyle=LS[mn], linewidth=LW[mn],
                marker=mk, markersize=5, label=mn)
    ax.set(xlabel='Number of Features (d)', ylabel='Accuracy (%)',
           title='Model Accuracy vs. Number of Features (d)')
    ax.legend(loc='best',title='Model'); fig.tight_layout()
    fig.savefig(os.path.join(output_dir,'fig4_acc_vs_d.png'),dpi=150,bbox_inches='tight')
    plt.close(fig); print("  图4 Acc vs d ✓")

    # —— 图 5: Acc vs k ——
    fig, ax = plt.subplots(figsize=(8,6))
    ds_kd = sorted([d for d in summary], key=lambda d: summary[d].get('meta',{}).get('n_classes',0))
    kvs = [summary[d]['meta']['n_classes'] for d in ds_kd]
    uk = sorted(set(kvs))
    for mn in available:
        avg_a, std_a = [], []
        for k in uk:
            ds_k = [d for d in ds_kd if summary[d]['meta']['n_classes']==k]
            a_k = [summary[d].get(mn,{}).get('mean',0) for d in ds_k]
            avg_a.append(np.mean(a_k)); std_a.append(np.std(a_k) if len(a_k)>1 else 0)
        mk = 'o' if mn=='VAE-HNF' else 's'
        ax.plot(uk, avg_a, color=COLORS[mn], linestyle=LS[mn], linewidth=LW[mn],
                marker=mk, markersize=5, label=mn)
        if any(s>0 for s in std_a):
            ax.fill_between(uk, [a-s for a,s in zip(avg_a,std_a)],
                            [a+s for a,s in zip(avg_a,std_a)],
                            color=COLORS[mn], alpha=0.1)
    ax.set(xlabel='Number of Target Classes (k)', ylabel='Accuracy (%)',
           title='Model Accuracy vs. Number of Target Classes (k)')
    ax.legend(loc='best',title='Model'); fig.tight_layout()
    fig.savefig(os.path.join(output_dir,'fig5_acc_vs_k.png'),dpi=150,bbox_inches='tight')
    plt.close(fig); print("  图5 Acc vs k ✓")


# =====================================================================
# 9. main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description='6-Model Full Comparison v3 (sequential)')
    parser.add_argument('--gpus', nargs='+', type=int, default=[1,2,3,4,5])
    parser.add_argument('--datasets', nargs='+', type=int, default=list(range(11)))
    parser.add_argument('--models', nargs='+', type=str,
                        default=['RF','XGBoost','TabPFN','HyperTab','TPOT','VAE-HNF'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--results', type=str, default=None)
    parser.add_argument('--no-gpu', action='store_true')
    args = parser.parse_args()

    out = Path('/data2/image_identification/src/output')
    if not out.exists(): out = Path('output')
    out.mkdir(exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    if args.plot_only:
        rf = args.results
        if not rf:
            fs = sorted(glob.glob(str(out/'58_results_*.json')) +
                        glob.glob(str(out/'57_results_*.json')))
            if not fs: print("找不到结果文件!"); return
            rf = fs[-1]
        print(f"加载: {rf}")
        with open(rf) as f: data = json.load(f)
        generate_plots(data['summary'], str(out))
        print("图表完成!"); return

    print("="*70)
    print("58_full_comparison_v3.py  –  6 模型全面对比 (顺序稳定版)")
    print("="*70)
    print(f"模型: {args.models}")
    print(f"数据集: {args.datasets}")
    print(f"GPU: {args.gpus}")
    print(f"Seed: {args.seed}")
    print("="*70)

    gpus = [] if args.no_gpu else args.gpus
    results_list, dataset_meta = run_comparison(
        gpus, args.datasets, args.models, args.seed)

    summary = aggregate_results(results_list, dataset_meta)

    # 打印表格
    print("\n" + "="*110)
    print("结果汇总表 (Mean±Std %)")
    print("="*110)
    hdr = f"{'Dataset':<15}"
    for m in args.models: hdr += f" {m:>15}"
    print(hdr); print("-"*110)

    mavg = {m: [] for m in args.models}
    mwin = {m: 0 for m in args.models}
    for dn in sorted(summary, key=lambda d: summary[d]['meta']['n_samples']):
        row = f"{dn:<15}"
        best_a, best_m = -1, None
        for m in args.models:
            d = summary[dn].get(m, {}); me = d.get('mean',0); st = d.get('std',0)
            row += f" {me:>6.2f}±{st:<5.2f}"; mavg[m].append(me)
            if me > best_a: best_a = me; best_m = m
        if best_m: mwin[best_m] += 1
        print(row)
    print("-"*110)
    ar = f"{'Average':<15}"
    for m in args.models: ar += f" {np.mean(mavg[m]) if mavg[m] else 0:>6.2f}      "
    print(ar)
    wr = f"{'Wins':<15}"
    for m in args.models: wr += f" {mwin[m]:>6d}      "
    print(wr); print("="*110)

    # 保存
    save_s = {}
    for dn, dd in summary.items():
        save_s[dn] = {}
        for k, v in dd.items():
            save_s[dn][k] = v if k == 'meta' else dict(
                mean=v.get('mean',0), std=v.get('std',0),
                fold_accs=v.get('fold_accs',[]),
                y_true=v.get('y_true',[]), y_pred=v.get('y_pred',[]),
                y_proba=v.get('y_proba',[]))
    rf = out / f'58_results_{ts}.json'
    with open(rf, 'w') as f:
        json.dump(dict(timestamp=ts, models=args.models, datasets=args.datasets,
                       dataset_meta=dataset_meta, summary=save_s),
                  f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {rf}")

    print("\n生成图表...")
    generate_plots(save_s, str(out))
    print("\n全部完成!")


if __name__ == '__main__':
    main()
