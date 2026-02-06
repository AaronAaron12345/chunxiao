#!/usr/bin/env python3
"""
50_multiclass_debug.py
单进程调试版 - 解决多类别标准差高的问题
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'output/50_log.txt', mode='w')
    ]
)
logger = logging.getLogger(__name__)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=8, n_classes=2):
        super().__init__()
        self.n_classes = n_classes
        hidden = max(16, input_dim * 2)
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + n_classes, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.2),
        )
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_var = nn.Linear(hidden, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + n_classes, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, input_dim),
        )
        
    def encode(self, x, y_onehot):
        xy = torch.cat([x, y_onehot], dim=1)
        h = self.encoder(xy)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, y_onehot):
        zy = torch.cat([z, y_onehot], dim=1)
        return self.decoder(zy)
    
    def forward(self, x, y_onehot):
        mu, logvar = self.encode(x, y_onehot)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y_onehot), mu, logvar


class StableHyperNet(nn.Module):
    """稳定版超网络 - 针对多类别优化"""
    def __init__(self, input_dim, n_classes, n_trees=15, tree_depth=4, n_heads=5):
        super().__init__()
        self.n_classes = n_classes
        self.n_trees = n_trees
        self.tree_depth = tree_depth
        self.n_heads = n_heads
        self.n_internal = 2**tree_depth - 1
        self.n_leaves = 2**tree_depth
        
        encoder_dim = 64
        
        # 多头编码器
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, encoder_dim),
                nn.LayerNorm(encoder_dim),
                nn.GELU(),
                nn.Linear(encoder_dim, encoder_dim),
            ) for _ in range(n_heads)
        ])
        
        # 参数生成器
        split_params = n_trees * self.n_internal * (input_dim + 1)
        leaf_params = n_trees * self.n_leaves * n_classes
        tree_params = n_trees
        total = split_params + leaf_params + tree_params
        
        self.generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(encoder_dim, 128),
                nn.GELU(),
                nn.Linear(128, total)
            ) for _ in range(n_heads)
        ])
        
        self.head_weights = nn.Parameter(torch.ones(n_heads) / n_heads)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        self._init()
    
    def _init(self):
        for module_list in [self.encoders, self.generators]:
            for module in module_list:
                for m in module.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.orthogonal_(m.weight, gain=0.5)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
    
    def forward(self, X_support, X_query):
        all_outputs = []
        hw = F.softmax(self.head_weights, dim=0)
        
        for h in range(self.n_heads):
            enc = self.encoders[h](X_support)
            ctx = enc.mean(dim=0, keepdim=True)
            params = self.generators[h](ctx)
            params = torch.tanh(params) * 2.0
            
            # 解析参数
            input_dim = X_support.shape[1]
            idx = 0
            sw_size = self.n_trees * self.n_internal * input_dim
            sb_size = self.n_trees * self.n_internal
            leaf_size = self.n_trees * self.n_leaves * self.n_classes
            
            split_w = params[0, idx:idx+sw_size].view(self.n_trees, self.n_internal, input_dim)
            idx += sw_size
            split_b = params[0, idx:idx+sb_size].view(self.n_trees, self.n_internal)
            idx += sb_size
            leaf_logits = params[0, idx:idx+leaf_size].view(self.n_trees, self.n_leaves, self.n_classes)
            idx += leaf_size
            tree_w = F.softmax(params[0, idx:idx+self.n_trees], dim=0)
            
            temp = torch.clamp(self.temperature, 0.1, 2.0)
            tree_outs = []
            
            for t in range(self.n_trees):
                batch_size = X_query.shape[0]
                reach = torch.ones(batch_size, 1, device=X_query.device)
                
                for d in range(self.tree_depth):
                    start = 2**d - 1
                    n_nodes = 2**d
                    new_reach = []
                    
                    for node in range(n_nodes):
                        node_idx = start + node
                        if node_idx >= self.n_internal:
                            break
                        
                        decision = torch.sigmoid(
                            (X_query @ split_w[t, node_idx] + split_b[t, node_idx]) / temp
                        )
                        
                        parent = reach[:, node:node+1]
                        new_reach.extend([
                            parent * (1 - decision.unsqueeze(1)),
                            parent * decision.unsqueeze(1)
                        ])
                    
                    if new_reach:
                        reach = torch.cat(new_reach, dim=1)
                
                leaf_probs = F.softmax(leaf_logits[t] / temp, dim=-1)
                tree_pred = torch.einsum('bl,lc->bc', reach, leaf_probs)
                tree_outs.append(tree_pred * tree_w[t])
            
            ensemble = torch.stack(tree_outs).sum(dim=0)
            all_outputs.append(ensemble * hw[h])
        
        return torch.stack(all_outputs).sum(dim=0)


def augment_data(X, y, vae, n_per_class, device):
    """类别平衡增强"""
    vae.eval()
    n_classes = len(np.unique(y))
    X_list = [X]
    y_list = [y]
    
    with torch.no_grad():
        for c in range(n_classes):
            X_c = X[y == c]
            n_need = max(0, n_per_class - len(X_c))
            if n_need == 0:
                continue
            
            X_t = torch.FloatTensor(X_c).to(device)
            y_oh = torch.zeros(len(X_c), n_classes, device=device)
            y_oh[:, c] = 1
            
            generated = []
            while len(generated) < n_need:
                mu, logvar = vae.encode(X_t, y_oh)
                z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
                recon = vae.decode(z, y_oh)
                generated.append(recon.cpu().numpy())
            
            generated = np.vstack(generated)[:n_need]
            X_list.append(generated)
            y_list.append(np.full(n_need, c))
    
    return np.vstack(X_list), np.concatenate(y_list)


def train_model(X_train, y_train, X_test, y_test, n_classes, device, seed=42):
    """训练并预测"""
    set_seed(seed)
    
    input_dim = X_train.shape[1]
    
    # 根据类别数调整参数
    n_heads = max(5, n_classes * 2)
    n_trees = max(15, n_classes * 3)
    n_per_class = max(30, 100 // n_classes)
    
    # 训练VAE
    vae = VAE(input_dim, latent_dim=max(4, input_dim), n_classes=n_classes).to(device)
    opt_vae = torch.optim.AdamW(vae.parameters(), lr=0.01)
    
    X_t = torch.FloatTensor(X_train).to(device)
    y_oh = torch.zeros(len(y_train), n_classes, device=device)
    for i, c in enumerate(y_train):
        y_oh[i, c] = 1
    
    vae.train()
    for _ in range(80):
        opt_vae.zero_grad()
        recon, mu, logvar = vae(X_t, y_oh)
        loss = F.mse_loss(recon, X_t) + 0.1 * (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean())
        loss.backward()
        opt_vae.step()
    
    # 数据增强
    X_aug, y_aug = augment_data(X_train, y_train, vae, n_per_class, device)
    
    # 训练HyperNet
    model = StableHyperNet(input_dim, n_classes, n_trees, 4, n_heads).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 150)
    
    X_aug_t = torch.FloatTensor(X_aug).to(device)
    y_aug_t = torch.LongTensor(y_aug).to(device)
    
    model.train()
    for epoch in range(150):
        opt.zero_grad()
        
        idx = np.random.choice(len(X_aug), min(64, len(X_aug)), replace=False)
        out = model(X_aug_t[idx], X_aug_t)
        loss = F.cross_entropy(out, y_aug_t)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    
    # 预测
    model.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test).to(device)
        preds = []
        for _ in range(3):
            idx = np.random.choice(len(X_aug), min(64, len(X_aug)), replace=False)
            out = model(X_aug_t[idx], X_test_t)
            preds.append(F.softmax(out, dim=-1))
        
        avg = torch.stack(preds).mean(dim=0)
        return avg.argmax(dim=1).cpu().numpy()


def load_dataset(dataset_id):
    """加载数据集"""
    data_dir = '/data2/image_identification/src'
    datasets = {
        0: ('data/Data_for_Jinming.csv', 'prostate'),
        1: ('small_data/1.balloons/adult+stretch.data', 'balloons'),
        2: ('small_data/2lens/lenses.data', 'lenses'),
        3: ('small_data/3.caesarian+section+classification+dataset/caesarian.csv', 'caesarian'),
        4: ('small_data/4.iris/iris.data', 'iris'),
        5: ('small_data/5.fertility/fertility_Diagnosis.txt', 'fertility'),
        6: ('small_data/6.zoo/zoo.data', 'zoo'),
        7: ('small_data/7.seeds/seeds_dataset.txt', 'seeds'),
        8: ('small_data/8.haberman+s+survival/haberman.data', 'haberman'),
        9: ('small_data/9.glass+identification/glass.data', 'glass'),
        10: ('small_data/10.yeast/yeast.data', 'yeast'),
    }
    
    if dataset_id not in datasets:
        return None, None, None
    
    filename, name = datasets[dataset_id]
    filepath = f'{data_dir}/{filename}'
    
    try:
        le = LabelEncoder()
        
        if dataset_id == 0:
            df = pd.read_csv(filepath)
            X = df[['LAA', 'Glutamate', 'Choline', 'Sarcosine']].values
            y = (df['Group'] == 'PCa').astype(int).values
        
        elif dataset_id == 1:
            df = pd.read_csv(filepath, header=None)
            for col in df.columns[:-1]:
                df[col] = le.fit_transform(df[col].astype(str))
            X = df.iloc[:, :-1].values.astype(float)
            y = le.fit_transform(df.iloc[:, -1].astype(str))
        
        elif dataset_id == 2:
            df = pd.read_csv(filepath, header=None, delim_whitespace=True)
            X = df.iloc[:, 1:-1].values.astype(float)
            y = (df.iloc[:, -1].values - 1).astype(int)
        
        elif dataset_id == 3:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            data_start = 0
            for i, line in enumerate(lines):
                if line.strip().lower() == '@data':
                    data_start = i + 1
                    break
            data_lines = [l.strip() for l in lines[data_start:] if l.strip()]
            data = [list(map(float, l.split(','))) for l in data_lines if l]
            data = np.array(data)
            X = data[:, :-1]
            y = data[:, -1].astype(int)
        
        elif dataset_id == 4:
            df = pd.read_csv(filepath, header=None)
            X = df.iloc[:, :-1].values.astype(float)
            y = le.fit_transform(df.iloc[:, -1])
        
        elif dataset_id == 5:
            df = pd.read_csv(filepath, header=None)
            X = df.iloc[:, :-1].values.astype(float)
            y = le.fit_transform(df.iloc[:, -1])
        
        elif dataset_id == 6:
            df = pd.read_csv(filepath, header=None)
            X = df.iloc[:, 1:-1].values.astype(float)
            y = (df.iloc[:, -1].values - 1).astype(int)
        
        elif dataset_id == 7:
            df = pd.read_csv(filepath, header=None, delim_whitespace=True)
            X = df.iloc[:, :-1].values.astype(float)
            y = (df.iloc[:, -1].values - 1).astype(int)
        
        elif dataset_id == 8:
            df = pd.read_csv(filepath, header=None)
            X = df.iloc[:, :-1].values.astype(float)
            y = (df.iloc[:, -1].values - 1).astype(int)
        
        elif dataset_id == 9:
            df = pd.read_csv(filepath, header=None)
            X = df.iloc[:, 1:-1].values.astype(float)
            y = le.fit_transform(df.iloc[:, -1])
        
        elif dataset_id == 10:
            df = pd.read_csv(filepath, header=None, delim_whitespace=True)
            X = df.iloc[:, 1:-1].values.astype(float)
            y = le.fit_transform(df.iloc[:, -1])
        
        return X, np.array(y).astype(int), name
    
    except Exception as e:
        logger.error(f"加载 {dataset_id} 失败: {e}")
        return None, None, name


def main():
    Path('output').mkdir(exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("50_multiclass_debug.py - 单进程调试版")
    logger.info("=" * 70)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    seeds = [42, 123, 456, 789, 1000]
    results = {}
    
    for ds_id in range(11):
        X, y, name = load_dataset(ds_id)
        if X is None:
            logger.info(f"{ds_id}. {name}: ERROR loading")
            continue
        
        n_classes = len(np.unique(y))
        logger.info(f"\n{ds_id}. {name}: {len(X)} samples, {X.shape[1]} features, {n_classes} classes")
        
        rf_accs = []
        vhn_accs = []
        
        for seed in seeds:
            set_seed(seed)
            kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            
            rf_preds, rf_labels = [], []
            vhn_preds, vhn_labels = [], []
            
            for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)
                
                # RF
                rf = RandomForestClassifier(n_estimators=100, random_state=seed)
                rf.fit(X_train_s, y_train)
                rf_preds.extend(rf.predict(X_test_s))
                rf_labels.extend(y_test)
                
                # VAE-HyperNet
                try:
                    preds = train_model(X_train_s, y_train, X_test_s, y_test, 
                                        n_classes, device, seed=seed+fold*100)
                    vhn_preds.extend(preds)
                    vhn_labels.extend(y_test)
                except Exception as e:
                    logger.error(f"  Fold {fold} error: {e}")
                    vhn_preds.extend([0] * len(y_test))
                    vhn_labels.extend(y_test)
            
            rf_acc = accuracy_score(rf_labels, rf_preds) * 100
            vhn_acc = accuracy_score(vhn_labels, vhn_preds) * 100
            
            rf_accs.append(rf_acc)
            vhn_accs.append(vhn_acc)
            
            logger.info(f"  Seed {seed}: RF={rf_acc:.1f}%, VHN={vhn_acc:.1f}%")
        
        results[ds_id] = {
            'name': name,
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_classes': n_classes,
            'rf_mean': np.mean(rf_accs),
            'rf_std': np.std(rf_accs),
            'vhn_mean': np.mean(vhn_accs),
            'vhn_std': np.std(vhn_accs)
        }
    
    # 汇总
    logger.info("\n" + "=" * 90)
    logger.info("汇总结果")
    logger.info("=" * 90)
    logger.info(f"{'ID':<4} {'Dataset':<15} {'Samples':<8} {'Classes':<8} {'RF (%)':<15} {'VAE-HyperNet (%)':<18} {'Winner'}")
    logger.info("-" * 90)
    
    for ds_id, r in sorted(results.items()):
        rf_str = f"{r['rf_mean']:.1f}±{r['rf_std']:.1f}"
        vhn_str = f"{r['vhn_mean']:.1f}±{r['vhn_std']:.1f}"
        
        if r['vhn_mean'] > r['rf_mean'] + 0.5:
            winner = "VAE-HyperNet ✅"
        elif r['rf_mean'] > r['vhn_mean'] + 0.5:
            winner = "RF"
        else:
            winner = "Tie"
        
        logger.info(f"{ds_id:<4} {r['name']:<15} {r['n_samples']:<8} {r['n_classes']:<8} {rf_str:<15} {vhn_str:<18} {winner}")


if __name__ == '__main__':
    main()
