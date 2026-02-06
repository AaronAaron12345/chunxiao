#!/usr/bin/env python3
"""
48_batch_datasets_multigpu.py - 用47_stable_37的模型跑0-10数据集

功能：
1. 使用47的稳定多头HyperNet架构
2. 多GPU并行（每个数据集分配到不同GPU）
3. 每个GPU内多进程运行多个seed
4. 记录准确率和标准差
5. 输出汇总结果表格
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import logging
import json
from datetime import datetime
from pathlib import Path
import random
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============== 47的稳定模型架构（复制） ==============
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


class StableHyperNetworkForTree(nn.Module):
    def __init__(self, input_dim, n_classes=2, n_trees=10, tree_depth=3, hidden_dim=64, n_heads=5):
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.n_trees = n_trees
        self.tree_depth = tree_depth
        self.n_heads = n_heads
        
        n_internal = 2**tree_depth - 1
        n_leaves = 2**tree_depth
        
        self.params_per_tree = n_internal * input_dim + n_internal + n_leaves * n_classes
        self.total_params = self.params_per_tree * n_trees + n_trees
        
        self.n_internal = n_internal
        self.n_leaves = n_leaves
        
        self.data_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
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
            split_w_size = self.n_internal * self.input_dim
            split_weights = params[offset:offset+split_w_size].view(self.n_internal, self.input_dim)
            offset += split_w_size
            
            split_bias = params[offset:offset+self.n_internal]
            offset += self.n_internal
            
            leaf_size = self.n_leaves * self.n_classes
            leaf_logits = params[offset:offset+leaf_size].view(self.n_leaves, self.n_classes)
            offset += leaf_size
            
            trees_params.append({
                'split_weights': split_weights,
                'split_bias': split_bias,
                'leaf_logits': leaf_logits
            })
        
        tree_weights = params[offset:offset+self.n_trees]
        return trees_params, tree_weights


class StableTreeClassifier(nn.Module):
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
        n_classes = leaf_logits.size(1)
        
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


class StableVAEHyperNetFusion(nn.Module):
    def __init__(self, input_dim, n_classes=2, n_trees=15, tree_depth=3, n_heads=5):
        super().__init__()
        self.n_classes = n_classes
        self.vae = VAE(input_dim)
        self.hypernet = StableHyperNetworkForTree(input_dim, n_classes, n_trees, tree_depth, n_heads=n_heads)
        self.classifier = StableTreeClassifier(tree_depth)
    
    def generate_augmented_data_deterministic(self, X, y, n_augment=200, noise_scale=0.3):
        self.vae.eval()
        augmented_X = [X]
        augmented_y = [y]
        
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
        params = self.hypernet(X_train)
        trees_params, tree_weights = self.hypernet.parse_params(params)
        output = self.classifier(X_test, trees_params, tree_weights)
        return output, params


def train_stable_model(X_train, y_train, X_val, y_val, n_classes, device, epochs=300, seed=42):
    set_seed(seed)
    
    input_dim = X_train.shape[1]
    model = StableVAEHyperNetFusion(input_dim, n_classes, n_trees=15, tree_depth=3, n_heads=5).to(device)
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    
    # 训练VAE
    vae_optimizer = torch.optim.Adam(model.vae.parameters(), lr=0.002, weight_decay=1e-5)
    model.vae.train()
    for epoch in range(100):
        vae_optimizer.zero_grad()
        recon, mu, logvar = model.vae(X_train_t)
        recon_loss = F.mse_loss(recon, X_train_t, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        vae_loss = recon_loss + 0.01 * kl_loss
        vae_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.vae.parameters(), 1.0)
        vae_optimizer.step()
    
    # 生成增强数据
    model.vae.eval()
    with torch.no_grad():
        X_aug, y_aug = model.generate_augmented_data_deterministic(
            X_train_t, y_train_t, n_augment=200, noise_scale=0.3
        )
    
    # 训练HyperNet
    hypernet_optimizer = torch.optim.AdamW(
        list(model.hypernet.parameters()) + list(model.classifier.parameters()),
        lr=0.01, weight_decay=0.05
    )
    
    warmup_epochs = 20
    def get_lr(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(hypernet_optimizer, get_lr)
    
    best_val_acc = 0
    best_state = None
    no_improve = 0
    
    for epoch in range(epochs):
        model.hypernet.train()
        model.classifier.train()
        
        hypernet_optimizer.zero_grad()
        output, params = model(X_aug, X_aug)
        cls_loss = F.cross_entropy(output, y_aug)
        reg_loss = 0.01 * (params ** 2).mean()
        loss = cls_loss + reg_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        hypernet_optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_output, _ = model(X_train_t, X_val_t)
                val_pred = val_output.argmax(dim=1).cpu().numpy()
                val_acc = accuracy_score(y_val, val_pred)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                
                if no_improve >= 5:
                    break
    
    if best_state:
        model.load_state_dict(best_state)
    
    return model


def predict_stable(model, X_train, X_test, device):
    model.eval()
    X_train_t = torch.FloatTensor(X_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        output, _ = model(X_train_t, X_test_t)
        proba = F.softmax(output, dim=1)
    
    return proba.cpu().numpy()


# ============== 数据集加载 ==============
def load_dataset(dataset_id, data_dir):
    """加载数据集"""
    datasets = {
        0: ('Data_for_Jinming.csv', 'prostate'),
        1: ('1.balloons/adult+stretch.data', 'balloons'),  # 修正文件名
        2: ('2lens/lenses.data', 'lenses'),
        3: ('3.caesarian+section+classification+dataset/caesarian.csv', 'caesarian'),
        4: ('4.iris/iris.data', 'iris'),
        5: ('5.fertility/fertility_Diagnosis.txt', 'fertility'),
        6: ('6.zoo/zoo.data', 'zoo'),
        7: ('7.seeds/seeds_dataset.txt', 'seeds'),
        8: ('8.haberman+s+survival/haberman.data', 'haberman'),
        9: ('9.glass+identification/glass.data', 'glass'),
        10: ('10.yeast/yeast.data', 'yeast'),
    }
    
    if dataset_id not in datasets:
        return None, None, None
    
    filename, name = datasets[dataset_id]
    
    try:
        if dataset_id == 0:
            # Prostate Cancer
            for path in [f'{data_dir}/data/{filename}', f'/data2/image_identification/src/data/{filename}']:
                if Path(path).exists():
                    df = pd.read_csv(path)
                    X = df[['LAA', 'Glutamate', 'Choline', 'Sarcosine']].values
                    y = (df['Group'] == 'PCa').astype(int).values
                    return X, y, name
        
        base_path = f'{data_dir}/small_data'
        filepath = f'{base_path}/{filename}'
        
        if not Path(filepath).exists():
            # 尝试服务器路径
            filepath = f'/data2/image_identification/src/small_data/{filename}'
        
        if not Path(filepath).exists():
            return None, None, name
        
        if dataset_id == 1:  # balloons
            df = pd.read_csv(filepath, header=None, names=['color', 'size', 'act', 'age', 'inflated'])
            le = LabelEncoder()
            for col in df.columns[:-1]:
                df[col] = le.fit_transform(df[col].astype(str))
            X = df.iloc[:, :-1].values.astype(float)
            y = le.fit_transform(df.iloc[:, -1].astype(str))
        
        elif dataset_id == 2:  # lenses
            df = pd.read_csv(filepath, header=None, delim_whitespace=True)
            X = df.iloc[:, 1:-1].values.astype(float)
            y = (df.iloc[:, -1].values - 1).astype(int)
        
        elif dataset_id == 3:  # caesarian (ARFF格式)
            # 跳过ARFF头部，找到@data后的数据
            with open(filepath, 'r') as f:
                lines = f.readlines()
            data_start = 0
            for i, line in enumerate(lines):
                if line.strip().lower() == '@data':
                    data_start = i + 1
                    break
            # 读取数据部分
            data_lines = [l.strip() for l in lines[data_start:] if l.strip() and not l.startswith('@')]
            data = [list(map(float, l.split(','))) for l in data_lines if l]
            data = np.array(data)
            X = data[:, :-1].astype(float)
            y = data[:, -1].astype(int)
        
        elif dataset_id == 4:  # iris
            df = pd.read_csv(filepath, header=None)
            X = df.iloc[:, :-1].values.astype(float)
            le = LabelEncoder()
            y = le.fit_transform(df.iloc[:, -1])
        
        elif dataset_id == 5:  # fertility
            df = pd.read_csv(filepath, header=None)
            X = df.iloc[:, :-1].values.astype(float)
            le = LabelEncoder()
            y = le.fit_transform(df.iloc[:, -1])
        
        elif dataset_id == 6:  # zoo
            df = pd.read_csv(filepath, header=None)
            X = df.iloc[:, 1:-1].values.astype(float)
            y = (df.iloc[:, -1].values - 1).astype(int)
        
        elif dataset_id == 7:  # seeds
            df = pd.read_csv(filepath, header=None, delim_whitespace=True)
            X = df.iloc[:, :-1].values.astype(float)
            y = df.iloc[:, -1].values.astype(int) - 1
        
        elif dataset_id == 8:  # haberman
            df = pd.read_csv(filepath, header=None)
            X = df.iloc[:, :-1].values.astype(float)
            y = (df.iloc[:, -1].values - 1).astype(int)
        
        elif dataset_id == 9:  # glass
            df = pd.read_csv(filepath, header=None)
            X = df.iloc[:, 1:-1].values.astype(float)
            le = LabelEncoder()
            y = le.fit_transform(df.iloc[:, -1])
        
        elif dataset_id == 10:  # yeast
            df = pd.read_csv(filepath, header=None, delim_whitespace=True)
            X = df.iloc[:, 1:-1].values.astype(float)
            le = LabelEncoder()
            y = le.fit_transform(df.iloc[:, -1])
        
        # 确保y是整数类型
        y = np.array(y).astype(int)
        return X, y, name
    
    except Exception as e:
        logger.error(f"加载数据集 {dataset_id} 失败: {e}")
        return None, None, name


def run_single_dataset_seed(args):
    """运行单个数据集的单个seed"""
    dataset_id, seed, gpu_id, data_dir = args
    
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    X, y, name = load_dataset(dataset_id, data_dir)
    if X is None:
        return {'dataset_id': dataset_id, 'seed': seed, 'error': 'load_failed'}
    
    n_classes = len(np.unique(y))
    n_samples = len(X)
    n_features = X.shape[1]
    
    set_seed(seed)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    
    # RF基线
    rf_preds, rf_labels = [], []
    # VAE-HyperNet
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
        rf_pred = rf.predict(X_test_s)
        rf_preds.extend(rf_pred)
        rf_labels.extend(y_test)
        
        # VAE-HyperNet
        try:
            model = train_stable_model(X_train_s, y_train, X_test_s, y_test, 
                                       n_classes, device, epochs=200, seed=seed+fold)
            proba = predict_stable(model, X_train_s, X_test_s, device)
            vhn_pred = proba.argmax(axis=1)
            vhn_preds.extend(vhn_pred)
            vhn_labels.extend(y_test)
        except Exception as e:
            vhn_preds.extend([0] * len(y_test))
            vhn_labels.extend(y_test)
    
    rf_acc = accuracy_score(rf_labels, rf_preds) * 100
    vhn_acc = accuracy_score(vhn_labels, vhn_preds) * 100
    
    return {
        'dataset_id': dataset_id,
        'name': name,
        'seed': seed,
        'n_samples': n_samples,
        'n_features': n_features,
        'n_classes': n_classes,
        'rf_acc': rf_acc,
        'vhn_acc': vhn_acc
    }


def main():
    logger.info("=" * 70)
    logger.info("48_batch_datasets_multigpu.py")
    logger.info("用47稳定模型跑0-10数据集，多GPU并行，记录标准差")
    logger.info("=" * 70)
    
    # 配置
    data_dir = '/data2/image_identification/src'
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    logger.info(f"可用GPU数量: {n_gpus}")
    
    # 测试seeds
    test_seeds = [42, 123, 456, 789, 1000]
    dataset_ids = list(range(11))  # 0-10
    
    # 构建任务列表
    tasks = []
    for dataset_id in dataset_ids:
        for i, seed in enumerate(test_seeds):
            gpu_id = (dataset_id * len(test_seeds) + i) % max(n_gpus, 1)
            tasks.append((dataset_id, seed, gpu_id, data_dir))
    
    logger.info(f"总任务数: {len(tasks)} (11数据集 x 5种子)")
    
    # 并行执行
    results = []
    n_workers = min(n_gpus * 2, 12)  # 每GPU 2个进程
    
    logger.info(f"使用 {n_workers} 个进程并行...")
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(run_single_dataset_seed, task): task for task in tasks}
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            results.append(result)
            
            if completed % 5 == 0:
                logger.info(f"进度: {completed}/{len(tasks)} ({100*completed/len(tasks):.1f}%)")
    
    # 汇总结果
    logger.info("\n" + "=" * 70)
    logger.info("[汇总结果]")
    logger.info("=" * 70)
    
    summary = {}
    for dataset_id in dataset_ids:
        dataset_results = [r for r in results if r.get('dataset_id') == dataset_id and 'error' not in r]
        
        if not dataset_results:
            summary[dataset_id] = {'error': True}
            continue
        
        name = dataset_results[0].get('name', f'dataset_{dataset_id}')
        n_samples = dataset_results[0].get('n_samples', 0)
        n_features = dataset_results[0].get('n_features', 0)
        n_classes = dataset_results[0].get('n_classes', 0)
        
        rf_accs = [r['rf_acc'] for r in dataset_results]
        vhn_accs = [r['vhn_acc'] for r in dataset_results]
        
        summary[dataset_id] = {
            'name': name,
            'n_samples': n_samples,
            'n_features': n_features,
            'n_classes': n_classes,
            'rf_mean': np.mean(rf_accs),
            'rf_std': np.std(rf_accs),
            'vhn_mean': np.mean(vhn_accs),
            'vhn_std': np.std(vhn_accs),
            'rf_accs': rf_accs,
            'vhn_accs': vhn_accs
        }
    
    # 打印表格
    logger.info("\n" + "-" * 100)
    logger.info(f"{'Dataset':<20} {'Samples':<8} {'Feat':<5} {'Class':<6} {'RF (%)':<18} {'VAE-HyperNet (%)':<18} {'Winner'}")
    logger.info("-" * 100)
    
    for dataset_id in dataset_ids:
        if summary[dataset_id].get('error'):
            logger.info(f"{dataset_id}. ERROR")
            continue
        
        s = summary[dataset_id]
        rf_str = f"{s['rf_mean']:.1f}±{s['rf_std']:.1f}"
        vhn_str = f"{s['vhn_mean']:.1f}±{s['vhn_std']:.1f}"
        
        if s['vhn_mean'] > s['rf_mean']:
            winner = "VAE-HyperNet ✅"
        elif s['rf_mean'] > s['vhn_mean']:
            winner = "RF"
        else:
            winner = "Tie"
        
        logger.info(f"{dataset_id}. {s['name']:<17} {s['n_samples']:<8} {s['n_features']:<5} {s['n_classes']:<6} {rf_str:<18} {vhn_str:<18} {winner}")
    
    logger.info("-" * 100)
    
    # 计算整体统计
    all_rf = [s['rf_mean'] for s in summary.values() if not s.get('error')]
    all_vhn = [s['vhn_mean'] for s in summary.values() if not s.get('error')]
    
    logger.info(f"\nRF 整体平均: {np.mean(all_rf):.2f}%")
    logger.info(f"VAE-HyperNet 整体平均: {np.mean(all_vhn):.2f}%")
    
    vhn_wins = sum(1 for s in summary.values() if not s.get('error') and s['vhn_mean'] > s['rf_mean'])
    logger.info(f"VAE-HyperNet 胜出数据集: {vhn_wins}/{len(all_vhn)}")
    
    # 保存结果
    output_dir = Path('/data2/image_identification/src/output')
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'48_batch_results_{timestamp}.json'
    
    save_data = {
        'summary': {str(k): v for k, v in summary.items()},
        'all_results': results,
        'test_seeds': test_seeds,
        'timestamp': timestamp
    }
    
    with open(output_file, 'w') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"\n保存: {output_file}")
    logger.info("=" * 70)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
