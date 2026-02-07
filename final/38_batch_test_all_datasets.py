#!/usr/bin/env python3
"""
38_batch_test_all_datasets.py - 批量测试所有10个小数据集
使用VAE-HyperNet-Fusion框架 (37_true_hypernet.py的方法)

修复版：支持多分类，正确处理标签编码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json
from datetime import datetime
from pathlib import Path
import random
import time
import warnings
warnings.filterwarnings('ignore')

# ============== 设置 ==============
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============== VAE 数据增强 ==============
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        hidden = max(32, input_dim * 2)
        
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


# ============== HyperNetwork ==============
class HyperNetworkForTree(nn.Module):
    def __init__(self, input_dim, n_classes, n_trees=15, tree_depth=3, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.n_trees = n_trees
        self.tree_depth = tree_depth
        self.n_classes = n_classes
        
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
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.hyper_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, self.total_params)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, X_train):
        encoded = self.data_encoder(X_train)
        context = encoded.mean(dim=0, keepdim=True)
        params = self.hyper_net(context)
        params = params.squeeze(0)
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


class GeneratedTreeClassifier(nn.Module):
    def __init__(self, tree_depth=3, n_classes=2):
        super().__init__()
        self.depth = tree_depth
        self.n_internal = 2**tree_depth - 1
        self.n_leaves = 2**tree_depth
        self.n_classes = n_classes
        self.temperature = 1.0
    
    def forward_single_tree(self, x, split_weights, split_bias, leaf_logits):
        batch_size = x.size(0)
        leaf_probs = torch.zeros(batch_size, self.n_leaves, device=x.device)
        
        for leaf_idx in range(self.n_leaves):
            node_idx = leaf_idx + self.n_internal
            prob = torch.ones(batch_size, 1, device=x.device)
            
            path = []
            while node_idx > 0:
                parent = (node_idx - 1) // 2
                is_right = (node_idx == 2 * parent + 2)
                path.append((parent, is_right))
                node_idx = parent
            
            path.reverse()
            
            for parent, is_right in path:
                if parent < self.n_internal:
                    split_logit = torch.sum(x * split_weights[parent], dim=1, keepdim=True) + split_bias[parent]
                    split_prob = torch.sigmoid(split_logit / self.temperature)
                    if is_right:
                        prob = prob * split_prob
                    else:
                        prob = prob * (1 - split_prob)
            
            leaf_probs[:, leaf_idx:leaf_idx+1] = prob
        
        leaf_distributions = F.softmax(leaf_logits, dim=-1)
        output = torch.einsum('bl,lc->bc', leaf_probs, leaf_distributions)
        
        return output
    
    def forward(self, x, trees_params, tree_weights):
        outputs = []
        for t_params in trees_params:
            out = self.forward_single_tree(
                x, 
                t_params['split_weights'],
                t_params['split_bias'],
                t_params['leaf_logits']
            )
            outputs.append(out)
        
        outputs = torch.stack(outputs, dim=0)
        weights = F.softmax(tree_weights, dim=0).view(-1, 1, 1)
        ensemble_output = (outputs * weights).sum(dim=0)
        
        return ensemble_output


class VAEHyperNetFusion(nn.Module):
    def __init__(self, input_dim, n_trees=15, tree_depth=3, n_classes=2):
        super().__init__()
        self.vae = VAE(input_dim, latent_dim=max(4, input_dim))
        self.hypernet = HyperNetworkForTree(input_dim, n_classes, n_trees, tree_depth)
        self.classifier = GeneratedTreeClassifier(tree_depth, n_classes)
        self.input_dim = input_dim
        self.n_classes = n_classes
    
    def augment_data(self, X, n_augment=100):
        self.vae.eval()
        with torch.no_grad():
            recon, mu, logvar = self.vae(X)
            
            augmented = [X]
            n_orig = X.size(0)
            
            for _ in range(n_augment // n_orig + 1):
                alpha = torch.rand(n_orig, 1, device=X.device) * 0.5 + 0.25
                aug = alpha * X + (1 - alpha) * recon
                noise = torch.randn_like(aug) * 0.05
                augmented.append(aug + noise)
            
            augmented = torch.cat(augmented, dim=0)[:n_augment + n_orig]
        
        return augmented
    
    def forward(self, X_train, X_test, y_train):
        params = self.hypernet(X_train)
        trees_params, tree_weights = self.hypernet.parse_params(params)
        logits = self.classifier(X_test, trees_params, tree_weights)
        return logits


# ============== 数据加载器 ==============
def load_dataset(dataset_name, data_dir):
    if dataset_name == '1.balloons':
        return load_balloons(data_dir / dataset_name)
    elif dataset_name == '2lens':
        return load_lenses(data_dir / dataset_name)
    elif dataset_name == '2.liver+disorders':
        return load_liver(data_dir / dataset_name)
    elif dataset_name == '3.caesarian+section+classification+dataset':
        return load_caesarian(data_dir / dataset_name)
    elif dataset_name == '4.iris':
        return load_iris(data_dir / dataset_name)
    elif dataset_name == '5.fertility':
        return load_fertility(data_dir / dataset_name)
    elif dataset_name == '6.zoo':
        return load_zoo(data_dir / dataset_name)
    elif dataset_name == '7.seeds':
        return load_seeds(data_dir / dataset_name)
    elif dataset_name == '8.haberman+s+survival':
        return load_haberman(data_dir / dataset_name)
    elif dataset_name == '9.glass+identification':
        return load_glass(data_dir / dataset_name)
    elif dataset_name == '10.yeast':
        return load_yeast(data_dir / dataset_name)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def load_balloons(path):
    dfs = []
    for f in ['adult+stretch.data', 'adult-stretch.data', 'yellow-small.data', 'yellow-small+adult-stretch.data']:
        fp = path / f
        if fp.exists():
            df = pd.read_csv(fp, header=None, names=['color', 'size', 'act', 'age', 'inflated'])
            dfs.append(df)
    df = pd.concat(dfs, ignore_index=True).drop_duplicates()
    for col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = LabelEncoder().fit_transform(df.iloc[:, -1].values)
    return X, y, 'Balloons'


def load_lenses(path):
    df = pd.read_csv(path / 'lenses.data', header=None, delim_whitespace=True)
    X = df.iloc[:, 1:-1].values.astype(np.float32)
    y = LabelEncoder().fit_transform(df.iloc[:, -1].values)
    return X, y, 'Lenses'


def load_liver(path):
    df = pd.read_csv(path / 'bupa.data', header=None)
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = LabelEncoder().fit_transform(df.iloc[:, -1].values)
    return X, y, 'Liver Disorders'


def load_caesarian(path):
    df = pd.read_csv(path / 'caesarian.csv')
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = LabelEncoder().fit_transform(df.iloc[:, -1].values)
    return X, y, 'Caesarian Section'


def load_iris(path):
    df = pd.read_csv(path / 'iris.data', header=None)
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = LabelEncoder().fit_transform(df.iloc[:, -1].values)
    return X, y, 'Iris'


def load_fertility(path):
    df = pd.read_csv(path / 'fertility_Diagnosis.txt', header=None)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = LabelEncoder().fit_transform(df.iloc[:, -1].values)
    return X, y, 'Fertility'


def load_zoo(path):
    df = pd.read_csv(path / 'zoo.data', header=None)
    X = df.iloc[:, 1:-1].values.astype(np.float32)
    y = LabelEncoder().fit_transform(df.iloc[:, -1].values)
    return X, y, 'Zoo'


def load_seeds(path):
    df = pd.read_csv(path / 'seeds_dataset.txt', header=None, delim_whitespace=True)
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = LabelEncoder().fit_transform(df.iloc[:, -1].values)
    return X, y, 'Seeds'


def load_haberman(path):
    df = pd.read_csv(path / 'haberman.data', header=None)
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = LabelEncoder().fit_transform(df.iloc[:, -1].values)
    return X, y, 'Haberman Survival'


def load_glass(path):
    df = pd.read_csv(path / 'glass.data', header=None)
    X = df.iloc[:, 1:-1].values.astype(np.float32)
    y = LabelEncoder().fit_transform(df.iloc[:, -1].values)
    return X, y, 'Glass Identification'


def load_yeast(path):
    df = pd.read_csv(path / 'yeast.data', header=None, delim_whitespace=True)
    X = df.iloc[:, 1:-1].values.astype(np.float32)
    y = LabelEncoder().fit_transform(df.iloc[:, -1].values)
    return X, y, 'Yeast'


# ============== 训练和评估 ==============
def train_vae(model, X, epochs=200, lr=0.01):
    optimizer = torch.optim.Adam(model.vae.parameters(), lr=lr)
    model.vae.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        recon, mu, logvar = model.vae(X)
        recon_loss = F.mse_loss(recon, X)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / X.size(0)
        loss = recon_loss + 0.001 * kl_loss
        loss.backward()
        optimizer.step()


def train_hypernet(model, X_train, y_train, epochs=300, lr=0.005):
    optimizer = torch.optim.AdamW(
        list(model.hypernet.parameters()) + list(model.classifier.parameters()),
        lr=lr, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    model.hypernet.train()
    model.classifier.train()
    
    best_loss = float('inf')
    patience = 50
    no_improve = 0
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        X_aug = model.augment_data(X_train, n_augment=150)
        y_aug = y_train.repeat((X_aug.size(0) // y_train.size(0)) + 1)[:X_aug.size(0)]
        
        logits = model(X_aug, X_aug, y_aug)
        loss = F.cross_entropy(logits, y_aug)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break


def evaluate_single_run(model, X_train_t, y_train_t, X_test_t, y_test, seed, n_classes):
    set_seed(seed)
    train_vae(model, X_train_t, epochs=200)
    train_hypernet(model, X_train_t, y_train_t, epochs=300)
    
    model.eval()
    with torch.no_grad():
        logits = model(X_train_t, X_test_t, y_train_t)
        preds = logits.argmax(dim=1).cpu().numpy()
    return preds


def evaluate_dataset(X, y, dataset_name, n_folds=5, n_runs=5):
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}, Classes: {len(np.unique(y))}")
    print(f"Class distribution: {np.bincount(y)}")
    print(f"{'='*60}")
    
    results = {
        'dataset': dataset_name,
        'n_samples': int(X.shape[0]),
        'n_features': int(X.shape[1]),
        'n_classes': int(len(np.unique(y))),
        'class_distribution': np.bincount(y).tolist(),
        'fold_results': [],
        'rf_fold_results': [],
    }
    
    n_classes = len(np.unique(y))
    input_dim = X.shape[1]
    
    actual_folds = min(n_folds, min(np.bincount(y)))
    if actual_folds < n_folds:
        print(f"Warning: Reducing folds from {n_folds} to {actual_folds}")
    
    skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
    
    all_preds = []
    all_true = []
    all_rf_preds = []
    all_probs = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        fold_start = time.time()
        print(f"\nFold {fold+1}/{actual_folds}", end='', flush=True)
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        X_train_t = torch.FloatTensor(X_train_scaled).to(device)
        X_test_t = torch.FloatTensor(X_test_scaled).to(device)
        y_train_t = torch.LongTensor(y_train).to(device)
        
        fold_preds_list = []
        fold_probs_list = []
        
        for run in range(n_runs):
            print(f".", end='', flush=True)
            model = VAEHyperNetFusion(input_dim, n_trees=15, tree_depth=3, n_classes=n_classes).to(device)
            preds = evaluate_single_run(model, X_train_t, y_train_t, X_test_t, y_test, seed=42+run+fold*100, n_classes=n_classes)
            fold_preds_list.append(preds)
            
            model.eval()
            with torch.no_grad():
                logits = model(X_train_t, X_test_t, y_train_t)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                fold_probs_list.append(probs)
        
        fold_preds_array = np.array(fold_preds_list)
        final_preds = np.apply_along_axis(lambda x: np.bincount(x, minlength=n_classes).argmax(), 0, fold_preds_array)
        avg_probs = np.mean(fold_probs_list, axis=0)
        
        fold_acc = accuracy_score(y_test, final_preds)
        all_preds.extend(final_preds)
        all_true.extend(y_test)
        all_probs.extend(avg_probs)
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_scaled, y_train)
        rf_preds = rf.predict(X_test_scaled)
        rf_acc = accuracy_score(y_test, rf_preds)
        all_rf_preds.extend(rf_preds)
        
        fold_time = time.time() - fold_start
        
        results['fold_results'].append({
            'fold': fold + 1,
            'accuracy': float(fold_acc),
            'n_test': len(y_test),
            'time_seconds': fold_time
        })
        results['rf_fold_results'].append({
            'fold': fold + 1,
            'accuracy': float(rf_acc)
        })
        
        print(f" VAE-HyperNet: {fold_acc*100:.2f}% | RF: {rf_acc*100:.2f}% ({fold_time:.1f}s)")
    
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    all_rf_preds = np.array(all_rf_preds)
    
    results['accuracy'] = float(accuracy_score(all_true, all_preds))
    results['accuracy_std'] = float(np.std([r['accuracy'] for r in results['fold_results']]))
    
    avg_type = 'binary' if n_classes == 2 else 'macro'
    results['precision'] = float(precision_score(all_true, all_preds, average=avg_type, zero_division=0))
    results['recall'] = float(recall_score(all_true, all_preds, average=avg_type, zero_division=0))
    results['f1'] = float(f1_score(all_true, all_preds, average=avg_type, zero_division=0))
    results['confusion_matrix'] = confusion_matrix(all_true, all_preds).tolist()
    
    results['rf_accuracy'] = float(accuracy_score(all_true, all_rf_preds))
    results['rf_accuracy_std'] = float(np.std([r['accuracy'] for r in results['rf_fold_results']]))
    
    results['predictions'] = {
        'true': all_true.tolist(),
        'pred': all_preds.tolist(),
        'probs': np.array(all_probs).tolist()
    }
    
    print(f"\n--- {dataset_name} Summary ---")
    print(f"VAE-HyperNet: {results['accuracy']*100:.2f}% (±{results['accuracy_std']*100:.2f}%)")
    print(f"RF Baseline:  {results['rf_accuracy']*100:.2f}% (±{results['rf_accuracy_std']*100:.2f}%)")
    print(f"Precision: {results['precision']*100:.2f}% | Recall: {results['recall']*100:.2f}% | F1: {results['f1']*100:.2f}%")
    
    return results


def main():
    data_dir = Path('/data2/image_identification/datasets/小数据')
    output_dir = Path('/data2/image_identification/src/final/output')
    output_dir.mkdir(exist_ok=True)
    
    datasets = [
        '1.balloons',
        '2lens',
        '3.caesarian+section+classification+dataset',
        '4.iris',
        '5.fertility',
        '6.zoo',
        '7.seeds',
        '8.haberman+s+survival',
        '9.glass+identification',
        '10.yeast',
    ]
    
    all_results = []
    start_time = time.time()
    
    for ds_name in datasets:
        try:
            ds_start = time.time()
            X, y, display_name = load_dataset(ds_name, data_dir)
            
            unique_labels = np.unique(y)
            print(f"\nLoading {display_name}: labels = {unique_labels}")
            assert np.all(unique_labels == np.arange(len(unique_labels))), f"Labels must be 0 to n-1"
            
            results = evaluate_dataset(X, y, display_name, n_folds=5, n_runs=5)
            results['time_seconds'] = time.time() - ds_start
            all_results.append(results)
            
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\nError processing {ds_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({'dataset': ds_name, 'error': str(e)})
    
    total_time = time.time() - start_time
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    output = {
        'timestamp': timestamp,
        'total_time_seconds': total_time,
        'total_time_minutes': total_time / 60,
        'device': str(device),
        'results': all_results
    }
    
    with open(output_dir / f'results_{timestamp}.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n" + "="*100)
    print("PAPER TABLE: VAE-HyperNet-Fusion Results")
    print("="*100)
    print(f"{'Dataset':<25} {'N':<6} {'Dim':<6} {'C':<4} {'VAE-HyperNet':<18} {'RF Baseline':<18} {'Diff':<8}")
    print("-"*100)
    
    valid_results = [r for r in all_results if 'error' not in r]
    
    for r in valid_results:
        acc_str = f"{r['accuracy']*100:.2f}±{r['accuracy_std']*100:.2f}"
        rf_str = f"{r['rf_accuracy']*100:.2f}±{r['rf_accuracy_std']*100:.2f}"
        diff = (r['accuracy'] - r['rf_accuracy']) * 100
        diff_str = f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}"
        print(f"{r['dataset']:<25} {r['n_samples']:<6} {r['n_features']:<6} {r['n_classes']:<4} {acc_str:<18} {rf_str:<18} {diff_str:<8}")
    
    print("-"*100)
    if valid_results:
        avg_acc = np.mean([r['accuracy'] for r in valid_results])
        avg_rf = np.mean([r['rf_accuracy'] for r in valid_results])
        avg_diff = (avg_acc - avg_rf) * 100
        print(f"{'Average':<25} {'':<6} {'':<6} {'':<4} {avg_acc*100:.2f}%{'':>10} {avg_rf*100:.2f}%{'':>10} {avg_diff:+.2f}%")
    
    print(f"\nTotal time: {total_time/60:.2f} minutes")
    print(f"Results saved to: {output_dir / f'results_{timestamp}.json'}")
    
    print("\n" + "="*80)
    print("Detailed Metrics")
    print("="*80)
    print(f"{'Dataset':<25} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-"*80)
    for r in valid_results:
        print(f"{r['dataset']:<25} {r['precision']*100:.2f}%{'':<5} {r['recall']*100:.2f}%{'':<5} {r['f1']*100:.2f}%")


if __name__ == '__main__':
    main()
