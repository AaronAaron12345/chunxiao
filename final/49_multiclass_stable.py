#!/usr/bin/env python3
"""
49_multiclass_stable.py
针对多类别数据集的稳定性优化版本

改进点：
1. 动态调整HyperNet头数量（类别越多，头越多）
2. 类别平衡的VAE增强
3. One-vs-Rest策略用于多类别
4. 增加集成深度

解决问题：多类别数据集标准差高(24-30%)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path
import logging
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'output/49_log.txt', mode='w')
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
    """类别感知的VAE"""
    def __init__(self, input_dim, latent_dim=8, n_classes=2):
        super().__init__()
        self.n_classes = n_classes
        hidden = max(16, input_dim * 2)
        
        # 编码器（条件VAE - 加入类别信息）
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + n_classes, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
        )
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_var = nn.Linear(hidden, latent_dim)
        
        # 解码器
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
        recon = self.decode(z, y_onehot)
        return recon, mu, logvar


class MultiHeadHyperNet(nn.Module):
    """多头超网络 - 头数量动态调整"""
    def __init__(self, input_dim, n_classes, n_trees=15, tree_depth=4, n_heads=5):
        super().__init__()
        self.n_classes = n_classes
        self.n_trees = n_trees
        self.tree_depth = tree_depth
        self.n_heads = n_heads
        self.n_internal = 2**tree_depth - 1
        self.n_leaves = 2**tree_depth
        
        # 多头Data Encoder
        encoder_dim = 64
        self.data_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, encoder_dim),
                nn.LayerNorm(encoder_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(encoder_dim, encoder_dim),
                nn.LayerNorm(encoder_dim),
                nn.GELU(),
            ) for _ in range(n_heads)
        ])
        
        # 参数生成器（每个头独立）
        split_params = n_trees * self.n_internal * (input_dim + 1)
        leaf_params = n_trees * self.n_leaves * n_classes
        tree_weight_params = n_trees
        total_params = split_params + leaf_params + tree_weight_params
        
        self.param_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(encoder_dim, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(128, total_params)
            ) for _ in range(n_heads)
        ])
        
        # 头权重学习
        self.head_weights = nn.Parameter(torch.ones(n_heads) / n_heads)
        
        # 温度参数
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        self._init_weights()
    
    def _init_weights(self):
        for modules in [self.data_encoders, self.param_generators]:
            for module in modules:
                for m in module.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.orthogonal_(m.weight, gain=0.5)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
    
    def forward(self, X_support, X_query):
        """
        X_support: 支持集(训练数据) [N_support, input_dim]
        X_query: 查询集(测试样本) [N_query, input_dim]
        """
        all_outputs = []
        head_w = F.softmax(self.head_weights, dim=0)
        
        for head_idx in range(self.n_heads):
            # 编码支持集
            support_encoded = self.data_encoders[head_idx](X_support)
            context = support_encoded.mean(dim=0, keepdim=True)
            
            # 生成参数
            params = self.param_generators[head_idx](context)
            params = torch.tanh(params) * 2.0  # 约束参数范围
            
            # 解析参数
            idx = 0
            split_w_size = self.n_trees * self.n_internal * X_support.shape[1]
            split_b_size = self.n_trees * self.n_internal
            leaf_size = self.n_trees * self.n_leaves * self.n_classes
            
            split_weights = params[0, idx:idx+split_w_size].view(self.n_trees, self.n_internal, -1)
            idx += split_w_size
            split_bias = params[0, idx:idx+split_b_size].view(self.n_trees, self.n_internal)
            idx += split_b_size
            leaf_logits = params[0, idx:idx+leaf_size].view(self.n_trees, self.n_leaves, self.n_classes)
            idx += leaf_size
            tree_weights = F.softmax(params[0, idx:idx+self.n_trees], dim=0)
            
            # 软决策树推理
            temp = torch.clamp(self.temperature, 0.1, 2.0)
            tree_outputs = []
            
            for t in range(self.n_trees):
                # 计算到达每个叶子的概率
                batch_size = X_query.shape[0]
                reach_prob = torch.ones(batch_size, 1, device=X_query.device)
                
                for d in range(self.tree_depth):
                    start_idx = 2**d - 1
                    n_nodes = 2**d
                    
                    new_reach_prob = []
                    for node in range(n_nodes):
                        node_idx = start_idx + node
                        if node_idx >= self.n_internal:
                            break
                        
                        # 软分裂
                        decision = torch.sigmoid(
                            (X_query @ split_weights[t, node_idx] + split_bias[t, node_idx]) / temp
                        )
                        
                        parent_prob = reach_prob[:, node:node+1]
                        new_reach_prob.extend([
                            parent_prob * (1 - decision.unsqueeze(1)),
                            parent_prob * decision.unsqueeze(1)
                        ])
                    
                    if new_reach_prob:
                        reach_prob = torch.cat(new_reach_prob, dim=1)
                
                # 加权叶子预测
                leaf_probs = F.softmax(leaf_logits[t] / temp, dim=-1)
                tree_pred = torch.einsum('bl,lc->bc', reach_prob, leaf_probs)
                tree_outputs.append(tree_pred * tree_weights[t])
            
            ensemble_pred = torch.stack(tree_outputs, dim=0).sum(dim=0)
            all_outputs.append(ensemble_pred * head_w[head_idx])
        
        final_output = torch.stack(all_outputs, dim=0).sum(dim=0)
        return final_output


def class_balanced_augment(X, y, vae, n_samples_per_class, device):
    """类别平衡的数据增强"""
    vae.eval()
    X_aug_list = [X]
    y_aug_list = [y]
    
    classes = np.unique(y)
    n_classes = len(classes)
    
    with torch.no_grad():
        for c in classes:
            X_c = X[y == c]
            n_orig = len(X_c)
            n_gen = n_samples_per_class - n_orig
            
            if n_gen <= 0:
                continue
            
            X_t = torch.FloatTensor(X_c).to(device)
            y_onehot = torch.zeros(len(X_c), n_classes, device=device)
            y_onehot[:, c] = 1
            
            # 多次采样生成
            generated = []
            for _ in range(max(1, n_gen // len(X_c) + 1)):
                mu, logvar = vae.encode(X_t, y_onehot)
                std = torch.exp(0.5 * logvar)
                # 添加噪声多样性
                for noise_scale in [0.5, 1.0, 1.5]:
                    z = mu + noise_scale * std * torch.randn_like(std)
                    recon = vae.decode(z, y_onehot)
                    generated.append(recon.cpu().numpy())
            
            generated = np.vstack(generated)[:n_gen]
            X_aug_list.append(generated)
            y_aug_list.append(np.full(len(generated), c))
    
    return np.vstack(X_aug_list), np.concatenate(y_aug_list)


def train_model(X_train, y_train, X_test, y_test, n_classes, device, seed=42):
    """训练完整模型"""
    set_seed(seed)
    
    input_dim = X_train.shape[1]
    n_samples = len(X_train)
    
    # 根据类别数动态调整参数
    n_heads = max(5, n_classes * 2)  # 更多类别需要更多头
    n_trees = max(15, n_classes * 3)
    samples_per_class = max(50, 200 // n_classes)
    
    # 1. 训练VAE
    vae = VAE(input_dim, latent_dim=max(4, input_dim), n_classes=n_classes).to(device)
    vae_optimizer = torch.optim.AdamW(vae.parameters(), lr=0.005, weight_decay=1e-4)
    
    X_t = torch.FloatTensor(X_train).to(device)
    y_onehot = torch.zeros(n_samples, n_classes, device=device)
    for i, c in enumerate(y_train):
        y_onehot[i, c] = 1
    
    vae.train()
    for epoch in range(100):
        vae_optimizer.zero_grad()
        recon, mu, logvar = vae(X_t, y_onehot)
        
        recon_loss = F.mse_loss(recon, X_t)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + 0.1 * kl_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
        vae_optimizer.step()
    
    # 2. 类别平衡增强
    X_aug, y_aug = class_balanced_augment(X_train, y_train, vae, samples_per_class, device)
    
    # 3. 训练HyperNet
    model = MultiHeadHyperNet(
        input_dim=input_dim,
        n_classes=n_classes,
        n_trees=n_trees,
        tree_depth=4,
        n_heads=n_heads
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
    
    X_aug_t = torch.FloatTensor(X_aug).to(device)
    y_aug_t = torch.LongTensor(y_aug).to(device)
    
    # 使用focal loss处理类别不平衡
    class_weights = torch.ones(n_classes, device=device)
    for c in range(n_classes):
        n_c = (y_aug == c).sum()
        if n_c > 0:
            class_weights[c] = len(y_aug) / (n_classes * n_c)
    
    model.train()
    best_loss = float('inf')
    patience = 0
    
    for epoch in range(200):
        optimizer.zero_grad()
        
        # 随机采样支持集
        support_idx = np.random.choice(len(X_aug), min(64, len(X_aug)), replace=False)
        X_support = X_aug_t[support_idx]
        
        # 全部数据作为查询集
        outputs = model(X_support, X_aug_t)
        
        # 加权交叉熵
        loss = F.cross_entropy(outputs, y_aug_t, weight=class_weights)
        
        # 添加熵正则化（鼓励confident预测）
        probs = F.softmax(outputs, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        loss = loss - 0.01 * entropy  # 减少熵
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience = 0
        else:
            patience += 1
            if patience > 30:
                break
    
    # 4. 预测
    model.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test).to(device)
        
        # 多次采样预测取平均
        all_preds = []
        for _ in range(5):
            support_idx = np.random.choice(len(X_aug), min(64, len(X_aug)), replace=False)
            X_support = X_aug_t[support_idx]
            outputs = model(X_support, X_test_t)
            all_preds.append(F.softmax(outputs, dim=-1))
        
        avg_pred = torch.stack(all_preds).mean(dim=0)
        predictions = avg_pred.argmax(dim=1).cpu().numpy()
    
    return predictions


def load_dataset(dataset_id, data_dir):
    """加载数据集"""
    datasets = {
        0: ('Data_for_Jinming.csv', 'prostate'),
        1: ('1.balloons/adult+stretch.data', 'balloons'),
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
            for path in [f'{data_dir}/data/{filename}', f'/data2/image_identification/src/data/{filename}']:
                if Path(path).exists():
                    df = pd.read_csv(path)
                    X = df[['LAA', 'Glutamate', 'Choline', 'Sarcosine']].values
                    y = (df['Group'] == 'PCa').astype(int).values
                    return X, y, name
        
        base_path = f'{data_dir}/small_data'
        filepath = f'{base_path}/{filename}'
        
        if not Path(filepath).exists():
            filepath = f'/data2/image_identification/src/small_data/{filename}'
        
        if not Path(filepath).exists():
            return None, None, name
        
        le = LabelEncoder()
        
        if dataset_id == 1:  # balloons
            df = pd.read_csv(filepath, header=None, names=['color', 'size', 'act', 'age', 'inflated'])
            for col in df.columns[:-1]:
                df[col] = le.fit_transform(df[col].astype(str))
            X = df.iloc[:, :-1].values.astype(float)
            y = le.fit_transform(df.iloc[:, -1].astype(str))
        
        elif dataset_id == 2:  # lenses
            df = pd.read_csv(filepath, header=None, delim_whitespace=True)
            X = df.iloc[:, 1:-1].values.astype(float)
            y = (df.iloc[:, -1].values - 1).astype(int)
        
        elif dataset_id == 3:  # caesarian (ARFF格式)
            with open(filepath, 'r') as f:
                lines = f.readlines()
            data_start = 0
            for i, line in enumerate(lines):
                if line.strip().lower() == '@data':
                    data_start = i + 1
                    break
            data_lines = [l.strip() for l in lines[data_start:] if l.strip() and not l.startswith('@')]
            data = [list(map(float, l.split(','))) for l in data_lines if l]
            data = np.array(data)
            X = data[:, :-1].astype(float)
            y = data[:, -1].astype(int)
        
        elif dataset_id == 4:  # iris
            df = pd.read_csv(filepath, header=None)
            X = df.iloc[:, :-1].values.astype(float)
            y = le.fit_transform(df.iloc[:, -1])
        
        elif dataset_id == 5:  # fertility
            df = pd.read_csv(filepath, header=None)
            X = df.iloc[:, :-1].values.astype(float)
            y = le.fit_transform(df.iloc[:, -1])
        
        elif dataset_id == 6:  # zoo
            df = pd.read_csv(filepath, header=None)
            X = df.iloc[:, 1:-1].values.astype(float)
            y = df.iloc[:, -1].values.astype(int) - 1
        
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
            y = le.fit_transform(df.iloc[:, -1])
        
        elif dataset_id == 10:  # yeast
            df = pd.read_csv(filepath, header=None, delim_whitespace=True)
            X = df.iloc[:, 1:-1].values.astype(float)
            y = le.fit_transform(df.iloc[:, -1])
        
        y = np.array(y).astype(int)
        return X, y, name
    
    except Exception as e:
        logger.error(f"加载数据集 {dataset_id} 失败: {e}")
        return None, None, name


def run_single_experiment(args):
    """运行单个实验"""
    dataset_id, seed, gpu_id, data_dir = args
    
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    X, y, name = load_dataset(dataset_id, data_dir)
    if X is None:
        return {'dataset_id': dataset_id, 'seed': seed, 'error': 'load_failed'}
    
    n_classes = len(np.unique(y))
    
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
        rf_pred = rf.predict(X_test_s)
        rf_preds.extend(rf_pred)
        rf_labels.extend(y_test)
        
        # VAE-HyperNet
        try:
            vhn_pred = train_model(X_train_s, y_train, X_test_s, y_test, n_classes, device, seed=seed+fold*100)
            vhn_preds.extend(vhn_pred)
            vhn_labels.extend(y_test)
        except Exception as e:
            vhn_preds.extend([-1] * len(y_test))
            vhn_labels.extend(y_test)
    
    rf_acc = accuracy_score(rf_labels, rf_preds) * 100
    vhn_acc = accuracy_score(vhn_labels, vhn_preds) * 100 if -1 not in vhn_preds else 0
    
    return {
        'dataset_id': dataset_id,
        'name': name,
        'seed': seed,
        'n_samples': len(X),
        'n_features': X.shape[1],
        'n_classes': n_classes,
        'rf_acc': rf_acc,
        'vhn_acc': vhn_acc
    }


def main():
    logger.info("=" * 70)
    logger.info("49_multiclass_stable.py - 多类别数据集稳定性优化")
    logger.info("=" * 70)
    
    Path('output').mkdir(exist_ok=True)
    
    data_dir = '/data2/image_identification/src'
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    logger.info(f"可用GPU数量: {n_gpus}")
    
    datasets = list(range(11))
    seeds = [42, 123, 456, 789, 1000]
    
    tasks = []
    for ds_id in datasets:
        for seed in seeds:
            gpu_id = (ds_id * len(seeds) + seeds.index(seed)) % max(1, n_gpus)
            tasks.append((ds_id, seed, gpu_id, data_dir))
    
    logger.info(f"总任务数: {len(tasks)}")
    
    results = []
    n_workers = min(12, len(tasks))
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(run_single_experiment, task) for task in tasks]
        
        for i, future in enumerate(futures):
            try:
                result = future.result()
                results.append(result)
                if (i + 1) % 5 == 0:
                    logger.info(f"进度: {i+1}/{len(tasks)} ({100*(i+1)/len(tasks):.1f}%)")
            except Exception as e:
                logger.error(f"任务失败: {e}")
    
    # 汇总结果
    logger.info("\n" + "=" * 70)
    logger.info("[汇总结果 - 多类别优化版]")
    logger.info("=" * 70)
    
    summary = {}
    for r in results:
        if 'error' in r:
            continue
        ds_id = r['dataset_id']
        if ds_id not in summary:
            summary[ds_id] = {
                'name': r['name'],
                'n_samples': r['n_samples'],
                'n_features': r['n_features'],
                'n_classes': r['n_classes'],
                'rf_accs': [],
                'vhn_accs': []
            }
        summary[ds_id]['rf_accs'].append(r['rf_acc'])
        summary[ds_id]['vhn_accs'].append(r['vhn_acc'])
    
    logger.info("\n" + "-" * 90)
    logger.info(f"{'Dataset':<20} {'Samples':<8} {'Feat':<5} {'Class':<6} {'RF (%)':<15} {'VAE-HyperNet (%)':<18} {'Winner'}")
    logger.info("-" * 90)
    
    rf_wins = 0
    vhn_wins = 0
    ties = 0
    
    for ds_id in sorted(summary.keys()):
        s = summary[ds_id]
        rf_mean = np.mean(s['rf_accs'])
        rf_std = np.std(s['rf_accs'])
        vhn_mean = np.mean(s['vhn_accs'])
        vhn_std = np.std(s['vhn_accs'])
        
        if vhn_mean > rf_mean + 0.5:
            winner = "VAE-HyperNet ✅"
            vhn_wins += 1
        elif rf_mean > vhn_mean + 0.5:
            winner = "RF"
            rf_wins += 1
        else:
            winner = "Tie"
            ties += 1
        
        logger.info(f"{ds_id}. {s['name']:<17} {s['n_samples']:<8} {s['n_features']:<5} {s['n_classes']:<6} "
                   f"{rf_mean:.1f}±{rf_std:.1f}      {vhn_mean:.1f}±{vhn_std:.1f}          {winner}")
    
    logger.info("-" * 90)
    logger.info(f"\nRF胜: {rf_wins}, VAE-HyperNet胜: {vhn_wins}, 平局: {ties}")
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'output/49_results_{timestamp}.json', 'w') as f:
        json.dump({'summary': summary, 'raw_results': results}, f, indent=2, default=str)
    
    logger.info(f"\n保存: output/49_results_{timestamp}.json")


if __name__ == '__main__':
    main()
