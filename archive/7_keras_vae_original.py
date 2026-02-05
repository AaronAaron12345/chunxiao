#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
7_keras_vae_original.py - 使用原始Keras VAE实现
================================================
复现原论文86.15%的准确率

关键差异：
1. 使用 Keras VAE（不是 PyTorch）
2. 使用 MinMaxScaler（不是 StandardScaler）
3. VAE hidden_dim = 512
4. binary_crossentropy 重建损失

运行: python 7_keras_vae_original.py
"""

import os
import sys
import json
import time
import warnings
import multiprocessing as mp
from datetime import datetime
from itertools import combinations
from functools import partial

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from skopt import BayesSearchCV
from sklearn.model_selection import KFold

# Keras imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import backend as K

warnings.filterwarnings('ignore')


# ==================== Keras VAE (原始实现) ====================
class CategorizedVAE:
    """原始论文中的VAE实现"""
    def __init__(self, input_dim, latent_dim=2, intermediate_dim=512, 
                 learning_rate=0.001, kl_weight=1.0, recon_weight=1.0):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self.learning_rate = learning_rate
        self.kl_weight = kl_weight
        self.recon_weight = recon_weight
        self.encoder = None
        self.decoder = None
        self.vae = self._build_vae()

    def _sampling(self, args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def _build_vae(self):
        # Encoder
        inputs = Input(shape=(self.input_dim,))
        h = Dense(self.intermediate_dim, activation='relu')(inputs)
        z_mean = Dense(self.latent_dim, activation='relu')(h)
        z_log_var = Dense(self.latent_dim, activation='relu')(h)
        z = Lambda(self._sampling, output_shape=(self.latent_dim,))([z_mean, z_log_var])
        self.encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        
        # Decoder
        latent_inputs = Input(shape=(self.latent_dim,))
        h_decoded = Dense(self.intermediate_dim, activation='relu')(latent_inputs)
        outputs_decoded = Dense(self.input_dim, activation='sigmoid')(h_decoded)
        self.decoder = Model(latent_inputs, outputs_decoded, name='decoder')
        
        # VAE
        outputs = self.decoder(self.encoder(inputs)[2])
        vae = Model(inputs, outputs)
        
        # Loss
        reconstruction_loss = binary_crossentropy(inputs, outputs) * self.input_dim * self.recon_weight
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1) * self.kl_weight * -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        vae.compile(optimizer=Adam(self.learning_rate))
        
        return vae

    def train(self, x_train, epochs=50, batch_size=128):
        self.vae.fit(x_train, shuffle=True, epochs=epochs, 
                     batch_size=batch_size, verbose=0,
                     callbacks=[EarlyStopping(monitor='loss', patience=10)])

    def encode(self, x):
        return self.encoder.predict(x, verbose=0)[2]

    def decode(self, z):
        return self.decoder.predict(z, verbose=0)


def linear_interpolate(point_a, point_b, num_points=5):
    """线性插值"""
    return np.linspace(point_a, point_b, num=num_points + 2)


def vae_augment_keras(X, y, num_interpolation_points=5):
    """使用Keras VAE进行数据增强（原始方法）"""
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(X)
    
    augmented_data = []
    augmented_labels = []
    
    for cls in np.unique(y):
        x_class = x_scaled[y == cls]
        
        # 训练VAE
        vae = CategorizedVAE(
            input_dim=x_scaled.shape[1],
            latent_dim=2,
            intermediate_dim=512,
            learning_rate=0.001,
            kl_weight=1.0,
            recon_weight=1.0
        )
        vae.train(x_class, epochs=50, batch_size=128)
        
        # 编码和解码
        encoded = vae.encode(x_class)
        decoded = vae.decode(encoded)
        
        # 线性插值生成增强数据
        for original, decoded_point in zip(x_class, decoded):
            interpolated = linear_interpolate(original, decoded_point, num_interpolation_points)
            # 只取中间的点（排除原始点和完全解码的点）
            augmented_data.extend(interpolated[1:-1])
            augmented_labels.extend([cls] * (len(interpolated) - 2))
    
    return np.array(augmented_data), np.array(augmented_labels), scaler


# ==================== 单fold处理 ====================
def process_fold(fold_info, X, y, config):
    """处理单个fold"""
    fold_idx, (train_idx, test_idx) = fold_info
    
    try:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # VAE增强（使用原始方法）
        X_aug, y_aug, scaler = vae_augment_keras(
            X_train, y_train, 
            num_interpolation_points=config['num_interpolation_points']
        )
        
        # 合并原始数据和增强数据
        X_train_scaled = scaler.transform(X_train)
        X_train_full = np.vstack([X_train_scaled, X_aug])
        y_train_full = np.concatenate([y_train, y_aug])
        
        # 标准化测试数据
        X_test_scaled = scaler.transform(X_test)
        
        # 随机森林
        clf = RandomForestClassifier(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            random_state=42,
            n_jobs=1
        )
        clf.fit(X_train_full, y_train_full)
        
        y_pred = clf.predict(X_test_scaled)
        y_prob = clf.predict_proba(X_test_scaled)
        
        return {
            'fold_idx': fold_idx,
            'accuracy': accuracy_score(y_test, y_pred),
            'y_true': y_test.tolist(),
            'y_pred': y_pred.tolist(),
            'y_prob': y_prob.tolist()
        }
    except Exception as e:
        return {'fold_idx': fold_idx, 'accuracy': 0.0, 'error': str(e)}


def run_cv(X, y, config, n_processes=None):
    """运行Leave-P-Out交叉验证"""
    n_samples = len(X)
    p = config['leave_p_out']
    
    all_indices = np.arange(n_samples)
    test_combos = list(combinations(all_indices, p))
    n_folds = len(test_combos)
    
    print(f"\n[INFO] Leave-{p}-Out 交叉验证")
    print(f"  样本数: {n_samples}, fold数: {n_folds}")
    
    fold_infos = []
    for fold_idx, test_idx in enumerate(test_combos):
        test_idx = np.array(test_idx)
        train_idx = np.setdiff1d(all_indices, test_idx)
        fold_infos.append((fold_idx, (train_idx, test_idx)))
    
    if n_processes is None:
        n_processes = min(mp.cpu_count(), n_folds, 32)
    
    print(f"  使用 {n_processes} 个进程")
    
    start_time = time.time()
    
    process_func = partial(process_fold, X=X, y=y, config=config)
    
    results = []
    with mp.Pool(processes=n_processes) as pool:
        for i, res in enumerate(pool.imap_unordered(process_func, fold_infos)):
            results.append(res)
            if (i + 1) % 50 == 0 or i + 1 == n_folds:
                pct = (i + 1) / n_folds * 100
                print(f"\r[进度] {i+1}/{n_folds} ({pct:.1f}%)", end='', flush=True)
    
    print()
    elapsed = time.time() - start_time
    
    # 统计
    accuracies = [r['accuracy'] for r in results if 'error' not in r]
    mean_acc = np.mean(accuracies) * 100
    std_acc = np.std(accuracies) * 100
    
    all_y_true, all_y_pred, all_y_prob = [], [], []
    for r in results:
        if 'error' not in r:
            all_y_true.extend(r['y_true'])
            all_y_pred.extend(r['y_pred'])
            all_y_prob.extend([p[1] if len(p) > 1 else p[0] for p in r['y_prob']])
    
    overall_acc = accuracy_score(all_y_true, all_y_pred) * 100
    
    try:
        auc = roc_auc_score(all_y_true, all_y_prob)
    except:
        auc = None
    
    print(f"\n{'='*60}")
    print(f"[结果] Leave-{p}-Out 完成")
    print(f"  平均准确率: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"  整体准确率: {overall_acc:.2f}%")
    if auc:
        print(f"  AUC: {auc:.4f}")
    print(f"  用时: {elapsed:.2f}秒")
    print(f"{'='*60}")
    
    return {
        'mean_accuracy': mean_acc / 100,
        'std_accuracy': std_acc / 100,
        'overall_accuracy': overall_acc / 100,
        'auc': auc,
        'elapsed_time': elapsed
    }


def main():
    print("=" * 60)
    print("7_keras_vae_original.py - 复现原论文方法")
    print("=" * 60)
    
    # 配置（与原论文一致）
    config = {
        'num_interpolation_points': 5,
        'n_estimators': 150,
        'max_depth': 5,
        'leave_p_out': 2
    }
    
    print(f"配置: {config}")
    
    # 加载数据
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'data', 'Data_for_Jinming.csv')
    
    df = pd.read_csv(data_path)
    feature_cols = ['LAA', 'Glutamate', 'Choline', 'Sarcosine']
    X = df[feature_cols].values.astype(np.float32)
    y = LabelEncoder().fit_transform(df['Group'].values)
    
    print(f"数据: {len(X)} 样本, {X.shape[1]} 特征")
    
    # 运行
    n_cpu = mp.cpu_count()
    results = run_cv(X, y, config, n_processes=min(32, n_cpu))
    
    # 保存
    output_dir = os.path.join(script_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(os.path.join(output_dir, f'keras_vae_results_{timestamp}.json'), 'w') as f:
        json.dump({**results, 'config': config}, f, indent=2)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
