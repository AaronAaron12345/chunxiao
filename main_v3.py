# -*- coding: utf-8 -*-
"""
VAE-HyperNetFusion V3: 优化版神经网络模型
========================================
针对小数据集的关键优化：
1. 更简单的网络架构（减少过拟合）
2. 正则化（L2权重衰减、Dropout）
3. 早停机制
4. 更多的VAE数据增强
5. 更好的训练策略（学习率调度）

作者: Jinming Zhang
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import LeavePOut
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
import json
import time
from datetime import datetime

from vae_augmentation import VAEDataAugmentation


class SimpleHyperNetFusion(nn.Module):
    """
    简化版 HyperNetFusion
    针对小数据集优化：更少的参数、更强的正则化
    """
    
    def __init__(self, input_dim, num_classes=2, hidden_dim=16, num_ensemble=5, dropout=0.3):
        super(SimpleHyperNetFusion, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_ensemble = num_ensemble
        
        # 超网络：根据输入动态调整权重（简化版）
        # 输出一个权重向量，用于加权集成
        self.hypernet = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_ensemble)
        )
        
        # 多个简单分类器（集成）
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            ) for _ in range(num_ensemble)
        ])
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # 超网络生成集成权重
        ensemble_weights = torch.softmax(self.hypernet(x), dim=1)  # [batch, num_ensemble]
        
        # 各分类器的输出
        outputs = []
        for clf in self.classifiers:
            outputs.append(clf(x))
        
        # 堆叠: [num_ensemble, batch, num_classes]
        outputs = torch.stack(outputs, dim=0)
        
        # 加权平均
        # weights: [batch, num_ensemble] -> [num_ensemble, batch, 1]
        weights = ensemble_weights.T.unsqueeze(2)
        
        # 加权求和: [batch, num_classes]
        weighted_output = (outputs * weights).sum(dim=0)
        
        return weighted_output
    
    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)


class VAEHyperNetFusionV3:
    """
    V3版本训练器：优化的HyperNetFusion神经网络
    
    关键优化：
    1. 更简单的网络架构
    2. 更多的数据增强
    3. 正则化策略（Dropout + L2）
    4. 学习率调度
    5. 多次训练取最佳
    """
    
    def __init__(self, config=None):
        self.config = config or self._default_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] 使用设备: {self.device}")
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.results = {}
    
    def _default_config(self):
        return {
            # VAE增强参数（增加数据量）
            'vae_latent_dim': 2,
            'vae_hidden_dim': 32,
            'vae_epochs': 100,
            'vae_lr': 0.001,
            'vae_kl_weight': 0.3,
            'num_interpolation_points': 10,  # 增加到10个插值点
            
            # 网络参数（简化）
            'hidden_dim': 16,
            'num_ensemble': 5,
            'dropout': 0.3,
            
            # 训练参数
            'train_epochs': 200,
            'train_lr': 0.01,
            'weight_decay': 0.01,  # L2正则化
            'batch_size': 16,
            'num_train_runs': 3,  # 多次训练取最佳
            
            # 验证参数
            'leave_p_out': 2
        }
    
    def load_data(self, filepath):
        print(f"[INFO] 加载数据: {filepath}")
        data = pd.read_csv(filepath)
        
        feature_cols = ['LAA', 'Glutamate', 'Choline', 'Sarcosine']
        X = data[feature_cols].values.astype(np.float32)
        y = data['Group'].values
        y = self.label_encoder.fit_transform(y)
        
        print(f"  - 样本数: {len(X)}")
        print(f"  - 特征数: {X.shape[1]}")
        print(f"  - 类别: {self.label_encoder.classes_}")
        print(f"  - 类别分布: {np.bincount(y)}")
        
        return X, y
    
    def _train_model(self, X_train, y_train, X_test, verbose=False):
        """训练单个模型，返回预测结果"""
        
        # VAE数据增强
        augmenter = VAEDataAugmentation(
            latent_dim=self.config['vae_latent_dim'],
            hidden_dim=self.config['vae_hidden_dim'],
            epochs=self.config['vae_epochs'],
            learning_rate=self.config['vae_lr'],
            kl_weight=self.config['vae_kl_weight'],
            num_interpolation_points=self.config['num_interpolation_points']
        )
        
        X_aug, y_aug = augmenter.augment(X_train, y_train)
        
        # 合并原始数据和增强数据
        X_train_full = np.vstack([X_train, X_aug])
        y_train_full = np.concatenate([y_train, y_aug])
        
        # 打乱训练数据
        shuffle_idx = np.random.permutation(len(X_train_full))
        X_train_full = X_train_full[shuffle_idx]
        y_train_full = y_train_full[shuffle_idx]
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_full)
        X_test_scaled = scaler.transform(X_test)
        
        # 转为PyTorch张量
        X_train_t = torch.FloatTensor(X_train_scaled).to(self.device)
        y_train_t = torch.LongTensor(y_train_full).to(self.device)
        X_test_t = torch.FloatTensor(X_test_scaled).to(self.device)
        
        best_pred = None
        best_proba = None
        best_train_acc = 0
        
        # 多次训练取最佳
        for run in range(self.config['num_train_runs']):
            # 创建模型
            model = SimpleHyperNetFusion(
                input_dim=X_train.shape[1],
                num_classes=len(np.unique(y_train_full)),
                hidden_dim=self.config['hidden_dim'],
                num_ensemble=self.config['num_ensemble'],
                dropout=self.config['dropout']
            ).to(self.device)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(
                model.parameters(), 
                lr=self.config['train_lr'],
                weight_decay=self.config['weight_decay']
            )
            
            # 学习率调度
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config['train_epochs']
            )
            
            # DataLoader
            train_dataset = TensorDataset(X_train_t, y_train_t)
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.config['batch_size'], 
                shuffle=True
            )
            
            # 训练
            model.train()
            for epoch in range(self.config['train_epochs']):
                epoch_loss = 0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                scheduler.step()
            
            # 评估训练准确率
            model.eval()
            with torch.no_grad():
                train_pred = model(X_train_t).argmax(dim=1).cpu().numpy()
                train_acc = accuracy_score(y_train_full, train_pred)
                
                # 测试预测
                proba = model.predict_proba(X_test_t).cpu().numpy()
                pred = np.argmax(proba, axis=1)
            
            # 保留训练准确率最高的模型
            if train_acc > best_train_acc:
                best_train_acc = train_acc
                best_pred = pred
                best_proba = proba
        
        return best_pred, best_proba
    
    def train_and_evaluate(self, X, y):
        """Leave-P-Out交叉验证"""
        p = self.config['leave_p_out']
        lpo = LeavePOut(p=p)
        n_splits = lpo.get_n_splits(X)
        
        print(f"\n[INFO] 开始Leave-{p}-Out交叉验证 (V3优化版)")
        print(f"  - 总splits数: {n_splits}")
        print(f"  - 网络配置: hidden_dim={self.config['hidden_dim']}, "
              f"num_ensemble={self.config['num_ensemble']}, dropout={self.config['dropout']}")
        print(f"  - 数据增强: {self.config['num_interpolation_points']}个插值点")
        
        start_time = time.time()
        
        all_preds = []
        all_true = []
        all_probs = []
        fold_accuracies = []
        
        for fold_id, (train_idx, test_idx) in enumerate(tqdm(lpo.split(X), total=n_splits, desc="Training V3")):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            pred, proba = self._train_model(X_train, y_train, X_test)
            
            fold_acc = accuracy_score(y_test, pred)
            fold_accuracies.append(fold_acc)
            
            all_preds.extend(pred.tolist())
            all_true.extend(y_test.tolist())
            if proba.shape[1] == 2:
                all_probs.extend(proba[:, 1].tolist())
            else:
                all_probs.extend(proba.tolist())
        
        elapsed_time = time.time() - start_time
        
        # 计算结果
        mean_acc = np.mean(fold_accuracies)
        std_acc = np.std(fold_accuracies)
        overall_acc = accuracy_score(all_true, all_preds)
        
        print(f"\n" + "="*60)
        print(f"[结果] Leave-{p}-Out 完成 (V3优化版)")
        print(f"  - 平均准确率: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
        print(f"  - 整体准确率: {overall_acc*100:.2f}%")
        print(f"  - 用时: {elapsed_time:.2f}秒")
        print("="*60)
        
        self.results = {
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'overall_accuracy': overall_acc,
            'fold_accuracies': fold_accuracies,
            'predictions': all_preds,
            'true_labels': all_true,
            'probabilities': all_probs,
            'elapsed_time': elapsed_time,
            'config': self.config
        }
        
        return self.results
    
    def save_results(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        result_file = os.path.join(output_dir, f'vae_hnf_v3_results_{timestamp}.json')
        with open(result_file, 'w') as f:
            json.dump({
                'mean_accuracy': self.results['mean_accuracy'],
                'std_accuracy': self.results['std_accuracy'],
                'overall_accuracy': self.results['overall_accuracy'],
                'elapsed_time': self.results['elapsed_time'],
                'config': self.results['config']
            }, f, indent=2)
        print(f"[INFO] 结果已保存: {result_file}")
        
        # 混淆矩阵
        cm = confusion_matrix(self.results['true_labels'], self.results['predictions'])
        cm_df = pd.DataFrame(cm, 
                            index=self.label_encoder.classes_, 
                            columns=self.label_encoder.classes_)
        cm_file = os.path.join(output_dir, f'confusion_matrix_v3_{timestamp}.csv')
        cm_df.to_csv(cm_file)
        
        # ROC数据
        roc_data = pd.DataFrame({
            'true_label': self.results['true_labels'],
            'probability': self.results['probabilities'],
            'prediction': self.results['predictions']
        })
        roc_file = os.path.join(output_dir, f'roc_data_v3_{timestamp}.csv')
        roc_data.to_csv(roc_file, index=False)
        
        return result_file


def main():
    print("="*60)
    print("VAE-HyperNetFusion V3: 优化版神经网络模型")
    print("="*60)
    
    config = {
        # VAE参数（增加数据量）
        'vae_latent_dim': 2,
        'vae_hidden_dim': 32,
        'vae_epochs': 100,
        'vae_lr': 0.001,
        'vae_kl_weight': 0.3,
        'num_interpolation_points': 10,  # 更多增强
        
        # 简化的网络（减少过拟合）
        'hidden_dim': 16,
        'num_ensemble': 5,
        'dropout': 0.3,
        
        # 训练参数
        'train_epochs': 200,
        'train_lr': 0.01,
        'weight_decay': 0.01,
        'batch_size': 16,
        'num_train_runs': 3,
        
        # 验证
        'leave_p_out': 2
    }
    
    trainer = VAEHyperNetFusionV3(config)
    
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'Data_for_Jinming.csv')
    X, y = trainer.load_data(data_path)
    
    results = trainer.train_and_evaluate(X, y)
    
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    trainer.save_results(output_dir)
    
    print("\n[完成] V3实验结束！")
    print(f"最终准确率: {results['mean_accuracy']*100:.2f}% ± {results['std_accuracy']*100:.2f}%")


if __name__ == '__main__':
    main()
