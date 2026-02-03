# -*- coding: utf-8 -*-
"""
VAE-HyperNetFusion: Main Training and Evaluation Script
=======================================================
完整的训练流程：VAE数据增强 + HyperNetFusion模型训练 + Leave-P-Out交叉验证

作者: Jinming Zhang
项目: 小型表格数据集的深度学习方法
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
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.metrics import precision_recall_curve, roc_curve
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import json
import time
from datetime import datetime

from vae_augmentation import VAEDataAugmentation
from hypernet_fusion import HyperNetFusion


class VAEHyperNetFusionTrainer:
    """
    VAE-HyperNetFusion 训练器
    
    工作流程：
    1. 加载原始数据
    2. 使用VAE进行数据增强
    3. 用增强数据训练HyperNetFusion模型
    4. 用原始数据进行验证（Leave-P-Out）
    """
    
    def __init__(self, config=None):
        self.config = config or self._default_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] 使用设备: {self.device}")
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # 结果存储
        self.results = {
            'accuracies': [],
            'predictions': [],
            'true_labels': [],
            'probabilities': []
        }
    
    def _default_config(self):
        """默认配置"""
        return {
            # VAE增强参数
            'vae_latent_dim': 2,
            'vae_hidden_dim': 64,
            'vae_epochs': 100,
            'vae_lr': 0.001,
            'vae_kl_weight': 0.5,
            'num_interpolation_points': 5,
            
            # HyperNetFusion参数
            'hypernet_hidden_dim': 128,
            'target_hidden_dim': 32,
            'num_target_nets': 10,
            
            # 训练参数
            'train_epochs': 100,
            'train_lr': 0.001,
            'batch_size': 32,
            
            # 验证参数
            'leave_p_out': 2,
            'max_workers': 4  # 并行进程数
        }
    
    def load_data(self, filepath):
        """加载前列腺癌数据集"""
        print(f"[INFO] 加载数据: {filepath}")
        
        data = pd.read_csv(filepath)
        
        # 特征列
        feature_cols = ['LAA', 'Glutamate', 'Choline', 'Sarcosine']
        X = data[feature_cols].values.astype(np.float32)
        
        # 标签列
        y = data['Group'].values
        y = self.label_encoder.fit_transform(y)
        
        print(f"  - 样本数: {len(X)}")
        print(f"  - 特征数: {X.shape[1]}")
        print(f"  - 类别: {self.label_encoder.classes_}")
        print(f"  - 类别分布: {np.bincount(y)}")
        
        self.feature_names = feature_cols
        return X, y
    
    def _train_single_fold(self, train_idx, test_idx, X, y, fold_id):
        """训练单个fold（用于并行计算）"""
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Step 1: VAE数据增强（只对训练集）
        augmenter = VAEDataAugmentation(
            latent_dim=self.config['vae_latent_dim'],
            hidden_dim=self.config['vae_hidden_dim'],
            epochs=self.config['vae_epochs'],
            learning_rate=self.config['vae_lr'],
            kl_weight=self.config['vae_kl_weight'],
            num_interpolation_points=self.config['num_interpolation_points']
        )
        
        X_aug, y_aug = augmenter.augment(X_train, y_train)
        
        # 合并原始训练数据和增强数据
        X_train_full = np.vstack([X_train, X_aug])
        y_train_full = np.concatenate([y_train, y_aug])
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_full)
        X_test_scaled = scaler.transform(X_test)
        
        # 转为PyTorch张量
        X_train_t = torch.FloatTensor(X_train_scaled).to(self.device)
        y_train_t = torch.LongTensor(y_train_full).to(self.device)
        X_test_t = torch.FloatTensor(X_test_scaled).to(self.device)
        
        # Step 2: 训练HyperNetFusion模型
        model = HyperNetFusion(
            input_dim=X.shape[1],
            hypernet_hidden_dim=self.config['hypernet_hidden_dim'],
            target_hidden_dim=self.config['target_hidden_dim'],
            num_classes=len(np.unique(y)),
            num_target_nets=self.config['num_target_nets']
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config['train_lr'])
        
        # 创建DataLoader
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        
        # 训练循环
        model.train()
        for epoch in range(self.config['train_epochs']):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # Step 3: 评估
        model.eval()
        with torch.no_grad():
            proba = model.predict_proba(X_test_t).cpu().numpy()
            pred = np.argmax(proba, axis=1)
        
        accuracy = accuracy_score(y_test, pred)
        
        return {
            'fold_id': fold_id,
            'accuracy': accuracy,
            'predictions': pred.tolist(),
            'true_labels': y_test.tolist(),
            'probabilities': proba[:, 1].tolist() if proba.shape[1] == 2 else proba.tolist()
        }
    
    def train_and_evaluate(self, X, y, use_parallel=True):
        """
        使用Leave-P-Out交叉验证进行训练和评估
        """
        p = self.config['leave_p_out']
        lpo = LeavePOut(p=p)
        n_splits = lpo.get_n_splits(X)
        
        print(f"\n[INFO] 开始Leave-{p}-Out交叉验证")
        print(f"  - 总splits数: {n_splits}")
        print(f"  - 并行处理: {use_parallel}")
        
        start_time = time.time()
        
        all_results = []
        
        if use_parallel and n_splits > 10:
            # 并行处理
            print(f"  - 使用 {self.config['max_workers']} 个进程并行处理")
            
            # 注意：由于CUDA在多进程中的限制，这里改用顺序处理但显示进度
            for fold_id, (train_idx, test_idx) in enumerate(tqdm(lpo.split(X), total=n_splits, desc="Training")):
                result = self._train_single_fold(train_idx, test_idx, X, y, fold_id)
                all_results.append(result)
        else:
            # 顺序处理（小数据集或调试）
            for fold_id, (train_idx, test_idx) in enumerate(tqdm(lpo.split(X), total=n_splits, desc="Training")):
                result = self._train_single_fold(train_idx, test_idx, X, y, fold_id)
                all_results.append(result)
        
        elapsed_time = time.time() - start_time
        
        # 汇总结果
        accuracies = [r['accuracy'] for r in all_results]
        all_preds = []
        all_true = []
        all_probs = []
        
        for r in all_results:
            all_preds.extend(r['predictions'])
            all_true.extend(r['true_labels'])
            all_probs.extend(r['probabilities'])
        
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        
        # 计算整体指标
        overall_acc = accuracy_score(all_true, all_preds)
        
        print(f"\n" + "="*60)
        print(f"[结果] Leave-{p}-Out 交叉验证完成")
        print(f"  - 平均准确率: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
        print(f"  - 整体准确率: {overall_acc*100:.2f}%")
        print(f"  - 用时: {elapsed_time:.2f}秒")
        print("="*60)
        
        # 存储结果
        self.results = {
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'overall_accuracy': overall_acc,
            'all_accuracies': accuracies,
            'predictions': all_preds,
            'true_labels': all_true,
            'probabilities': all_probs,
            'elapsed_time': elapsed_time,
            'n_splits': n_splits
        }
        
        return self.results
    
    def save_results(self, output_dir):
        """保存实验结果"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存主要结果
        result_file = os.path.join(output_dir, f'results_{timestamp}.json')
        with open(result_file, 'w') as f:
            json.dump({
                'mean_accuracy': self.results['mean_accuracy'],
                'std_accuracy': self.results['std_accuracy'],
                'overall_accuracy': self.results['overall_accuracy'],
                'elapsed_time': self.results['elapsed_time'],
                'n_splits': self.results['n_splits'],
                'config': self.config
            }, f, indent=2)
        print(f"[INFO] 结果已保存到: {result_file}")
        
        # 保存ROC数据
        roc_data = pd.DataFrame({
            'true_label': self.results['true_labels'],
            'probability': self.results['probabilities'],
            'prediction': self.results['predictions']
        })
        roc_file = os.path.join(output_dir, f'roc_data_{timestamp}.csv')
        roc_data.to_csv(roc_file, index=False)
        print(f"[INFO] ROC数据已保存到: {roc_file}")
        
        # 保存混淆矩阵
        cm = confusion_matrix(self.results['true_labels'], self.results['predictions'])
        cm_df = pd.DataFrame(cm, 
                            index=self.label_encoder.classes_, 
                            columns=self.label_encoder.classes_)
        cm_file = os.path.join(output_dir, f'confusion_matrix_{timestamp}.csv')
        cm_df.to_csv(cm_file)
        print(f"[INFO] 混淆矩阵已保存到: {cm_file}")
        
        return result_file


def main():
    """主函数"""
    print("="*60)
    print("VAE-HyperNetFusion: 小型表格数据集深度学习框架")
    print("="*60)
    
    # 配置
    config = {
        # VAE增强参数
        'vae_latent_dim': 2,
        'vae_hidden_dim': 64,
        'vae_epochs': 50,
        'vae_lr': 0.001,
        'vae_kl_weight': 0.5,
        'num_interpolation_points': 5,
        
        # HyperNetFusion参数
        'hypernet_hidden_dim': 64,
        'target_hidden_dim': 32,
        'num_target_nets': 10,
        
        # 训练参数
        'train_epochs': 50,
        'train_lr': 0.001,
        'batch_size': 32,
        
        # 验证参数
        'leave_p_out': 2,
        'max_workers': 4
    }
    
    # 初始化训练器
    trainer = VAEHyperNetFusionTrainer(config)
    
    # 加载数据
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'Data_for_Jinming.csv')
    X, y = trainer.load_data(data_path)
    
    # 训练和评估
    results = trainer.train_and_evaluate(X, y, use_parallel=False)
    
    # 保存结果
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    trainer.save_results(output_dir)
    
    print("\n[完成] 实验结束！")
    print(f"最终准确率: {results['mean_accuracy']*100:.2f}% ± {results['std_accuracy']*100:.2f}%")


if __name__ == '__main__':
    main()
