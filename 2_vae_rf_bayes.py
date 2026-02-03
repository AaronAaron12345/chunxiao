# -*- coding: utf-8 -*-
"""
VAE-HyperNetFusion: Optimized Main Script (Version 2)
=====================================================
优化版本：改进的VAE增强 + 随机森林分类器（更稳定的方案）

基于你之前的成功实验，使用VAE数据增强 + 随机森林分类器
这个组合在你的论文中获得了86.15%的准确率
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import LeavePOut, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from tqdm import tqdm
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入VAE增强模块
from vae_augmentation import VAEDataAugmentation


class VAEHyperNetFusionV2:
    """
    VAE-HyperNetFusion V2 - 使用VAE增强 + 随机森林
    
    这是基于你之前成功实验的配置重现
    """
    
    def __init__(self, config=None):
        self.config = config or self._default_config()
        self.label_encoder = LabelEncoder()
        self.scaler = MinMaxScaler()
        self.best_model = None
        self.results = {}
    
    def _default_config(self):
        return {
            # VAE参数
            'vae_latent_dim': 2,
            'vae_hidden_dim': 64,
            'vae_epochs': 100,
            'vae_lr': 0.001,
            'vae_kl_weight': 0.5,
            'num_interpolation_points': 5,
            
            # 随机森林参数 (贝叶斯优化搜索)
            'rf_n_estimators_range': (100, 200),
            'rf_max_depth_range': (1, 10),
            'rf_bayes_n_iter': 50,
            
            # 验证
            'leave_p_out': 2
        }
    
    def load_prostate_data(self, filepath):
        """加载前列腺癌数据集"""
        print(f"[INFO] 加载数据: {filepath}")
        data = pd.read_csv(filepath)
        
        feature_cols = ['LAA', 'Glutamate', 'Choline', 'Sarcosine']
        X = data[feature_cols].values.astype(np.float32)
        y = data['Group'].values
        y = self.label_encoder.fit_transform(y)
        
        print(f"  - 样本数: {len(X)}, 特征数: {X.shape[1]}")
        print(f"  - 类别: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
        self.feature_names = feature_cols
        return X, y
    
    def augment_data(self, X_train, y_train):
        """使用VAE进行数据增强"""
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
        X_combined = np.vstack([X_train, X_aug])
        y_combined = np.concatenate([y_train, y_aug])
        
        return X_combined, y_combined
    
    def train_random_forest(self, X_train, y_train, optimize=True):
        """训练随机森林模型（可选贝叶斯优化）"""
        
        if optimize:
            # 贝叶斯优化
            rf = RandomForestClassifier(random_state=42, n_jobs=-1)
            param_space = {
                'n_estimators': Integer(*self.config['rf_n_estimators_range']),
                'max_depth': Integer(*self.config['rf_max_depth_range'])
            }
            
            opt = BayesSearchCV(
                rf, param_space, 
                n_iter=self.config['rf_bayes_n_iter'],
                cv=5, n_jobs=-1, random_state=42,
                verbose=0
            )
            opt.fit(X_train, y_train)
            model = opt.best_estimator_
            print(f"  - 最佳参数: {opt.best_params_}")
        else:
            model = RandomForestClassifier(
                n_estimators=150, max_depth=5,
                random_state=42, n_jobs=-1
            )
            model.fit(X_train, y_train)
        
        return model
    
    def run_leave_p_out(self, X, y):
        """Leave-P-Out 交叉验证"""
        p = self.config['leave_p_out']
        lpo = LeavePOut(p=p)
        n_splits = lpo.get_n_splits(X)
        
        print(f"\n[INFO] Leave-{p}-Out 交叉验证")
        print(f"  - 总splits: {n_splits}")
        
        all_preds = []
        all_true = []
        all_probs = []
        fold_accuracies = []
        
        start_time = time.time()
        
        for train_idx, test_idx in tqdm(lpo.split(X), total=n_splits, desc="Training"):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # 1. VAE数据增强
            X_train_aug, y_train_aug = self.augment_data(X_train, y_train)
            
            # 2. 标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_aug)
            X_test_scaled = scaler.transform(X_test)
            
            # 3. 训练随机森林 (不优化，用固定参数加快速度)
            model = self.train_random_forest(X_train_scaled, y_train_aug, optimize=False)
            
            # 4. 预测
            pred = model.predict(X_test_scaled)
            prob = model.predict_proba(X_test_scaled)
            
            all_preds.extend(pred.tolist())
            all_true.extend(y_test.tolist())
            if prob.shape[1] == 2:
                all_probs.extend(prob[:, 1].tolist())
            
            acc = accuracy_score(y_test, pred)
            fold_accuracies.append(acc)
        
        elapsed = time.time() - start_time
        
        # 计算统计
        mean_acc = np.mean(fold_accuracies)
        std_acc = np.std(fold_accuracies)
        overall_acc = accuracy_score(all_true, all_preds)
        
        print(f"\n{'='*60}")
        print(f"[结果] Leave-{p}-Out 完成")
        print(f"  - 平均准确率: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
        print(f"  - 整体准确率: {overall_acc*100:.2f}%")
        print(f"  - 用时: {elapsed:.2f}秒")
        print(f"{'='*60}")
        
        self.results = {
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'overall_accuracy': overall_acc,
            'fold_accuracies': fold_accuracies,
            'predictions': all_preds,
            'true_labels': all_true,
            'probabilities': all_probs,
            'elapsed_time': elapsed
        }
        
        return self.results
    
    def train_final_model(self, X, y, optimize=True):
        """
        训练最终模型（用全部数据增强后训练）
        用于后续的模型部署或进一步评估
        """
        print("\n[INFO] 训练最终模型...")
        
        # VAE增强
        X_aug, y_aug = self.augment_data(X, y)
        
        # 标准化
        self.scaler.fit(X_aug)
        X_scaled = self.scaler.transform(X_aug)
        
        # 训练
        self.best_model = self.train_random_forest(X_scaled, y_aug, optimize=optimize)
        
        # 在原始数据上评估
        X_orig_scaled = self.scaler.transform(X)
        pred = self.best_model.predict(X_orig_scaled)
        acc = accuracy_score(y, pred)
        
        print(f"  - 在原始数据上的准确率: {acc*100:.2f}%")
        
        return self.best_model, acc
    
    def save_results(self, output_dir):
        """保存结果"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存JSON结果
        result_file = os.path.join(output_dir, f'vae_rf_results_{timestamp}.json')
        save_data = {
            'mean_accuracy': self.results['mean_accuracy'],
            'std_accuracy': self.results['std_accuracy'],
            'overall_accuracy': self.results['overall_accuracy'],
            'elapsed_time': self.results['elapsed_time'],
            'config': self.config
        }
        with open(result_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"[INFO] 结果已保存: {result_file}")
        
        # 保存混淆矩阵
        cm = confusion_matrix(self.results['true_labels'], self.results['predictions'])
        cm_file = os.path.join(output_dir, f'confusion_matrix_{timestamp}.csv')
        pd.DataFrame(cm).to_csv(cm_file, index=False)
        
        # 保存ROC数据
        roc_file = os.path.join(output_dir, f'roc_data_{timestamp}.csv')
        pd.DataFrame({
            'true_label': self.results['true_labels'],
            'probability': self.results['probabilities'],
            'prediction': self.results['predictions']
        }).to_csv(roc_file, index=False)
        
        return result_file


def main():
    print("="*60)
    print("VAE-HyperNetFusion V2: VAE增强 + 随机森林")
    print("="*60)
    
    # 配置
    config = {
        'vae_latent_dim': 2,
        'vae_hidden_dim': 64,
        'vae_epochs': 50,  # 减少epoch加快速度
        'vae_lr': 0.001,
        'vae_kl_weight': 0.5,
        'num_interpolation_points': 5,
        'rf_n_estimators_range': (100, 200),
        'rf_max_depth_range': (1, 10),
        'rf_bayes_n_iter': 30,
        'leave_p_out': 2
    }
    
    # 初始化
    trainer = VAEHyperNetFusionV2(config)
    
    # 加载数据
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'Data_for_Jinming.csv')
    X, y = trainer.load_prostate_data(data_path)
    
    # Leave-P-Out 交叉验证
    results = trainer.run_leave_p_out(X, y)
    
    # 训练最终模型并用贝叶斯优化
    print("\n[INFO] 使用贝叶斯优化训练最终模型...")
    final_model, final_acc = trainer.train_final_model(X, y, optimize=True)
    
    # 保存结果
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    trainer.save_results(output_dir)
    
    print(f"\n{'='*60}")
    print(f"[完成] 实验结束！")
    print(f"  Leave-P-Out 准确率: {results['mean_accuracy']*100:.2f}% ± {results['std_accuracy']*100:.2f}%")
    print(f"  最终模型准确率: {final_acc*100:.2f}%")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
