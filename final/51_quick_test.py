#!/usr/bin/env python3
"""
51_quick_test.py
快速测试 - 只测试prostate数据集确认模型工作
"""

import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


def log(msg):
    print(msg)
    sys.stdout.flush()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SimpleVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(16, latent_dim)
        self.fc_var = nn.Linear(16, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
        )
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return self.decode(z), mu, logvar


class SimpleHyperNet(nn.Module):
    """简化版HyperNet - 更快训练"""
    def __init__(self, input_dim, n_classes=2, n_trees=5):
        super().__init__()
        self.n_trees = n_trees
        self.n_classes = n_classes
        
        # 数据编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )
        
        # 简单的分类器生成器
        classifier_size = n_trees * (input_dim * 8 + 8 * n_classes + n_classes)
        self.generator = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, classifier_size)
        )
        
        self._init()
    
    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, X_support, X_query):
        # 编码支持集
        enc = self.encoder(X_support)
        ctx = enc.mean(dim=0, keepdim=True)
        
        # 生成分类器参数
        params = self.generator(ctx)
        params = torch.tanh(params)
        
        # 解析参数并分类
        input_dim = X_support.shape[1]
        tree_outputs = []
        idx = 0
        
        for t in range(self.n_trees):
            # 两层MLP
            w1 = params[0, idx:idx+input_dim*8].view(input_dim, 8)
            idx += input_dim * 8
            w2 = params[0, idx:idx+8*self.n_classes].view(8, self.n_classes)
            idx += 8 * self.n_classes
            bias = params[0, idx:idx+self.n_classes]
            idx += self.n_classes
            
            h = torch.relu(X_query @ w1)
            out = h @ w2 + bias
            tree_outputs.append(out)
        
        # 平均集成
        return torch.stack(tree_outputs).mean(dim=0)


def train_and_predict(X_train, y_train, X_test, n_classes, device, seed=42):
    set_seed(seed)
    input_dim = X_train.shape[1]
    
    # 训练VAE增强
    vae = SimpleVAE(input_dim).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=0.01)
    
    X_t = torch.FloatTensor(X_train).to(device)
    
    vae.train()
    for _ in range(50):
        opt.zero_grad()
        recon, mu, logvar = vae(X_t)
        loss = F.mse_loss(recon, X_t) + 0.1 * torch.mean(-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()))
        loss.backward()
        opt.step()
    
    # 增强数据
    vae.eval()
    with torch.no_grad():
        mu, logvar = vae.encode(X_t)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        X_aug = vae.decode(z).cpu().numpy()
    
    X_all = np.vstack([X_train, X_aug])
    y_all = np.concatenate([y_train, y_train])
    
    # 训练HyperNet
    model = SimpleHyperNet(input_dim, n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    
    X_all_t = torch.FloatTensor(X_all).to(device)
    y_all_t = torch.LongTensor(y_all).to(device)
    
    model.train()
    for _ in range(100):
        opt.zero_grad()
        out = model(X_all_t, X_all_t)
        loss = F.cross_entropy(out, y_all_t)
        loss.backward()
        opt.step()
    
    # 预测
    model.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test).to(device)
        out = model(X_all_t, X_test_t)
        return out.argmax(dim=1).cpu().numpy()


def main():
    log("=" * 60)
    log("51_quick_test.py - 快速验证模型")
    log("=" * 60)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    log(f"Device: {device}")
    
    # 加载prostate数据
    df = pd.read_csv('/data2/image_identification/src/data/Data_for_Jinming.csv')
    X = df[['LAA', 'Glutamate', 'Choline', 'Sarcosine']].values
    y = (df['Group'] == 'PCa').astype(int).values
    n_classes = 2
    
    log(f"Data: {len(X)} samples, {X.shape[1]} features")
    
    seeds = [42, 123, 456]
    rf_accs = []
    vhn_accs = []
    
    for seed in seeds:
        set_seed(seed)
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        
        rf_p, rf_l = [], []
        vhn_p, vhn_l = [], []
        
        for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            
            # RF
            rf = RandomForestClassifier(n_estimators=100, random_state=seed)
            rf.fit(X_train_s, y_train)
            rf_p.extend(rf.predict(X_test_s))
            rf_l.extend(y_test)
            
            # VAE-HyperNet
            try:
                pred = train_and_predict(X_train_s, y_train, X_test_s, n_classes, device, seed+fold*100)
                vhn_p.extend(pred)
                vhn_l.extend(y_test)
            except Exception as e:
                log(f"Error fold {fold}: {e}")
                vhn_p.extend([0] * len(y_test))
                vhn_l.extend(y_test)
        
        rf_acc = accuracy_score(rf_l, rf_p) * 100
        vhn_acc = accuracy_score(vhn_l, vhn_p) * 100
        
        log(f"Seed {seed}: RF={rf_acc:.1f}%, VHN={vhn_acc:.1f}%")
        rf_accs.append(rf_acc)
        vhn_accs.append(vhn_acc)
    
    log("\n" + "=" * 60)
    log("汇总:")
    log(f"RF: {np.mean(rf_accs):.1f}% ± {np.std(rf_accs):.1f}%")
    log(f"VAE-HyperNet: {np.mean(vhn_accs):.1f}% ± {np.std(vhn_accs):.1f}%")
    log("=" * 60)


if __name__ == '__main__':
    main()
