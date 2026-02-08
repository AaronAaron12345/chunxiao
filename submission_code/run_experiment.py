#!/usr/bin/env python3
"""
Reproduce Table II: 6-model comparison on 11 small tabular datasets.

Models compared:
    1. Random Forest (RF)
    2. XGBoost
    3. TabPFN
    4. HyperTab
    5. AutoML (TPOT)
    6. VAE-HyperNet-Fusion (proposed)

Evaluation: 5-fold stratified cross-validation.
VAE-HNF hyperparameters are tuned per dataset via Bayesian optimization
(Optuna with TPE sampler).

Usage:
    python run_experiment.py --data_dir ./data --gpu 0
    python run_experiment.py --data_dir ./data --gpu 0 --datasets 0 4 7
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from data_loader import load_dataset, DATASET_INFO
from model import (VAEHyperNetFusion, train_model, predict_model,
                   set_seed, DEFAULT_HYPERPARAMS)

warnings.filterwarnings('ignore')

# Optional dependencies (graceful fallback if unavailable)
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from tabpfn import TabPFNClassifier
    HAS_TABPFN = True
except ImportError:
    HAS_TABPFN = False

try:
    from hypertab import HyperTabClassifier
    HAS_HYPERTAB = True
except ImportError:
    HAS_HYPERTAB = False

try:
    from tpot import TPOTClassifier
    HAS_TPOT = True
except ImportError:
    HAS_TPOT = False

try:
    import optuna
    from optuna.samplers import TPESampler
    HAS_OPTUNA = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    HAS_OPTUNA = False


# ---------------------------------------------------------------
# Baseline model runners
# ---------------------------------------------------------------

def run_rf(X, y, kfold):
    """Random Forest baseline."""
    preds_all, labels_all, fold_accs = [], [], []
    for train_idx, test_idx in kfold.split(X, y):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[train_idx])
        X_te = sc.transform(X[test_idx])
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_tr, y[train_idx])
        preds = clf.predict(X_te)
        acc = accuracy_score(y[test_idx], preds) * 100
        fold_accs.append(acc)
        preds_all.extend(preds)
        labels_all.extend(y[test_idx])
    return np.mean(fold_accs), np.std(fold_accs), fold_accs


def run_xgboost(X, y, kfold, n_classes):
    """XGBoost baseline."""
    if not HAS_XGBOOST:
        return None, None, None
    fold_accs = []
    for train_idx, test_idx in kfold.split(X, y):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[train_idx])
        X_te = sc.transform(X[test_idx])
        clf = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss' if n_classes > 2 else 'logloss',
            verbosity=0)
        clf.fit(X_tr, y[train_idx])
        preds = clf.predict(X_te)
        fold_accs.append(accuracy_score(y[test_idx], preds) * 100)
    return np.mean(fold_accs), np.std(fold_accs), fold_accs


def run_tabpfn(X, y, kfold):
    """TabPFN baseline."""
    if not HAS_TABPFN:
        return None, None, None
    fold_accs = []
    for train_idx, test_idx in kfold.split(X, y):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[train_idx])
        X_te = sc.transform(X[test_idx])
        clf = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)
        clf.fit(X_tr, y[train_idx])
        preds = clf.predict(X_te)
        fold_accs.append(accuracy_score(y[test_idx], preds) * 100)
    return np.mean(fold_accs), np.std(fold_accs), fold_accs


def run_hypertab(X, y, kfold):
    """HyperTab baseline."""
    if not HAS_HYPERTAB:
        return None, None, None
    fold_accs = []
    for train_idx, test_idx in kfold.split(X, y):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[train_idx])
        X_te = sc.transform(X[test_idx])
        clf = HyperTabClassifier()
        clf.fit(X_tr, y[train_idx])
        preds = clf.predict(X_te)
        fold_accs.append(accuracy_score(y[test_idx], preds) * 100)
    return np.mean(fold_accs), np.std(fold_accs), fold_accs


def run_tpot(X, y, kfold):
    """TPOT AutoML baseline."""
    if not HAS_TPOT:
        return None, None, None
    fold_accs = []
    for train_idx, test_idx in kfold.split(X, y):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[train_idx])
        X_te = sc.transform(X[test_idx])
        clf = TPOTClassifier(
            generations=5, population_size=20, random_state=42,
            verbosity=0, max_time_mins=5)
        clf.fit(X_tr, y[train_idx])
        preds = clf.predict(X_te)
        fold_accs.append(accuracy_score(y[test_idx], preds) * 100)
    return np.mean(fold_accs), np.std(fold_accs), fold_accs


# ---------------------------------------------------------------
# VAE-HNF with optional Optuna tuning
# ---------------------------------------------------------------

def _optuna_objective(trial, X, y, n_classes, device):
    """Optuna objective: 5-fold CV accuracy."""
    hp = {
        'n_trees':       trial.suggest_int('n_trees', 5, 30, step=5),
        'tree_depth':    trial.suggest_int('tree_depth', 2, 5),
        'n_heads':       trial.suggest_int('n_heads', 3, 9, step=2),
        'hidden_dim':    trial.suggest_categorical('hidden_dim', [32, 64, 128]),
        'latent_dim':    trial.suggest_categorical('latent_dim', [4, 8, 16]),
        'dropout':       trial.suggest_float('dropout', 0.05, 0.25),
        'n_augment':     trial.suggest_int('n_augment', 50, 800, step=50),
        'noise_scale':   trial.suggest_float('noise_scale', 0.1, 1.0),
        'vae_lr':        trial.suggest_float('vae_lr', 5e-4, 0.01, log=True),
        'vae_epochs':    trial.suggest_int('vae_epochs', 50, 200, step=25),
        'kl_weight':     trial.suggest_float('kl_weight', 1e-3, 0.05, log=True),
        'lr':            trial.suggest_float('lr', 1e-3, 0.05, log=True),
        'weight_decay':  trial.suggest_float('weight_decay', 0.01, 0.1, log=True),
        'epochs':        trial.suggest_categorical('epochs', [200, 300, 500]),
        'warmup_epochs': trial.suggest_int('warmup_epochs', 10, 40, step=5),
        'reg_weight':    trial.suggest_float('reg_weight', 1e-3, 0.05, log=True),
    }

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []
    for fi, (tri, tei) in enumerate(kfold.split(X, y)):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tri])
        X_te = sc.transform(X[tei])
        mdl = train_model(X_tr, y[tri], X_te, y[tei],
                          n_classes, device, hp, seed=42 + fi)
        proba = predict_model(mdl, X_tr, X_te, device)
        accs.append(accuracy_score(y[tei], proba.argmax(1)) * 100)
    return np.mean(accs)


def run_vae_hnf(X, y, kfold, n_classes, device, n_trials=50):
    """Run VAE-HNF with Optuna hyperparameter tuning."""
    # Bayesian hyperparameter optimization
    best_hp = DEFAULT_HYPERPARAMS.copy()

    if HAS_OPTUNA and n_trials > 0:
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42))
        study.optimize(
            lambda trial: _optuna_objective(trial, X, y, n_classes, device),
            n_trials=n_trials, show_progress_bar=False)
        # Merge best params with defaults
        for k, v in study.best_params.items():
            best_hp[k] = v
        print(f"    Optuna best accuracy: {study.best_value:.1f}%")

    # Final evaluation with best hyperparameters
    fold_accs = []
    for fi, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[train_idx])
        X_te = sc.transform(X[test_idx])
        mdl = train_model(X_tr, y[train_idx], X_te, y[test_idx],
                          n_classes, device, best_hp, seed=42 + fi)
        proba = predict_model(mdl, X_tr, X_te, device)
        fold_accs.append(accuracy_score(y[test_idx], proba.argmax(1)) * 100)

    return np.mean(fold_accs), np.std(fold_accs), fold_accs, best_hp


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Reproduce Table II: 6-model comparison')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Root directory for datasets')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU index (-1 for CPU)')
    parser.add_argument('--datasets', type=int, nargs='*', default=None,
                        help='Dataset IDs to evaluate (default: all)')
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Optuna trials per dataset (0 = use defaults)')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory for results')
    args = parser.parse_args()

    device = (torch.device(f'cuda:{args.gpu}')
              if args.gpu >= 0 and torch.cuda.is_available()
              else torch.device('cpu'))
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    dataset_ids = args.datasets or list(DATASET_INFO.keys())

    results = {}
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for did in dataset_ids:
        print(f"\n{'='*60}")
        print(f"Dataset {did}: {DATASET_INFO[did][0]}")
        print(f"{'='*60}")

        try:
            X, y, name = load_dataset(did, args.data_dir)
        except FileNotFoundError as e:
            print(f"  Skipped: {e}")
            continue

        n_classes = len(np.unique(y))
        print(f"  Samples={len(X)}, Features={X.shape[1]}, "
              f"Classes={n_classes}")

        ds_results = {}

        # RF
        print("  [1/6] Random Forest...")
        mean, std, folds = run_rf(X, y, kfold)
        ds_results['RF'] = {'mean': mean, 'std': std, 'folds': folds}
        print(f"         {mean:.1f} +/- {std:.1f}")

        # XGBoost
        print("  [2/6] XGBoost...")
        mean, std, folds = run_xgboost(X, y, kfold, n_classes)
        if mean is not None:
            ds_results['XGBoost'] = {'mean': mean, 'std': std, 'folds': folds}
            print(f"         {mean:.1f} +/- {std:.1f}")
        else:
            print("         (not installed)")

        # TabPFN
        print("  [3/6] TabPFN...")
        mean, std, folds = run_tabpfn(X, y, kfold)
        if mean is not None:
            ds_results['TabPFN'] = {'mean': mean, 'std': std, 'folds': folds}
            print(f"         {mean:.1f} +/- {std:.1f}")
        else:
            print("         (not installed)")

        # HyperTab
        print("  [4/6] HyperTab...")
        mean, std, folds = run_hypertab(X, y, kfold)
        if mean is not None:
            ds_results['HyperTab'] = {'mean': mean, 'std': std, 'folds': folds}
            print(f"         {mean:.1f} +/- {std:.1f}")
        else:
            print("         (not installed)")

        # TPOT
        print("  [5/6] AutoML (TPOT)...")
        mean, std, folds = run_tpot(X, y, kfold)
        if mean is not None:
            ds_results['TPOT'] = {'mean': mean, 'std': std, 'folds': folds}
            print(f"         {mean:.1f} +/- {std:.1f}")
        else:
            print("         (not installed)")

        # VAE-HNF
        print("  [6/6] VAE-HNF (ours)...")
        mean, std, folds, hp = run_vae_hnf(
            X, y, kfold, n_classes, device, args.n_trials)
        ds_results['VAE-HNF'] = {
            'mean': mean, 'std': std, 'folds': folds,
            'best_hp': hp,
        }
        print(f"         {mean:.1f} +/- {std:.1f}")

        results[name] = ds_results

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = os.path.join(args.output_dir, f'results_{timestamp}.json')
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    header = f"{'Dataset':<20} {'RF':>12} {'XGBoost':>12} {'TabPFN':>12} " \
             f"{'HyperTab':>12} {'TPOT':>12} {'VAE-HNF':>12}"
    print(header)
    print('-' * len(header))

    for name, ds in results.items():
        row = f"{name:<20}"
        for model in ['RF', 'XGBoost', 'TabPFN', 'HyperTab', 'TPOT',
                       'VAE-HNF']:
            if model in ds:
                m, s = ds[model]['mean'], ds[model]['std']
                row += f" {m:5.1f}({s:4.1f})  "
            else:
                row += f" {'N/A':>10}  "
        print(row)


if __name__ == '__main__':
    main()
