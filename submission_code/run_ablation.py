#!/usr/bin/env python3
"""
Reproduce Table III: Ablation study for VAE-HyperNet-Fusion.

Evaluates the contribution of each component by systematically removing
or replacing them:

    A: Full model (VAE-HNF)               — all components active
    B: No VAE augmentation                 — remove VAE, train on raw data
    C: No interpolation (noise_scale=0)    — VAE copies, no latent noise
    D: No hypernetwork                     — directly trained tree ensemble
    E: Single head (n_heads=1)             — remove multi-head ensembling

Usage:
    python run_ablation.py --data_dir ./data --gpu 0
"""

import argparse
import json
import os
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from data_loader import load_dataset, DATASET_INFO
from model import (VAEHyperNetFusion, train_model, predict_model,
                   set_seed, DEFAULT_HYPERPARAMS)

warnings.filterwarnings('ignore')


def run_ablation_variant(X, y, kfold, n_classes, device,
                         hp, variant='full'):
    """Run a single ablation variant.

    Args:
        variant: One of 'full', 'no_vae', 'no_interp', 'no_hypernet',
                 'single_head'.
    """
    mod_hp = hp.copy()

    if variant == 'no_vae':
        mod_hp['n_augment'] = 0
    elif variant == 'no_interp':
        mod_hp['noise_scale'] = 0.0
    elif variant == 'single_head':
        mod_hp['n_heads'] = 1

    fold_accs = []
    for fi, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[train_idx])
        X_te = sc.transform(X[test_idx])

        if variant == 'no_hypernet':
            # Replace with Random Forest (no neural network)
            clf = RandomForestClassifier(
                n_estimators=mod_hp.get('n_trees', 15), random_state=42)
            clf.fit(X_tr, y[train_idx])
            preds = clf.predict(X_te)
            fold_accs.append(accuracy_score(y[test_idx], preds) * 100)
        else:
            mdl = train_model(X_tr, y[train_idx], X_te, y[test_idx],
                              n_classes, device, mod_hp, seed=42 + fi)
            proba = predict_model(mdl, X_tr, X_te, device)
            fold_accs.append(
                accuracy_score(y[test_idx], proba.argmax(1)) * 100)

    return np.mean(fold_accs), np.std(fold_accs), fold_accs


VARIANTS = [
    ('full',         'A: Full model'),
    ('no_vae',       'B: w/o VAE augmentation'),
    ('no_interp',    'C: w/o interpolation'),
    ('no_hypernet',  'D: w/o hypernetwork'),
    ('single_head',  'E: Single head (n_heads=1)'),
]


def main():
    parser = argparse.ArgumentParser(
        description='Reproduce Table III: Ablation study')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--datasets', type=int, nargs='*', default=None)
    parser.add_argument('--output_dir', type=str, default='./output')
    args = parser.parse_args()

    device = (torch.device(f'cuda:{args.gpu}')
              if args.gpu >= 0 and torch.cuda.is_available()
              else torch.device('cpu'))
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    dataset_ids = args.datasets or list(DATASET_INFO.keys())
    hp = DEFAULT_HYPERPARAMS.copy()

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

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
        for variant_key, variant_name in VARIANTS:
            print(f"  {variant_name}...", end=' ', flush=True)
            mean, std, folds = run_ablation_variant(
                X, y, kfold, n_classes, device, hp, variant_key)
            ds_results[variant_key] = {
                'name': variant_name,
                'mean': mean,
                'std': std,
                'folds': folds,
            }
            print(f"{mean:.1f} +/- {std:.1f}")

        results[name] = ds_results

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = os.path.join(
        args.output_dir, f'ablation_{timestamp}.json')
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")

    # Summary table
    print(f"\n{'='*80}")
    print("ABLATION SUMMARY")
    print(f"{'='*80}")

    header = f"{'Dataset':<20}"
    for _, vname in VARIANTS:
        header += f" {vname.split(':')[0]:>8}"
    print(header)
    print('-' * len(header))

    for name, ds in results.items():
        row = f"{name:<20}"
        for vkey, _ in VARIANTS:
            if vkey in ds:
                row += f" {ds[vkey]['mean']:7.1f} "
            else:
                row += f" {'N/A':>7} "
        print(row)


if __name__ == '__main__':
    main()
