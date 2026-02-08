#!/usr/bin/env python3
"""
Dataset loading utilities for 11 small tabular classification benchmarks.

Datasets 1-10 are from the UCI Machine Learning Repository.
Dataset 0 (Prostate Cancer) is a private clinical dataset included in
the ``data/`` directory.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


DATASET_INFO = {
    0:  ("prostate",   None),
    1:  ("balloons",   "1.balloons/adult+stretch.data"),
    2:  ("lenses",     "2lens/lenses.data"),
    3:  ("caesarian",  "3.caesarian+section+classification+dataset/caesarian.csv"),
    4:  ("iris",       "4.iris/iris.data"),
    5:  ("fertility",  "5.fertility/fertility_Diagnosis.txt"),
    6:  ("zoo",        "6.zoo/zoo.data"),
    7:  ("seeds",      "7.seeds/seeds_dataset.txt"),
    8:  ("haberman",   "8.haberman+s+survival/haberman.data"),
    9:  ("glass",      "9.glass+identification/glass.data"),
    10: ("yeast",      "10.yeast/yeast.data"),
}


def load_dataset(dataset_id: int, data_dir: str = "./data"):
    """Load a dataset by its integer ID.

    Args:
        dataset_id: Integer key in {0, 1, ..., 10}.
        data_dir:   Root directory containing dataset sub-folders.
                    For dataset 0, ``data_dir`` should contain
                    ``prostate.csv`` (or ``Data_for_Jinming.csv``).

    Returns:
        X:    Feature array of shape (n_samples, n_features), dtype float.
        y:    Label array of shape (n_samples,), integer-encoded.
        name: Dataset name string.
    """
    if dataset_id not in DATASET_INFO:
        raise ValueError(f"Unknown dataset ID: {dataset_id}. "
                         f"Valid IDs: {list(DATASET_INFO.keys())}")

    name, rel_path = DATASET_INFO[dataset_id]

    if dataset_id == 0:
        return _load_prostate(data_dir, name)

    file_path = os.path.join(data_dir, rel_path)
    if not Path(file_path).exists():
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}. "
            f"Please download from the UCI ML Repository.")

    loaders = {
        1:  _load_balloons,
        2:  _load_lenses,
        3:  _load_caesarian,
        4:  _load_iris,
        5:  _load_csv_last_col,
        6:  _load_zoo,
        7:  _load_whitespace_last_col,
        8:  _load_csv_last_col,
        9:  _load_glass,
        10: _load_yeast,
    }

    X, y = loaders[dataset_id](file_path)
    X, y = _encode(X, y)
    return X, y, name


# ---------------------------------------------------------------
# Per-dataset loaders
# ---------------------------------------------------------------

def _load_prostate(data_dir: str, name: str):
    """Load prostate cancer dataset."""
    candidates = [
        os.path.join(data_dir, "prostate.csv"),
        os.path.join(data_dir, "Data_for_Jinming.csv"),
    ]
    for path in candidates:
        if Path(path).exists():
            df = pd.read_csv(path)
            X = df.iloc[:, 2:].values.astype(float)
            y = LabelEncoder().fit_transform(df['Group'].values)
            return X, y, name
    raise FileNotFoundError(
        f"Prostate dataset not found in {data_dir}. "
        f"Expected 'prostate.csv' or 'Data_for_Jinming.csv'.")


def _load_balloons(path: str):
    df = pd.read_csv(path, header=None)
    return df.iloc[:, :-1].values, df.iloc[:, -1].values


def _load_lenses(path: str):
    df = pd.read_csv(path, sep=r'\s+', header=None, engine='python')
    return df.iloc[:, 1:-1].values, df.iloc[:, -1].values


def _load_caesarian(path: str):
    rows = []
    with open(path) as f:
        started = False
        for line in f:
            if '@data' in line.lower():
                started = True
                continue
            if started and line.strip() and not line.startswith('%'):
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    rows.append(parts)
    df = pd.DataFrame(rows)
    return df.iloc[:, :-1].values.astype(float), df.iloc[:, -1].values


def _load_iris(path: str):
    df = pd.read_csv(path, header=None).dropna(how='all')
    df = df[df.iloc[:, -1].astype(str).str.strip() != '']
    return df.iloc[:, :-1].values, df.iloc[:, -1].values


def _load_csv_last_col(path: str):
    df = pd.read_csv(path, header=None)
    return df.iloc[:, :-1].values, df.iloc[:, -1].values


def _load_zoo(path: str):
    df = pd.read_csv(path, header=None)
    return df.iloc[:, 1:-1].values, df.iloc[:, -1].values


def _load_whitespace_last_col(path: str):
    df = pd.read_csv(path, sep=r'\s+', header=None, engine='python')
    return df.iloc[:, :-1].values, df.iloc[:, -1].values


def _load_glass(path: str):
    df = pd.read_csv(path, header=None)
    return df.iloc[:, 1:-1].values, df.iloc[:, -1].values


def _load_yeast(path: str):
    df = pd.read_csv(path, sep=r'\s+', header=None, engine='python')
    return df.iloc[:, 1:-1].values, df.iloc[:, -1].values


# ---------------------------------------------------------------
# Encoding utility
# ---------------------------------------------------------------

def _encode(X, y):
    """Ensure X is float and y is integer-encoded."""
    for c in range(X.shape[1]):
        try:
            X[:, c] = X[:, c].astype(float)
        except (ValueError, TypeError):
            X[:, c] = LabelEncoder().fit_transform(X[:, c].astype(str))
    X = X.astype(float)
    y = LabelEncoder().fit_transform(y.astype(str))
    return X, y
