# data/data_loader.py
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from models.vae_pytorch import VAEDataAugmentationPyTorch
import torch
import sys


class TabularDataset(Dataset):
    def __init__(self, data, labels, scaler=None):
        self.labels = labels
        if scaler:
            self.scaler = scaler
            self.data = self.scaler.transform(data)
        else:
            self.scaler = StandardScaler()
            self.data = self.scaler.fit_transform(data)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


def get_data_loaders(csv_path, target_column, train_indices, val_indices, batch_size=32, augment=False):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()  # Remove spaces and convert to lowercase
    print(f"Processed column names: {df.columns.tolist()}")  # Print processed column names

    if target_column.lower() not in df.columns:
        print(f"Error: Target column '{target_column}' does not exist in the CSV file.")
        sys.exit(1)

    # Remove target column and 'sample id' column
    feature_columns = [col for col in df.columns if col not in [target_column.lower(), 'sample id']]
    X = df[feature_columns].values
    y = df[target_column.lower()].values

    # Use LabelEncoder to convert string labels to integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"Label encoding mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

    # Get training and validation data
    X_train, X_val = X[train_indices], X[val_indices]
    y_train, y_val = y_encoded[train_indices], y_encoded[val_indices]

    # Data augmentation (applied only to training data)
    if augment:
        augmenter = VAEDataAugmentationPyTorch(
            input_dim=X_train.shape[1],
            latent_dim=20,  # Adjust as needed
            hidden_dim=512,
            learning_rate=0.001,
            kl_weight=1.0,
            recon_weight=1.0,
            l1_reg=0.0,
            l2_reg=0.0,
            activation=nn.ReLU,  # Ensure that a class, not an instance, is passed
            early_stopping=True,
            patience=10,
            num_interpolation_points=5,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        augmented_X, augmented_y = augmenter.augment_data(X_train, y_train)
        X_train = np.vstack((X_train, augmented_X))
        y_train = np.concatenate((y_train, augmented_y))
        print(
            f"Original training data size: {len(y_train) - len(augmented_y)}, Augmented data size: {len(augmented_y)}, Total training data size: {len(y_train)}")

    # Fit scaler only on training data
    scaler = StandardScaler()
    scaler.fit(X_train)  # Fit scaler only on training data

    train_dataset = TabularDataset(X_train, y_train, scaler=scaler)
    val_dataset = TabularDataset(X_val, y_val, scaler=scaler)  # Use the same scaler

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, scaler, label_encoder
