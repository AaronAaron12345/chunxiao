# -*- coding: utf-8 -*-
# cross_validator.py
import sys

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
import numpy as np
from data.data_loader import get_data_loaders
from models.hypernet_fusion import HyperNetFusion
from utils.utils import save_model, calculate_accuracy  # Ensure these functions are properly defined

class CrossValidator:
    def __init__(self, csv_path, target_column, num_folds=5, augment=False, batch_size=32, num_epochs=100, learning_rate=1e-3):
        self.csv_path = csv_path
        self.target_column = target_column
        self.num_folds = num_folds
        self.augment = augment
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        # Hyperparameter settings
        self.hypernet_input_dim = 4  # Adjust based on actual input data
        self.hypernet_hidden_dim = 128
        self.target_net_input_dim = 4  # LAA, Glutamate, Choline, Sarcosine
        self.target_net_hidden_dim = 64
        self.target_net_output_dim = 2  # 'non-PCa' and 'PCa'
        self.num_target_nets = 10

        self.results = []

    def run(self):
        # Read the entire dataset
        df = pd.read_csv(self.csv_path)
        df.columns = df.columns.str.strip().str.lower()  # Remove spaces and convert to lowercase
        print(f"Processed column names: {df.columns.tolist()}")  # Print processed column names

        if self.target_column.lower() not in df.columns:
            print(f"Error: Target column '{self.target_column}' not found in CSV file.")
            sys.exit(1)

        # Remove the target column and 'sample id' column
        feature_columns = [col for col in df.columns if col not in [self.target_column.lower(), 'sample id']]
        X = df[feature_columns].values
        y = df[self.target_column.lower()].values

        # Use LabelEncoder to convert string labels to integers
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        print(f"Label encoding mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

        # Initialize StratifiedKFold
        skf = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=42)

        # Record the performance of each fold
        self.results = []

        for fold, (train_indices, val_indices) in enumerate(skf.split(X, y_encoded), 1):
            print(f"\n--- Fold {fold}/{self.num_folds} ---")
            # Get the training and validation data loaders for the current fold
            train_loader, val_loader, scaler, _ = get_data_loaders(
                self.csv_path, self.target_column, train_indices, val_indices, batch_size=self.batch_size, augment=self.augment
            )
            print(f"Fold {fold}: Training size: {len(train_loader.dataset)}, Validation size: {len(val_loader.dataset)}")

            # Model initialization
            print("Initializing model...")
            model = HyperNetFusion(
                hypernet_input_dim=self.hypernet_input_dim,
                hypernet_hidden_dim=self.hypernet_hidden_dim,
                target_net_input_dim=self.target_net_input_dim,
                target_net_hidden_dim=self.target_net_hidden_dim,
                target_net_output_dim=self.target_net_output_dim,
                num_target_nets=self.num_target_nets
            )
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            print(f"Model initialized, using device: {device}")

            # Loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

            # Training loop
            for epoch in range(self.num_epochs):
                model.train()
                running_loss = 0.0
                running_corrects = 0
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)

                    # Get the latent vector from VAE to use as hypernet_input
                    hypernet_input = batch_x  # [batch_size, hypernet_input_dim]

                    optimizer.zero_grad()
                    outputs = model(batch_x, hypernet_input)  # [batch_size, output_dim]
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * batch_x.size(0)
                    running_corrects += torch.sum(outputs.argmax(1) == batch_y)

                epoch_loss = running_loss / len(train_loader.dataset)
                epoch_acc = running_corrects.double() / len(train_loader.dataset)
                if epoch % 10 == 0 or epoch == self.num_epochs - 1:
                    print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_corrects = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    hypernet_input = batch_x  # [batch_size, hypernet_input_dim]

                    outputs = model(batch_x, hypernet_input)
                    loss = criterion(outputs, batch_y)

                    val_loss += loss.item() * batch_x.size(0)
                    val_corrects += torch.sum(outputs.argmax(1) == batch_y)

            val_epoch_loss = val_loss / len(val_loader.dataset)
            val_epoch_acc = val_corrects.double() / len(val_loader.dataset)
            print(f'Fold {fold} Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_acc:.4f}')

            # Record the results of the current fold
            self.results.append({
                'fold': fold,
                'validation_loss': val_epoch_loss,
                'validation_accuracy': val_epoch_acc.item()
            })

            # Optional: Save the model for each fold
            # save_model(model, f'model_fold_{fold}.pth')
            # print(f"Model for Fold {fold} has been saved to model_fold_{fold}.pth")

        # Summarize the results from all folds
        avg_val_loss = np.mean([result['validation_loss'] for result in self.results])
        avg_val_acc = np.mean([result['validation_accuracy'] for result in self.results])
        std_val_loss = np.std([result['validation_loss'] for result in self.results])
        std_val_acc = np.std([result['validation_accuracy'] for result in self.results])

        print("\n=== Cross-validation Results ===")
        for result in self.results:
            print(f"Fold {result['fold']}: Validation Loss: {result['validation_loss']:.4f}, Validation Accuracy: {result['validation_accuracy']:.4f}")
        print(f"Average Validation Loss: {avg_val_loss:.4f} ± {std_val_loss:.4f}")
        print(f"Average Validation Accuracy: {avg_val_acc:.4f} ± {std_val_acc:.4f}")

        return self.results, avg_val_loss, avg_val_acc, std_val_loss, std_val_acc
