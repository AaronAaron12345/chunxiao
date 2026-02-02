# -*- coding: utf-8 -*-
# evaluate.py
import torch
from models.hypernet_fusion import HyperNetFusion
from data.data_loader import get_data_loaders
from utils.utils import load_model, calculate_accuracy
import argparse
import pandas as pd

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Evaluate HyperNetFusion Model')
    parser.add_argument('--csv_path', type=str, default='data/prostate_cancer.csv', help='Path to the CSV data file')
    parser.add_argument('--target_column', type=str, default='label', help='Name of the target column')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--model_path', type=str, default='model_final.pth', help='Path to the trained model')
    parser.add_argument('--augment', action='store_true', help='Whether to perform data augmentation using VAE')
    args = parser.parse_args()

    # Hyperparameter settings (should match training settings)
    hypernet_input_dim = 20
    hypernet_hidden_dim = 128
    target_net_input_dim = 30
    target_net_hidden_dim = 64
    target_net_output_dim = len(pd.read_csv(args.csv_path)[args.target_column].unique())
    num_target_nets = 10
    batch_size = args.batch_size
    csv_path = args.csv_path
    target_column = args.target_column
    model_path = args.model_path

    # Data loading and augmentation
    _, test_loader, scaler = get_data_loaders(csv_path, target_column, test_size=0.2, batch_size=batch_size,
                                              augment=args.augment)

    # Initialize model
    model = HyperNetFusion(
        hypernet_input_dim=hypernet_input_dim,
        hypernet_hidden_dim=hypernet_hidden_dim,
        target_net_input_dim=target_net_input_dim,
        target_net_hidden_dim=target_net_hidden_dim,
        target_net_output_dim=target_net_output_dim,
        num_target_nets=num_target_nets
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Load the trained model
    load_model(model, model_path)

    # Evaluation
    model.eval()
    running_acc = 0.0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            hypernet_input = batch_x  # [batch_size, hypernet_input_dim]
            outputs = model(batch_x, hypernet_input)
            running_acc += (outputs.argmax(1) == batch_y).sum().item()

    test_acc = running_acc / len(test_loader.dataset)
    print(f'Test Accuracy: {test_acc:.4f}')


if __name__ == '__main__':
    main()
