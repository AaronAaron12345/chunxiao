# -*- coding: utf-8 -*-
# train.py
import argparse
from cross_validator import CrossValidator

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train HyperNetFusion Model with Cross-Validation')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the CSV data file')
    parser.add_argument('--target_column', type=str, default='group', help='Name of the target column')
    parser.add_argument('--augment', action='store_true', help='Whether to perform data augmentation using VAE')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs per fold')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for optimizer')
    parser.add_argument('--num_folds', type=int, default=5, help='Number of folds for cross-validation')
    args = parser.parse_args()

    # Initialize CrossValidator
    cross_validator = CrossValidator(
        csv_path=args.csv_path,
        target_column=args.target_column,
        num_folds=args.num_folds,
        augment=args.augment,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate
    )

    # Run five-fold cross-validation
    results, avg_val_loss, avg_val_acc, std_val_loss, std_val_acc = cross_validator.run()

    # Optional: Save the overall results to a file
    # import json
    # with open('cross_validation_results.json', 'w') as f:
    #     json.dump(results, f, indent=4)
    # print("Cross-validation results saved to cross_validation_results.json")

if __name__ == '__main__':
    main()
