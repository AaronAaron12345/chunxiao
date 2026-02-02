import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import statsmodels.api as sm

# Define the function to plot ROC curves
def plot_roc_curves(models, dataset_name):
    plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 16})

    # Get the colormap
    colormap = plt.get_cmap('tab10')

    # Generate a list of colors from the colormap
    colors = [colormap(i % colormap.N) for i in range(len(models))]

    for idx, (model_key, model_name) in enumerate(models.items()):
        roc_file = f'{dataset_name}_{model_key}_roc_data.csv'
        auc_file = roc_file.replace('.csv', '_auc.txt')
        if os.path.exists(roc_file) and os.path.exists(auc_file):
            # Read ROC data
            roc_data = pd.read_csv(roc_file)
            fpr = roc_data['FPR']
            tpr = roc_data['TPR']
            # Read AUC value
            with open(auc_file, 'r') as f:
                auc_line = f.readline()
                auc_value = float(auc_line.strip().split(': ')[1])
            # Highlight the VAE-HNF line
            if model_name == 'VAE-HNF':
                line_color = 'red'
                line_width = 3
                line_style = '-'
            else:
                line_color = colors[idx]
                line_width = 2
                line_style = '--'
            # Plot the ROC curve
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_value:.2f})',
                     linestyle=line_style,
                     color=line_color,
                     linewidth=line_width)
        else:
            print(f'File not found for {model_name}, skipping.')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate (FPR)', fontsize=24)
    plt.ylabel('True Positive Rate (TPR)', fontsize=24)
    plt.title('ROC Curves of Different Models', fontsize=24)
    plt.legend(loc='lower right', fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('roc_curves_comparison.png', dpi=300)
    plt.savefig('roc_curves_comparison.eps', format='eps', dpi=300)
    plt.savefig('precision_recall_curves_comparison.eps', format='eps', dpi=300)

    plt.show()

# Define the function to plot Precision-Recall curves
def plot_precision_recall_curves(models, dataset_name):
    plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 16})
    colormap = plt.get_cmap('tab10')
    colors = [colormap(i % colormap.N) for i in range(len(models))]

    for idx, (model_key, model_name) in enumerate(models.items()):
        pr_file = f'{dataset_name}_{model_key}_pr_data.csv'
        if os.path.exists(pr_file):
            # Read Precision-Recall data
            pr_data = pd.read_csv(pr_file)
            precision = pr_data['Precision']
            recall = pr_data['Recall']
            # Highlight the VAE-HNF line
            if model_name == 'VAE-HNF':
                line_color = 'red'
                line_width = 3
                line_style = '-'
            else:
                line_color = colors[idx]
                line_width = 2
                line_style = '--'
            # Plot the Precision-Recall curve
            plt.plot(recall, precision, label=model_name,
                     linestyle=line_style,
                     color=line_color,
                     linewidth=line_width)
        else:
            print(f'File not found for {model_name}, skipping.')
    plt.xlabel('Recall', fontsize=24)
    plt.ylabel('Precision', fontsize=24)
    plt.title('Precision-Recall Curves of Different Models', fontsize=24)
    plt.legend(loc='lower left', fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('precision_recall_curves_comparison.png', dpi=300)
    plt.show()

# Define the function to plot confusion matrices
def plot_confusion_matrices(models, dataset_name):
    # Collect models with available data
    models_with_data = []
    for model_key, model_name in models.items():
        cm_file = f'{dataset_name}_{model_key}_confusion_matrix.csv'
        if os.path.exists(cm_file):
            models_with_data.append((model_key, model_name))
        else:
            print(f'File not found for {model_name}, skipping.')

    num_models = len(models_with_data)
    if num_models == 0:
        print("No confusion matrices found to plot.")
        return

    fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 5))
    plt.rcParams.update({'font.size': 16})
    if num_models == 1:
        axes = [axes]  # Ensure axes is iterable

    for idx, (model_key, model_name) in enumerate(models_with_data):
        cm_file = f'{dataset_name}_{model_key}_confusion_matrix.csv'
        # Read confusion matrix data
        cm_data = pd.read_csv(cm_file, header=None).values
        # Ensure cm_data is square
        if cm_data.shape[0] != cm_data.shape[1]:
            print(f'Confusion matrix for {model_name} is not square, skipping.')
            continue
        # Plot confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_data)
        disp.plot(ax=axes[idx], colorbar=False, cmap='Blues')
        axes[idx].set_title(model_name, fontsize=16)
        axes[idx].set_xlabel('Predicted Label', fontsize=14)
        axes[idx].set_ylabel('True Label', fontsize=14)

    plt.tight_layout()
    plt.savefig('confusion_matrices_comparison.png', dpi=300)
    plt.savefig('confusion_matrices_comparison.eps', format='eps', dpi=300)

    plt.show()

# Define the function to plot noise robustness
def plot_noise_robustness(models, dataset_name):
    plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 16})
    colormap = plt.get_cmap('tab10')
    colors = [colormap(i % colormap.N) for i in range(len(models))]

    for idx, (model_key, model_name) in enumerate(models.items()):
        noise_file = f'{dataset_name}_{model_key}_noise_robustness.csv'
        if os.path.exists(noise_file):
            # Read noise robustness data
            noise_data = pd.read_csv(noise_file)
            noise_levels = noise_data['Noise Level']
            accuracies = noise_data['Accuracy']
            # Highlight the VAE-HNF line
            if model_name == 'VAE-HNF':
                line_color = 'red'
                line_width = 3
                line_style = '-'
                marker_style = 'o'
            else:
                line_color = colors[idx]
                line_width = 2
                line_style = '--'
                marker_style = 's'
            # Plot the noise robustness curve
            plt.plot(noise_levels, accuracies, label=model_name,
                     linestyle=line_style,
                     color=line_color,
                     linewidth=line_width,
                     marker=marker_style,
                     markersize=8)
        else:
            print(f'File not found for {model_name}, skipping.')

    plt.xlabel('Noise Level (Standard Deviation)', fontsize=24)
    plt.ylabel('Accuracy', fontsize=24)
    plt.title('Robustness of Models at Different Noise Levels', fontsize=24)
    plt.legend(loc='best', fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('noise_robustness_comparison.png', dpi=300)
    plt.savefig('noise_robustness_comparison.eps', format='eps', dpi=300)

    plt.show()

# Define the function to plot ablation study results
def plot_ablation_study(models, dataset_name):
    plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 16})
    colormap = plt.get_cmap('tab10')
    colors = [colormap(i % colormap.N) for i in range(len(models))]

    for idx, (model_key, model_name) in enumerate(models.items()):
        ablation_file = f'{dataset_name}_{model_key}_ablation_study.csv'
        if os.path.exists(ablation_file):
            # Read ablation study data
            ablation_data = pd.read_csv(ablation_file)
            components = ablation_data['Removed Component']
            accuracies = ablation_data['Accuracy']
            # Highlight the VAE-HNF line
            if model_name == 'VAE-HNF':
                line_color = 'red'
                line_width = 3
                line_style = '-'
                marker_style = 'o'
            else:
                line_color = colors[idx]
                line_width = 2
                line_style = '--'
                marker_style = 's'
            # Plot the ablation study results
            plt.plot(components, accuracies, label=model_name,
                     linestyle=line_style,
                     color=line_color,
                     linewidth=line_width,
                     marker=marker_style,
                     markersize=8)
        else:
            print(f'File not found for {model_name}, skipping.')
    plt.xlabel('Removed Component', fontsize=24)
    plt.ylabel('Accuracy', fontsize=24)
    plt.title('Ablation Study Results Comparison', fontsize=24)
    plt.legend(loc='best', fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('ablation_study_comparison.png', dpi=300)
    plt.savefig('ablation_study_comparison.eps', format='eps', dpi=300)

    plt.show()

# Main function to call all plotting functions
def main():
    # Define the models and their display names
    models = {
        'random_forest': 'Random Forest',
        'xgboost': 'XGBoost',
        'tabpfn': 'TabPFN',
        'hypertab': 'HyperTab',
        'tpot': 'TPOT',
        'vae': 'VAE-HNF'  # Keep 'vae' for file names, display as 'VAE-HNF'
    }

    # Dataset name
    dataset_name = 'balloons'  # Replace with your dataset name

    # Call the plotting functions
    plot_roc_curves(models, dataset_name)
    plot_precision_recall_curves(models, dataset_name)
    # plot_confusion_matrices(models, dataset_name)
    plot_noise_robustness(models, dataset_name)
    # plot_ablation_study(models, dataset_name)

if __name__ == '__main__':
    main()
