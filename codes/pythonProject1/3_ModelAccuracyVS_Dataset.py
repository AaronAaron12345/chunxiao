import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set global font sizes
plt.rcParams.update({
    'font.size': 24,         # Font size
    'axes.titlesize': 22,    # Title font size
    'axes.labelsize': 20,    # Axes label font size
    'xtick.labelsize': 18,   # X-axis tick label font size
    'ytick.labelsize': 18,   # Y-axis tick label font size
    'legend.fontsize': 18    # Legend font size
})
# Prepare data (same as in your original code)
data = {
    'Dataset': ['Prostate Cancer', 'Balloons', 'Lenses', 'Caesarian Section', 'Iris', 'Fertility', 'Zoo', 'Seeds', 'Haberman\'s Survival', 'Glass Identification', 'Yeast'],
    'n': [26, 16, 24, 80, 150, 100, 101, 210, 306, 214, 1484],
    'd': [4, 4, 4, 5, 4, 9, 16, 7, 3, 9, 8],
    'k': [2, 2, 3, 2, 2, 2, 2, 3, 2, 6, 10],
    'RF': [84.00, 77.58, 63.00, 59.23, 94.66, 88.00, 96.04, 92.85, 73.21, 78.93, 61.72],
    'XGBoost': [76.66, 77.50, 74.00, 64.57, 95.33, 88.00, 95.04, 93.33, 72.24, 77.54, 62.12],
    'TabPFN': [70.00, 81.41, 63.00, 60.38, 96.00, 88.00, 96.04, 95.23, 73.22, 71.96, 59.49],
    'HyperTab': [80.00, 68.08, 63.00, 63.23, 96.00, 88.00, 92.04, 86.19, 72.22, 46.73, 42.05],
    'AutoML': [76.00, 74.91, 60.00, 60.47, 96.00, 86.99, 96.04, 93.80, 71.25, 73.84, 61.65],
    'VAE-HNF': [92.30, 67.10, 95.83, 73.23, 97.33, 90.00, 100.00, 93.80, 73.52, 84.11, 62.73]
}

df = pd.DataFrame(data)

# Define a custom palette with significant color for VAE-HNF
palette = sns.color_palette("hsv", 6)
palette[5] = (1.0, 0.0, 0.0)  # Set VAE-HNF color to red

# Plot 1: Model Accuracy vs. Dataset Size (n)
plt.figure(figsize=(12, 8))
sns.lineplot(data=df.melt(id_vars=['Dataset', 'n'], value_vars=['RF', 'XGBoost', 'TabPFN', 'HyperTab', 'AutoML', 'VAE-HNF'], var_name='Model', value_name='Accuracy'), x='n', y='Accuracy', hue='Model', palette=palette)
plt.title('Model Accuracy vs. Dataset Size (n)', fontsize=30)
plt.xlabel('Number of Records (n)', fontsize=30)
plt.ylabel('Accuracy', fontsize=30)
plt.xscale('log')
plt.legend(title='Model',loc='best', fontsize=24)
plt.grid(True)

# Save as EPS file with high resolution
plt.savefig('model_accuracy_vs_dataset_size.eps', format='eps', dpi=300)

# Plot 2: Model Accuracy vs. Number of Features (d)
plt.figure(figsize=(12, 8))
sns.lineplot(data=df.melt(id_vars=['Dataset', 'd'], value_vars=['RF', 'XGBoost', 'TabPFN', 'HyperTab', 'AutoML', 'VAE-HNF'], var_name='Model', value_name='Accuracy'), x='d', y='Accuracy', hue='Model', palette=palette)
plt.title('Model Accuracy vs. Number of Features (d)', fontsize=30)
plt.xlabel('Number of Features (d)', fontsize=30)
plt.ylabel('Accuracy', fontsize=30)
plt.legend(title='Model',loc='best', fontsize=24)
plt.grid(True)

# Save as EPS file with high resolution
plt.savefig('model_accuracy_vs_number_of_features.eps', format='eps', dpi=300)

# Plot 3: Model Accuracy vs. Number of Target Classes (k)
plt.figure(figsize=(12, 8))
sns.lineplot(data=df.melt(id_vars=['Dataset', 'k'], value_vars=['RF', 'XGBoost', 'TabPFN', 'HyperTab', 'AutoML', 'VAE-HNF'], var_name='Model', value_name='Accuracy'), x='k', y='Accuracy', hue='Model', palette=palette)
plt.title('Model Accuracy vs. Number of Target Classes (k)', fontsize=30)
plt.xlabel('Number of Target Classes (k)', fontsize=30)
plt.ylabel('Accuracy', fontsize=30)
plt.legend(title='Model',loc='best', fontsize=24)
plt.grid(True)

# Save as EPS file with high resolution
plt.savefig('model_accuracy_vs_number_of_classes.png', format='png', dpi=300)

plt.show()
