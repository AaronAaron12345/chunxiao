import numpy as np
import matplotlib.pyplot as plt


def plot_comparison():
    # Load data
    data = np.load('plot_data.npz', allow_pickle=True)
    x_orig = data['x_original']
    x_dec = data['x_decoded']
    cls = data['classes']
    features = data['features'].tolist()

    # Set plotting parameters
    plt.rcParams.update({'font.size': 14, 'font.family': 'Times New Roman'})
    plt.figure(figsize=(10, 6))

    # Get feature indices
    laa_idx = features.index('LAA')
    glu_idx = features.index('Glutamate')

    # Plot original data
    plt.scatter(x_orig[cls == 0, laa_idx], x_orig[cls == 0, glu_idx],
                c='royalblue', alpha=0.6, label='non-PCa Original', s=50)
    plt.scatter(x_orig[cls == 1, laa_idx], x_orig[cls == 1, glu_idx],
                c='crimson', alpha=0.6, label='PCa Original', s=50)

    # Plot reconstructed data
    plt.scatter(x_dec[cls == 0, laa_idx], x_dec[cls == 0, glu_idx],
                edgecolors='navy', facecolors='none',
                marker='s', label='non-PCa Decoded', s=60)
    plt.scatter(x_dec[cls == 1, laa_idx], x_dec[cls == 1, glu_idx],
                edgecolors='darkred', facecolors='none',
                marker='^', label='PCa Decoded', s=60)

    # Add labels and legend
    plt.xlabel('LAA (Normalized)', fontsize=14)
    plt.ylabel('Glutamate (Normalized)', fontsize=14)
    plt.title('Original vs Decoded Feature Distribution', fontsize=16)
    plt.legend(loc='upper right', framealpha=0.9)

    # Save and display
    plt.savefig('Comparison.eps', format='eps', dpi=600, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    plot_comparison()