"""
We plot two confusion matrices, and the difference between them.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    # Looks like we forgot about class imbalance, so we need to normalize the confusion matrices
    ratios = [50, 100, 100, 86, 86, 100, 86, 100, 100, 100]

    # class names
    class_names = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']

    # Our method
    matrix1 = [[44., 3., 0., 0., 0., 0., 0., 2., 1., 0.],
               [0., 99., 0., 0., 0., 0., 0., 0., 1., 0.],
               [0., 2., 97., 0., 0., 0., 0., 0., 0., 1.],
               [0., 0., 2., 68., 1., 0., 2., 8., 5., 0.],
               [0., 0., 0., 1., 80., 0., 5., 0., 0., 0.],
               [0., 1., 0., 0., 2., 96., 0., 0., 1., 0.],
               [0., 0., 0., 1., 24., 0., 54., 0., 7., 0.],
               [0., 0., 0., 0., 0., 0., 1., 99., 0., 0.],
               [0., 0., 0., 15., 0., 0., 1., 0., 84., 0.],
               [1., 0., 0., 0., 0., 0., 0., 0., 0., 99.]]

    # Baseline
    matrix2 = [[41., 3., 0., 0., 0., 0., 0., 3., 1., 2.],
               [0., 95., 0., 0., 0., 0., 0., 3., 1., 1.],
               [0., 2., 94., 0., 0., 0., 0., 0., 0., 4.],
               [1., 2., 1., 62., 3., 0., 0., 10., 7., 0.],
               [0., 0., 0., 0., 76., 0., 10., 0., 0., 0.],
               [1., 0., 0., 0., 2., 96., 0., 0., 1., 0.],
               [0., 0., 0., 1., 28., 0., 51., 0., 6., 0.],
               [0., 1., 0., 0., 0., 0., 1., 98., 0., 0.],
               [0., 0., 0., 22., 0., 0., 0., 0., 78., 0.],
               [1., 0., 0., 0., 0., 0., 0., 0., 0., 99.]]

    matrix1 = np.array(matrix1)
    matrix2 = np.array(matrix2)

    matrix1 = matrix1.T / ratios
    matrix2 = matrix2.T / ratios

    matrix1 *= 100
    matrix2 *= 100

    # convert to int
    matrix1 = matrix1.astype(int)
    matrix2 = matrix2.astype(int)

    # Wider print
    np.set_printoptions(linewidth=200)

    print(matrix1)
    print()
    print(matrix2)

    # Plot the matrices
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    im1 = ax1.imshow(matrix1, cmap='Blues')
    ax1.set_xticks(np.arange(len(class_names)))
    ax1.set_yticks(np.arange(len(class_names)))
    ax1.set_xticklabels(class_names)
    ax1.set_yticklabels(class_names)
    # Angle the x axis labels
    for tick in ax1.get_xticklabels():
        tick.set_rotation(45)

    ax1.set_title('Our method')
    ax1.set_xlabel('Predicted label')
    ax1.set_ylabel('True label')
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = ax1.text(j, i, f'{matrix1[i, j]}',
                            ha="center", va="center", color="w" if matrix2[i, j] > 50 else "k")

    im2 = ax2.imshow(matrix2, cmap='Blues')
    ax2.set_xticks(np.arange(len(class_names)))
    ax2.set_yticks(np.arange(len(class_names)))
    ax2.set_xticklabels(class_names)
    ax2.set_yticklabels(class_names)
    # Angle the x axis labels
    for tick in ax2.get_xticklabels():
        tick.set_rotation(45)
    ax2.set_title('Baseline')
    ax2.set_xlabel('Predicted label')
    ax2.set_ylabel('True label')
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = ax2.text(j, i, f'{matrix2[i, j]}',
                            ha="center", va="center", color="w" if matrix2[i, j] > 50 else "k")

    im3 = ax3.imshow(matrix1 - matrix2, cmap='Blues')
    ax3.set_xticks(np.arange(len(class_names)))
    ax3.set_yticks(np.arange(len(class_names)))
    ax3.set_xticklabels(class_names)
    ax3.set_yticklabels(class_names)
    # Angle the x axis labels
    for tick in ax3.get_xticklabels():
        tick.set_rotation(45)
    ax3.set_title('Difference')
    ax3.set_xlabel('Predicted label')
    ax3.set_ylabel('True label')
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = ax3.text(j, i, f'{matrix1[i, j] - matrix2[i, j]}',
                            ha="center", va="center", color="w" if matrix2[i, j] > 50 else "k")

    fig.tight_layout()
    # save
    os.makedirs(Path(__file__).parent.parent / 'assets', exist_ok=True)
    plt.savefig(Path(__file__).parent.parent / 'assets' / 'confusion_matrices.pdf')
    plt.show()


if __name__ == "__main__":
    main()
