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
    matrix1 = [[44., 03., 00., 00., 00., 00., 00., 02., 01., 00.],
               [00., 99., 00., 00., 00., 00., 00., 00., 01., 00.],
               [00., 02., 97., 00., 00., 00., 00., 00., 00., 01.],
               [00., 00., 02., 68., 01., 00., 02., 08., 05., 00.],
               [00., 00., 00., 01., 80., 00., 05., 00., 00., 00.],
               [00., 01., 00., 00., 02., 96., 00., 00., 01., 00.],
               [00., 00., 00., 01., 24., 00., 54., 00., 07., 00.],
               [00., 00., 00., 00., 00., 00., 01., 99., 00., 00.],
               [00., 00., 00., 15., 00., 00., 01., 00., 84., 00.],
               [01., 00., 00., 00., 00., 00., 00., 00., 00., 99.]]

    matrix1 = [[44.,  3.,  0.,  0.,  0.,  0.,  0.,  2.,  1.,  0.],
               [ 0., 95.,  0.,  0.,  0.,  0.,  0.,  4.,  0.,  1.],
               [ 0.,  1., 94.,  0.,  0.,  0.,  0.,  2.,  0.,  3.],
               [ 0.,  1.,  2., 68.,  2.,  0.,  1.,  6.,  6.,  0.],
               [ 0.,  0.,  0.,  0., 79.,  0.,  7.,  0.,  0.,  0.],
               [ 1.,  0.,  0.,  0.,  1., 98.,  0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  1., 24.,  0., 54.,  0.,  7.,  0.],
               [ 0.,  1.,  0.,  0.,  0.,  0.,  1., 98.,  0.,  0.],
               [ 0.,  0.,  0., 18.,  0.,  0.,  0.,  0., 82.,  0.],
               [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 99.],]


    # Baseline
    matrix2 = [[41., 03., 00., 00., 00., 00., 00., 03., 01., 02.],
               [00., 95., 00., 00., 00., 00., 00., 03., 01., 01.],
               [00., 02., 94., 00., 00., 00., 00., 00., 00., 04.],
               [01., 02., 01., 62., 03., 00., 00., 10., 07., 00.],
               [00., 00., 00., 00., 76., 00., 10., 00., 00., 00.],
               [01., 00., 00., 00., 02., 96., 00., 00., 01., 00.],
               [00., 00., 00., 01., 28., 00., 51., 00., 06., 00.],
               [00., 01., 00., 00., 00., 00., 01., 98., 00., 00.],
               [00., 00., 00., 22., 00., 00., 00., 00., 78., 00.],
               [01., 00., 00., 00., 00., 00., 00., 00., 00., 99.]]

    matrix1 = np.array(matrix1)
    matrix2 = np.array(matrix2)

    matrix1 = matrix1.T / ratios
    matrix2 = matrix2.T / ratios

    matrix1 *= 100
    matrix2 *= 100

    # convert to int
    matrix1 = matrix1.astype(int)
    matrix2 = matrix2.astype(int)

    # transpose back
    matrix1 = matrix1.T
    matrix2 = matrix2.T

    # Wider print
    np.set_printoptions(linewidth=200)

    print(matrix1)
    print()
    print(matrix2)

    # Plot the first matrix
    fig, (ax1) = plt.subplots(1, figsize=(5, 5))
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

    fig.tight_layout()
    # save
    os.makedirs(Path(__file__).parent.parent / 'assets', exist_ok=True)
    plt.savefig(Path(__file__).parent.parent / 'assets' / 'confusion_matrix1.pdf')

    # Plot the second matrix
    fig, (ax2) = plt.subplots(1, figsize=(5, 5))
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

    fig.tight_layout()
    # save
    os.makedirs(Path(__file__).parent.parent / 'assets', exist_ok=True)
    plt.savefig(Path(__file__).parent.parent / 'assets' / 'confusion_matrix2.pdf')
    plt.show()

    print(f'Instance accuracy1: {np.trace(matrix1) / np.sum(matrix1) * 100:.2f}%')
    print(f'Class accuracy1: {np.mean(np.diag(matrix1) / np.sum(matrix1, axis=1)) * 100:.2f}%')

    print(f'Instance accuracy2: {np.trace(matrix2) / np.sum(matrix2) * 100:.2f}%')
    print(f'Class accuracy2: {np.mean(np.diag(matrix2) / np.sum(matrix2, axis=1)) * 100:.2f}%')

    # Plot the difference
    fig, ax3 = plt.subplots(1, 1, figsize=(5, 5))

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
    plt.savefig(Path(__file__).parent.parent / 'assets' / 'confusion_matrix_diff.pdf')
    plt.show()


if __name__ == "__main__":
    main()
