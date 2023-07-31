"""
Bar chart of class distribution for ModelNet10.
One bar per class, height of the bar is the number of objects in that class.
Stack the bars for training and test set.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    counts_train = {}
    counts_test = {}

    for class_name in Path(os.path.expanduser('~/datasets/ModelNet10/')).iterdir():
        if class_name.name == 'README.txt':
            continue
        else:
            counts_train[class_name.name] = len(list((class_name / 'train').iterdir()))
            counts_test[class_name.name] = len(list((class_name / 'test').iterdir()))

    print(counts_train)
    print(counts_test)

    # Sort the dictionaries by key
    counts_train = dict(sorted(counts_train.items()))
    counts_test = dict(sorted(counts_test.items()))

    # Plot the stacked bar chart
    plt.bar(np.arange(len(counts_train)), counts_train.values(), label='Train set')
    plt.bar(np.arange(len(counts_test)), counts_test.values(), bottom=list(counts_train.values()), label='Test set')

    # Add the class names as xticks
    plt.xticks(np.arange(len(counts_train)), counts_train.keys(), rotation=45)
    plt.legend()

    # We keep the yticks on the left, and we add a 2nd y axis on the right, with the same ticks, but multiplied by 40
    # We use this 2nd axis to display the number of images captured for each class, at the same positions as the ticks
    # on the left y axis
    ax = plt.gca()
    ax2 = ax.twinx()
    # We set the yticks on the right axis to be the same as the left axis, but multiplied by 40, and divided by 1000,
    # suffixing them with a K
    yticks = ax.get_yticks()
    ax2.set_yticks([ytick * 40 for ytick in yticks])
    ax2.set_yticklabels([f'{int(ytick *40 / 1000)}K' for ytick in yticks])
    ax2.set_ylim(ax.get_ylim()[0] * 40, ax.get_ylim()[1] * 40)

    # for the left yticks, we set the label to "Number of object meshes"
    # for the right yticks, we set the label to "Number of captured images"
    ax.set_ylabel('Number of objects')
    ax2.set_ylabel('Number of captured views (40 per object)')

    # A bit more space at the bottom
    plt.subplots_adjust(bottom=0.2)

    plt.xlabel('Class')
    plt.title('Class distribution for ModelNet10')

    # Create assets dir if it doesn't exist
    if not os.path.exists('../assets/'):
        os.makedirs('../assets/')
    plt.savefig('../assets/dataset/modelnet10_class_distribution.pdf')

    plt.show()


if __name__ == '__main__':
    main()