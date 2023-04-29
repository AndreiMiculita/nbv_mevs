# NOTE: this turned out to be unnecessary because the networks were just not being loaded the right way
# This visualizes the sum of network weights as a pcolor plot
import os

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import torch

from pointnet2.models.pointnet2_ssg_entropyflat import PointNet2EntropySSG

ckpt_dir = 'outputs/entr-ssg'

# Increase width of terminal to show full arrays
np.set_printoptions(linewidth=500, suppress=True)


def main():
    list_of_lists_of_sums = []
    list_of_lists_of_stds = []

    ckpt_files = os.listdir(ckpt_dir)
    # Sort by epoch
    ckpt_files.sort(key=lambda x: int(x.split('=')[1].split('-')[0]))

    for ckpt in ckpt_files:
        if ckpt.endswith('.ckpt'):
            try:
                model = PointNet2EntropySSG.load_from_checkpoint(os.path.join(ckpt_dir, ckpt))
                model.eval()

                # Sum the weights of each layer, calculate std of each layer
                layer_sums = []
                layer_stds = []
                for name, param in model.named_parameters():
                    layer_sums.append((name, param.sum().abs().item()))
                    layer_stds.append((name, param.std().abs().item()))

                # Add to list of lists
                list_of_lists_of_sums.append((ckpt, layer_sums))
                list_of_lists_of_stds.append((ckpt, layer_stds))

            except RuntimeError as e:
                print(f'RuntimeError for {ckpt}: {e}')
                continue

    # Print the last ckpt
    print(f'Last ckpt: {ckpt}')
    model = PointNet2EntropySSG.load_from_checkpoint(os.path.join(ckpt_dir, ckpt))
    model.eval()
    with torch.no_grad():
        for name, param in model.named_parameters():
            print(f'{name}:\n{param}')

    # Convert to numpy array
    arr_sums = np.zeros((len(list_of_lists_of_sums), len(list_of_lists_of_sums[0][1])))
    arr_stds = np.zeros((len(list_of_lists_of_stds), len(list_of_lists_of_stds[0][1])))

    for i, (ckpt, layer_sums) in enumerate(list_of_lists_of_sums):
        for j, (name, sum) in enumerate(layer_sums):
            arr_sums[i, j] = sum
    for i, (ckpt, layer_stds) in enumerate(list_of_lists_of_stds):
        for j, (name, std) in enumerate(layer_stds):
            arr_stds[i, j] = std

    print(f'arr_sums.shape {arr_sums.shape}')
    print(f'ckpts handled: {len(list_of_lists_of_sums)}')
    print(f'arr_sums\n{arr_sums}')
    print(f'arr_stds\n{arr_stds}')

    # Plot 2 pcolor plots, one for sums and one for stds, log scale for colors, colorbars
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].set_title('Sum of weights per layer')
    ax[1].set_title('Std of weights per layer')
    ax[0].set_xlabel('Layer')
    ax[1].set_xlabel('Layer')
    ax[0].set_ylabel('Epoch')
    ax[1].set_ylabel('Epoch')
    ax[0].set_xticks(np.arange(0, arr_sums.shape[1]))
    ax[1].set_xticks(np.arange(0, arr_stds.shape[1]))
    # Set xticklabels to layer names
    ax[0].set_xticklabels([x[0] for x in list_of_lists_of_sums[0][1]])
    ax[1].set_xticklabels([x[0] for x in list_of_lists_of_sums[0][1]])
    # Diagonal xticklabels, move them left a bit to align with ticks
    for tick in ax[0].get_xticklabels():
        tick.set_rotation(45)
        tick.set_horizontalalignment('right')
    for tick in ax[1].get_xticklabels():
        tick.set_rotation(45)
        tick.set_horizontalalignment('right')
    ax[0].set_yticks(np.arange(0, arr_sums.shape[0]))
    ax[1].set_yticks(np.arange(0, arr_stds.shape[0]))
    ax[0].set_yticklabels([x[0] for x in list_of_lists_of_sums])
    ax[1].set_yticklabels([x[0] for x in list_of_lists_of_sums])
    ax[0].pcolor(arr_sums, norm=colors.LogNorm(vmin=arr_sums.min(), vmax=arr_sums.max()))
    ax[1].pcolor(arr_stds, norm=colors.LogNorm(vmin=arr_stds.min(), vmax=arr_stds.max()))
    # Make them both a bit wider horizontally
    ax[0].set_aspect(1.2)
    ax[1].set_aspect(1.2)
    fig.colorbar(ax[0].collections[0], ax=ax[0])
    fig.colorbar(ax[1].collections[0], ax=ax[1])
    # Add some padding to the bottom so the tick labels are visible
    fig.subplots_adjust(bottom=0.2)
    # Move everthing to the right a bit so the tick labels are visible
    fig.subplots_adjust(left=0.2)
    # Add a bit of space between the two plots
    fig.subplots_adjust(wspace=0.4)
    # Less padding on the left, to make the plots wider
    fig.subplots_adjust(right=1.0)
    plt.show()


if __name__ == '__main__':
    main()
