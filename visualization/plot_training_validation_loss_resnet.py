"""
The log only existed in stdout, so we have to be creative with parsing the output
The lines in the format "[0, 100] loss: 1.7620644104480743" are the training loss values,
 where the first number is the epoch and the second number is the last step the running loss was calculated for
 e.g. [1, 200] means epoch 1, and the running loss was calculated for steps 100-199
The lines prefixed with "Validation loss: " are the validation loss values
"""

import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

root_dir = Path(__file__).parent.parent
data_dir = root_dir / "data"
training_logs_dir = data_dir / "training_logs"
assets_dir = root_dir / "assets"

epoch_counts = []


def main(filename):
    file = training_logs_dir / filename

    with open(file, "r") as f:
        lines = f.readlines()

    train_loss = []
    val_loss = []
    for line in lines:
        print(line)
        # There are two types of lines in the logs: training loss and validation loss
        # Also for some files, the training loss is in the format "[0, 100] loss: 1.7620644104480743"
        # and for others it's in the format "Training loss for epoch 0, steps 0-99: 0.7559846869111061"
        # Also the validation loss can be in the format "Validation loss: 0.7559846869111061" or
        # "Validation loss for epoch 0: 0.7559846869111061"
        if "] loss:" in line:
            loss = float(re.findall(r"\[[0-9.]+, [0-9.]+] loss: ([0-9.]+)", line)[0])
            train_loss.append(loss)
        elif "Training loss for epoch" in line:
            loss = float(re.findall(r"Training loss for epoch [0-9.]+, steps [0-9.]+-[0-9.]+: ([0-9.]+)", line)[0])
            train_loss.append(loss)
        elif "Validation loss:" in line:
            loss = float(re.findall(r"Validation loss: ([0-9.]+)", line)[0])
            val_loss.append(loss)
        elif "Validation loss for epoch" in line:
            loss = float(re.findall(r"Validation loss for epoch [0-9.]+: ([0-9.]+)", line)[0])
            val_loss.append(loss)

    print(f'Train loss: {train_loss}')
    print(f'Val loss: {val_loss}')
    print(f'file: {file}')

    steps_per_epoch = len(train_loss) / len(val_loss)

    # Plot the training and validation loss
    plt.plot(np.arange(len(train_loss)), train_loss, label="Training")
    plt.plot(np.arange(len(val_loss)) * steps_per_epoch + steps_per_epoch, val_loss, label="Validation")

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Loss for model {filename.split('_')[0].replace('resnet', 'ResNet-')},"
              f" {filename.split('_')[2][:2]} {filename.split('_')[1]} views")
    plt.legend()

    os.makedirs(assets_dir / "resnet", exist_ok=True)
    # filename based on the name of the log file
    output_filename = file.stem + ".pdf"
    plt.savefig(assets_dir / "resnet" / output_filename)
    plt.show()

    print("Final training loss: ", train_loss[-1])
    print("Final validation loss: ", val_loss[-1])

    epoch_counts.append(len(val_loss))


if __name__ == "__main__":
    for filename in [
        'resnet18_image_10views_training_output.txt',
        'resnet18_depth_40views_training_output.txt',
        'resnet18_image_40views_training_output.txt',
        'resnet34_depth_40views_training_output.txt',
    ]:
        main(filename)
    print(epoch_counts)
