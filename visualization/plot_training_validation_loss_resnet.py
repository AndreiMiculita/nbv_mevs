"""
The log only existed in stdout, so we have to be creative with parsing the output
The lines in the format "[0, 100] loss: 1.7620644104480743" are the training loss values,
 where the first number is the epoch and the second number is the last step the running loss was calculated for
 e.g. [1, 200] means epoch 1, and the running loss was calculated for steps 100-199
The lines prefixed with "Validation loss: " are the validation loss values
"""

import os
import re

import matplotlib.pyplot as plt
import numpy as np


def main():
    file = "../data/training_progress/resnet_training.txt"

    with open(file, "r") as f:
        lines = f.readlines()

    train_loss = []
    val_loss = []
    for line in lines:
        print(line)
        if "] loss:" in line:
            print("train loss")
            loss = float(re.findall(r"\[[0-9.]+, [0-9.]+] loss: ([0-9.]+)", line)[0])
            train_loss.append(loss)
        elif "Validation loss:" in line:

            print("val loss")
            loss = float(re.findall(r"Validation loss: ([0-9.]+)", line)[0])
            val_loss.append(loss)

    steps_per_epoch = len(train_loss) // len(val_loss)

    plt.plot(np.array(range(1, len(train_loss) + 1)) / steps_per_epoch, train_loss, label="Training Loss")
    plt.plot(np.array(range(steps_per_epoch, len(train_loss) + 1, steps_per_epoch)) / steps_per_epoch, val_loss, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    # Create assets dir if it doesn't exist
    if not os.path.exists("../assets/resnet/"):
        os.makedirs("../assets/resnet/")
    plt.savefig("../assets/resnet/training_validation_loss.pdf")
    plt.show()

    print("Final training loss: ", train_loss[-1])
    print("Final validation loss: ", val_loss[-1])


if __name__ == "__main__":
    main()
