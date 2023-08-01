import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Generate a histogram of the data in the "entropy" column of the csv file (normalized to 0-1), with 100 bins
# Print the mean, standard deviation, and mode of the data

entropy_csv = "../data/entropy-dataset-mnet10-10samples-full/entropy_dataset.csv"
entropy_table = pd.read_csv(entropy_csv)
entropy_table["entropy"] = entropy_table["entropy"] / entropy_table["entropy"].max()

# Split into train and test
# The test set can be obtained by parsing the filenames in ~/datasets/ModelNet10/
# All the off files in which the parent directory is "test" are in the test set
test_set_filenames = set()
for file in Path("/home/andrei/datasets/ModelNet10/").rglob("*.off"):
    if file.parent.name == "test":
        test_set_filenames.add(file.name)
print("Test set filenames size: ", len(test_set_filenames))
print("Test set filenames sample: ", list(test_set_filenames)[0])

# The filenames are of the format {label}_{obj_ind}.off, where obj_ind is padded with zeros to 4 digits
# We select from the table the rows in which the columns "label" and "obj_ind" concatenated are in the test set
test_set_indices = set()
for index, row in entropy_table.iterrows():
    if f"{row['label']}_{row['obj_ind']:04d}.off" in test_set_filenames:
        test_set_indices.add(index)

test_set = entropy_table.iloc[list(test_set_indices)]
train_set = entropy_table.iloc[list(set(entropy_table.index) - test_set_indices)]

print("Train set size: ", len(train_set))
print("Test set size: ", len(test_set))
# Plot the entropy distribution for the train and test sets, separately (train in blue, test in red)
plt.hist([train_set["entropy"], test_set["entropy"]], bins=50, label="Train")
# plt.hist(, bins=50, label="Test")

plt.xlabel("Entropy")
plt.ylabel("Frequency")
plt.title("Entropy Distribution")
plt.legend()
# Create assets dir if it doesn't exist
if not os.path.exists("../assets/dataset/"):
    os.makedirs("../assets/dataset/")
plt.show()

#sets up the axis and gets histogram data
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.hist([train_set["entropy"], test_set["entropy"]], bins=30, label=["Train","Test"], color=["C0", "C1"])
n, bins, patches = ax1.hist([train_set["entropy"],test_set["entropy"]], bins=30, label=["Train","Test"], color=["C0", "C1"])
ax1.cla() #clear the axis

#plots the histogram data
width = (bins[1] - bins[0]) * 0.5
bins_shifted = bins - width
ax1.bar(bins[:-1], n[0], width, align='edge', color='C0', label="Train")
ax2.bar(bins_shifted[:-1], n[1], width, align='edge', color='C1', label="Test")

# Add legend manually (C0 is Train, C1 is Test)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# finishes the plot
ax1.set_ylabel("Train Set Count")
ax2.set_ylabel("Test Set Count")
ax1.set_xlabel("Entropy")
ax1.tick_params('y')
ax2.tick_params('y')
plt.tight_layout()
plt.savefig("../assets/dataset/entropy_distribution.pdf")
plt.show()

print("Mean: ", entropy_table["entropy"].mean())
print("Standard Deviation: ", entropy_table["entropy"].std())
print("Mode: ", entropy_table["entropy"].mode())
