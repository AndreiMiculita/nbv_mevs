import os

import matplotlib.pyplot as plt
import pandas as pd

# Generate a histogram of the data in the "entropy" column of the csv file (normalized to 0-1), with 100 bins
# Print the mean, standard deviation, and mode of the data

entropy_csv = "../data/entropy-dataset-mnet10-10samples-full/entropy_dataset.csv"
entropy_table = pd.read_csv(entropy_csv)
entropy_table["entropy"] = entropy_table["entropy"] / entropy_table["entropy"].max()
entropy_table["entropy"].hist(bins=50)
plt.xlabel("Entropy")
plt.ylabel("Frequency")
plt.title("Entropy Distribution")
# Create assets dir if it doesn't exist
if not os.path.exists("../assets/dataset/"):
    os.makedirs("../assets/dataset/")
plt.savefig("../assets/dataset/entropy_distribution.pdf")
plt.show()

print("Mean: ", entropy_table["entropy"].mean())
print("Standard Deviation: ", entropy_table["entropy"].std())
print("Mode: ", entropy_table["entropy"].mode())
