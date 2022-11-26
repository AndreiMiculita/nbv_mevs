import pandas as pd
import matplotlib.pyplot as plt

# Plot the training and validation loss over time; the table has a Step and a Value column,
# where the Step column is the number of steps taken and the Value column is the loss
# The csv files are exported from Tensorboard, they are expected to contain the columns "Wall time", "Step" and "Value"

train_file = "run-version_38-tag-train_loss.csv"
val_file = "run-version_38-tag-val_loss.csv"

train_table = pd.read_csv(train_file)
val_table = pd.read_csv(val_file)

# smooth the training loss by averaging over 10 points
train_table["Value"] = train_table["Value"].rolling(10).mean()

plt.plot(train_table["Step"], train_table["Value"], label="Training Loss")
plt.plot(val_table["Step"], val_table["Value"], label="Validation Loss")

plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.savefig("training_validation_loss.pdf")
plt.show()

print("Final training loss: ", train_table["Value"].iloc[-1])
print("Final validation loss: ", val_table["Value"].iloc[-1])