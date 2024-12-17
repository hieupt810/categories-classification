import matplotlib.pyplot as plt
import pandas as pd

fig, ax = plt.subplots(2, 1, figsize=(16, 12))

result = pd.read_csv("./results.csv")

min_val_loss = float("inf")
for _, row in result.iterrows():
    if row["valid_loss"] < min_val_loss:
        min_val_loss = row["valid_loss"]
        min_val_loss_row = row

print(f"Best epoch:\n{min_val_loss_row}")

fig_loss, ax_loss = plt.subplots(figsize=(8, 6))
ax_loss.plot(result["epoch"], result["train_loss"], label="Train loss")
ax_loss.plot(result["epoch"], result["valid_loss"], label="Valid loss")
ax_loss.set_title("Loss")
ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("Loss")
ax_loss.set_xticks(result["epoch"][::2])
ax_loss.plot(min_val_loss_row["epoch"], min_val_loss_row["valid_loss"], "ro")
ax_loss.legend()
fig_loss.tight_layout()
fig_loss.savefig("./loss_plot.png")

fig_acc, ax_acc = plt.subplots(figsize=(8, 6))
ax_acc.plot(result["epoch"], result["train_acc"], label="Train accuracy")
ax_acc.plot(result["epoch"], result["valid_acc"], label="Valid accuracy")
ax_acc.set_title("Accuracy")
ax_acc.set_xlabel("Epoch")
ax_acc.set_ylabel("Accuracy")
ax_acc.set_xticks(result["epoch"][::2])
ax_acc.plot(min_val_loss_row["epoch"], min_val_loss_row["valid_acc"], "ro")
ax_acc.legend()
fig_acc.tight_layout()
fig_acc.savefig("./accuracy_plot.png")
