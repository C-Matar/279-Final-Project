import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

df = pd.read_csv("predictions.csv")

plt.figure()
plt.scatter(df["true_energy"], df["predicted_energy"], alpha=0.6, s=10)
plt.plot([df["true_energy"].min(), df["true_energy"].max()],
         [df["true_energy"].min(), df["true_energy"].max()], 'r--')
plt.xlabel("True Energy")
plt.ylabel("Predicted Energy")
plt.title("True vs. Predicted Energy")
plt.grid(True)
plt.tight_layout()
plt.savefig("true_vs_predicted_energy.png", dpi=300)

df["abs_error"] = np.abs(df["true_energy"] - df["predicted_energy"])

plt.figure()
plt.hist(df["abs_error"], bins=50, edgecolor='black')
plt.xlabel("Absolute Error")
plt.ylabel("Frequency")
plt.title("Histogram of Prediction Errors")
plt.tight_layout()
plt.savefig("error_histogram.png", dpi=300)

mae = mean_absolute_error(df["true_energy"], df["predicted_energy"])
rmse = np.sqrt(mean_squared_error(df["true_energy"], df["predicted_energy"]))
print(f"MAE: {mae:.6f}  |  RMSE: {rmse:.6f}")


