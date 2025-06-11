import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# Data
data = {
    "HCD %": [1, 5, 10, 15, 20, 25, 30, 35, 40, 50, 80],
    "135 m/z": [100, 100, 100, 99.8, 53.5, 20, 7.7, 8.4, 5.38, 0, 0],
    "117 m/z": [62.7, 55.2, 52.7, 100, 77.9, 43.8, 29.6, 17.4, 8.84, 0, 0],
    "77 m/z": [39.5, 38.7, 32.5, 84.9, 68.9, 51.4, 43.5, 35.8, 18.3, 16.6, 0],
    "59 m/z": [31.9, 28.7, 20.8, 88.1, 100, 100, 100, 100, 100, 100, 70]
}

df = pd.DataFrame(data)

# Sigmoid function
def sigmoid(x, L, x0, k, b):
    return L / (1 + np.exp(-k * (x - x0))) + b

# R² calculation
def calculate_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# Output folder
output_dir = "sigmoid_fits"
os.makedirs(output_dir, exist_ok=True)

# Fit and plot each m/z
x = df["HCD %"].values

for col in df.columns[1:]:
    y = df[col].values

    # Determine trend: increasing or decreasing
    increasing = y[-1] > y[0]

    # Bounds depending on trend
    if increasing:
        bounds = ([50, 0, 0, 0], [150, 100, 1, 50])
    else:
        bounds = ([50, 0, -1, 0], [150, 100, 0, 50])

    # Fit curve
    try:
        popt, _ = curve_fit(sigmoid, x, y, bounds=bounds)
        y_fit = sigmoid(x, *popt)
        r2 = calculate_r2(y, y_fit)

        # Plot
        plt.figure(figsize=(8, 5))
        plt.plot(x, y, 'o', label=f"{col} data")
        plt.plot(x, y_fit, '--', label=f"{col} fit (R²={r2:.3f})")
                # Equation string
        eq = f"{popt[0]:.1f} / (1 + exp({-popt[2]:.2f}(x - {popt[1]:.1f}))) + {popt[3]:.1f}"

        # Positioning the text based on trend
        if increasing:
            text_x = min(x) + 2
            text_y = min(y) + (max(y) - min(y)) * 0.1
        else:
            text_x = min(x) + 2
            text_y = max(y) - (max(y) - min(y)) * 0.2

        # Add the equation with a bounding box
        plt.text(
            text_x, text_y,
            f"{col}:\n{eq}",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray")
        )

        plt.title(f"Sigmoid Fit for {col}")
        plt.xlabel("HCD %")
        plt.ylabel("Relative Intensity")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save plot
        safe_col = col.replace(" ", "_").replace("/", "_")
        filename = os.path.join(output_dir, f"fit_{safe_col}.png")
        plt.savefig(filename)
                # ---------- Plot Derivative ----------
        def sigmoid_derivative(x, L, x0, k, b):
            exp_term = np.exp(-k * (x - x0))
            return (L * k * exp_term) / (1 + exp_term)**2

        y_derivative = sigmoid_derivative(x, *popt)

        plt.figure(figsize=(8, 5))
        plt.plot(x, y_derivative, 'g-', marker='o', label="Derivative")
        plt.title(f"Derivative of Sigmoid Fit for {col}")
        plt.xlabel("HCD %")
        plt.ylabel("d(Intensity)/d(HCD %)")
        plt.grid(True)
        plt.legend()

        # Save derivative plot
        safe_col = col.replace(" ", "_").replace("/", "_")
        deriv_filename = os.path.join(output_dir, f"derivative_{safe_col}.png")
        plt.savefig(deriv_filename)
        plt.close()
        print(f"Saved: {deriv_filename}")

        plt.close()
        print(f"Saved: {filename}")
    except RuntimeError:
        print(f"Fit failed for {col}")
