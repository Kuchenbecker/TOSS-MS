import matplotlib.pyplot as plt
import pandas as pd

# Define the data
data = {
    "HCD %": [1, 5, 10, 15, 20, 25, 30, 35, 40, 50, 80],
    "135 m/z": [100, 100, 100, 99.8, 53.5, 20, 7.7, 8.4, 5.38, 0, 0],
    "117 m/z": [62.7, 55.2, 52.7, 100, 77.9, 43.8, 29.6, 17.4, 8.84, 0, 0],
    "77 m/z": [39.5, 38.7, 32.5, 84.9, 68.9, 51.4, 43.5, 35.8, 18.3, 16.6, 0],
    "59 m/z": [31.9, 28.7, 20.8, 88.1, 100, 100, 100, 100, 100, 100, 70]
}

# Create DataFrame
df = pd.DataFrame(data)

# Plotting
plt.figure(figsize=(10, 6))
for col in df.columns[1:]:
    plt.plot(df["HCD %"], df[col], marker='o', label=col)

plt.title("Fragment Ion Intensity vs HCD %")
plt.xlabel("HCD %")
plt.ylabel("Relative Intensity")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
