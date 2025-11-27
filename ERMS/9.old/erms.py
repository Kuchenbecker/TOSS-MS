
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_sy_curve(csv_path, ions, out_path=None):
    df = pd.read_csv(csv_path)
    df["mz_rounded"] = df["mz"].round(4)
    grouped_df = df.groupby(["collision_energy", "mz_rounded"], as_index=False)["intensity"].mean()
    ces = sorted(grouped_df["collision_energy"].unique())
    sy_data = {ion: [] for ion in ions}

    for ce in ces:
        subset = grouped_df[grouped_df["collision_energy"] == ce]
        for ion in ions:
            matched = subset[np.abs(subset["mz_rounded"] - ion) <= 0.005]
            mean_intensity = matched["intensity"].mean() if not matched.empty else 0.0
            sy_data[ion].append(mean_intensity)

    plt.figure(figsize=(10, 6))
    for ion in ions:
        plt.plot(ces, sy_data[ion], marker='o', label=f"m/z {ion:.4f}")
    plt.xlabel("Collision Energy (CE)")
    plt.ylabel("Average Intensity")
    plt.title("Survival Yield Breakdown Curve")
    plt.legend()
    plt.grid(True)

    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Generate Survival Yield Breakdown Curve from CSV.")
    parser.add_argument("--csv", required=True, help="Path to input CSV file.")
    parser.add_argument("--ions", required=True, help="Comma-separated list of m/z values to plot (4 decimal places).")
    parser.add_argument("--out", required=False, help="Output path to save the plot (with extension).")

    args = parser.parse_args()
    ion_list = [round(float(i), 4) for i in args.ions.split(",")]
    generate_sy_curve(args.csv, ion_list, args.out)

if __name__ == "__main__":
    main()
