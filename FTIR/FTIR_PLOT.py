import argparse
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import numpy as np
from pathlib import Path

def load_dpt(filename):
    df = pd.read_csv(filename, sep='\t', header=None, names=["Wavenumber", "Intensity"])
    return df

def baseline_correction(intensity, window_size=101, poly_order=3):
    baseline = savgol_filter(intensity, window_size, poly_order)
    return intensity - baseline

def parse_range(range_str):
    try:
        start, end = map(float, range_str.split("-"))
        return min(start, end), max(start, end)
    except ValueError:
        raise argparse.ArgumentTypeError("Range must be in the format 'start-end', e.g., '3000-3600'.")

def plot_ftir(filenames, apply_baseline=False, wn_range=None, mark=None, out_path=None):
    plt.figure(figsize=(10, 6))
    
    for file in filenames:
        df = load_dpt(file)

        if wn_range:
            low, high = wn_range
            df = df[(df["Wavenumber"] >= low) & (df["Wavenumber"] <= high)]

        x = df["Wavenumber"]
        y = df["Intensity"]
        if apply_baseline:
            y = baseline_correction(y)

        label = Path(file).stem  # filename without extension
        plt.plot(x, y, label=label)

    if mark:
        for w in mark:
            plt.axvline(x=w, color='gray', linestyle='--')
            plt.text(w, plt.ylim()[1]*0.95, f"{w:.0f} cm⁻¹", rotation=90, verticalalignment='top', color='gray')

    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("Intensity")
    plt.title("FTIR Spectrum")
    plt.gca().invert_xaxis()
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if out_path:
        out_path = Path(out_path)
        if out_path.suffix.lower() not in [".png", ".svg"]:
            out_path = out_path.with_suffix(".png")
        plt.savefig(out_path, dpi=300)
        print(f"Plot saved to: {out_path.resolve()}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs="?", help="Single DPT file to plot")
    parser.add_argument("--baseline", action="store_true", help="Apply baseline correction")
    parser.add_argument("--range", type=parse_range, help="Wavenumber range to plot, e.g., '3000-3600'")
    parser.add_argument("--outdir", type=str, help="Output file path or name (.png or .svg)")
    parser.add_argument("--multiadd", type=str, help="Folder containing multiple .dpt files")
    parser.add_argument("--mark", type=float, nargs="+", help="Wavenumber(s) to mark on the plot")

    args = parser.parse_args()

    if args.multiadd:
        folder = Path(args.multiadd)
        files = sorted(str(f) for f in folder.glob("*.dpt"))
    elif args.filename:
        files = [args.filename]
    else:
        raise ValueError("Either a single filename or --multiadd must be provided.")

    plot_ftir(files, apply_baseline=args.baseline, wn_range=args.range,
              mark=args.mark, out_path=args.outdir)
