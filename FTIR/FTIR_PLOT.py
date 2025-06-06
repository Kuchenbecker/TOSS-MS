import argparse
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import numpy as np
from pathlib import Path

def load_dpt(filename):
    df = pd.read_csv(filename, sep='\t', header=None, names=["Wavenumber", "Intensity"])
    return df

def parse_range(range_str):
    try:
        start, end = map(float, range_str.split("-"))
        return min(start, end), max(start, end)
    except ValueError:
        raise argparse.ArgumentTypeError("Range must be in the format 'start-end', e.g., '3000-3600'.")

def parse_mark(mark_str):
    try:
        return [float(m.strip()) for m in mark_str.split(",") if m.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError("Marks must be comma-separated numbers, e.g., '2850,2920'.")

def baseline_align(intensity, wavenumber, baseline_region=(1900, 2200)):
    """Simple alignment using average intensity in a baseline region."""
    mask = (wavenumber >= baseline_region[0]) & (wavenumber <= baseline_region[1])
    if mask.sum() == 0:
        return intensity  # fallback: do nothing
    baseline_val = np.mean(intensity[mask])
    return intensity - baseline_val

def plot_ftir(filenames, apply_baseline=False, wn_range=None, marks=None, mark_offset=0.02, out_path=None):
    plt.figure(figsize=(10, 6))
    
    for file in filenames:
        df = load_dpt(file)

        if wn_range:
            low, high = wn_range
            df = df[(df["Wavenumber"] >= low) & (df["Wavenumber"] <= high)]

        x = df["Wavenumber"].values
        y = df["Intensity"].values

        if apply_baseline:
            y = baseline_align(y, x)

        label = Path(file).stem
        plt.plot(x, y, label=label)

    if marks:
        ymin, ymax = plt.ylim()
        for w in marks:
            if wn_range and not (wn_range[0] <= w <= wn_range[1]):
                continue  # Skip out-of-range marks
            plt.axvline(x=w, color='gray', linestyle='--')
            plt.text(w, ymax - mark_offset * (ymax - ymin), f"{w:.0f} cm⁻¹",
                     rotation=90, verticalalignment='top', horizontalalignment='right', color='gray')

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
    parser.add_argument("--baseline", action="store_true", help="Align spectra to same baseline level")
    parser.add_argument("--range", type=parse_range, help="Wavenumber range to plot, e.g., '3000-3600'")
    parser.add_argument("--outdir", type=str, help="Output file path (.png or .svg)")
    parser.add_argument("--multiadd", type=str, help="Folder with multiple .dpt files")
    parser.add_argument("--mark", type=parse_mark, help="Comma-separated wavenumbers to mark")
    parser.add_argument("--mark-offset", type=float, default=0.02,
                        help="Vertical offset for mark label as fraction of y-range (default=0.02)")

    args = parser.parse_args()

    if args.multiadd:
        folder = Path(args.multiadd)
        files = sorted(str(f) for f in folder.glob("*.dpt"))
    elif args.filename:
        files = [args.filename]
    else:
        raise ValueError("Either a single filename or --multiadd must be provided.")

    plot_ftir(
        files,
        apply_baseline=args.baseline,
        wn_range=args.range,
        marks=args.mark,
        mark_offset=args.mark_offset,
        out_path=args.outdir
    )
