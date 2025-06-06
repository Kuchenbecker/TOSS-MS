import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import os

# Define fitting functions
def linear_func(x, a, b): return a * x + b
def quadratic_func(x, a, b, c): return a * x**2 + b * x + c
def polynomial_func(x, *coeffs): return np.polyval(coeffs, x)
def log_func(x, a, b): return a * np.log10(x) + b
def ln_func(x, a, b): return a * np.log(x) + b
def exp_func(x, a, b, c): return a * np.exp(b * x) + c
def sigmoid_func(x, a, b, c, d): return a / (1 + np.exp(-b * (x - c))) + d

def main():
    parser = argparse.ArgumentParser(description='Plot data from CSV file with optional fitting.')
    parser.add_argument('filename', help='Path to the CSV file')
    parser.add_argument('--fit', choices=['linear', 'quadratic', 'poly', 'log', 'ln', 'exp', 'sig'], help='Type of function to fit')
    parser.add_argument('--polydeg', type=int, default=2, help='Degree of polynomial fit')
    parser.add_argument('--outdir', type=str, default=None, help='Path to save output file (supports .png or .svg)')
    parser.add_argument('--title', type=str, help='Title for the plot')
    parser.add_argument('--yaxis', type=int, default=2, help='Column number for y-axis (1-based)')
    parser.add_argument('--xlabel', type=str, default=None, help='Custom x-axis label')
    parser.add_argument('--ylabel', type=str, default=None, help='Custom y-axis label')
    parser.add_argument('--dlabel', type=str, default=None, help='Column number (1-based) for right-side y-axis labels')
    parser.add_argument('--show', action='store_true', help='Show Y-values above each point')

    args = parser.parse_args()

    try:
        data = pd.read_csv(args.filename)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    if len(data.columns) < 2:
        print("Error: CSV file must have at least 2 columns")
        return

    y_col = args.yaxis - 1
    x_col = 0 if y_col != 0 else 1

    if y_col >= len(data.columns) or y_col < 0:
        print(f"Error: yaxis value {args.yaxis} is out of range.")
        return

    x_data = data.iloc[:, x_col].values
    y_data = data.iloc[:, y_col].values

    x_label = args.xlabel if args.xlabel else data.columns[x_col]
    y_label = args.ylabel if args.ylabel else data.columns[y_col]

    # Optional dlabel column
    dlabel_values = None
    if args.dlabel and args.dlabel.isdigit():
        dlabel_col = int(args.dlabel) - 1
        if 0 <= dlabel_col < len(data.columns):
            dlabel_values = data.iloc[:, dlabel_col].astype(str).values
        else:
            print(f"Warning: dlabel column {args.dlabel} out of range")

    # Create main figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x_data, y_data, color='blue', label='Data')

    # --show: Display y-values with vertical jitter to avoid overlap
    if args.show:
        sorted_indices = np.argsort(y_data)
        seen = {}
        for idx in sorted_indices:
            y_val = y_data[idx]
            offset_count = seen.get(y_val, 0)
            offset = offset_count * 0.03 * (max(y_data) - min(y_data))
            seen[y_val] = offset_count + 1
            ax.annotate(f'{y_val:.2f}', (x_data[idx], y_val + offset),
                        fontsize=12, ha='center', va='bottom',
                        xytext=(0, 5), textcoords='offset points',
                        color='darkgreen')

    # Right-side Y-axis for dlabel
    if dlabel_values is not None:
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(y_data)
        ax2.set_yticklabels(dlabel_values, fontsize=12)
        ax2.tick_params(axis='y', which='both', length=0)
        ax2.set_ylabel("Labels", fontsize=12)

    # Fitting
    if args.fit:
        try:
            if args.fit == 'linear':
                popt, _ = curve_fit(linear_func, x_data, y_data)
                y_fit = linear_func(x_data, *popt)
                equation = f'y = {popt[0]:.4f}x + {popt[1]:.4f}'
            elif args.fit == 'quadratic':
                popt, _ = curve_fit(quadratic_func, x_data, y_data)
                y_fit = quadratic_func(x_data, *popt)
                equation = f'y = {popt[0]:.4f}x² + {popt[1]:.4f}x + {popt[2]:.4f}'
            elif args.fit == 'poly':
                coeffs = np.polyfit(x_data, y_data, args.polydeg)
                y_fit = polynomial_func(x_data, *coeffs)
                equation = 'y = ' + ' + '.join([f'{coeffs[i]:.4f}x^{args.polydeg - i}' for i in range(args.polydeg + 1)])
            elif args.fit == 'log':
                popt, _ = curve_fit(log_func, x_data, y_data)
                y_fit = log_func(x_data, *popt)
                equation = f'y = {popt[0]:.4f}log(x) + {popt[1]:.4f}'
            elif args.fit == 'ln':
                popt, _ = curve_fit(ln_func, x_data, y_data)
                y_fit = ln_func(x_data, *popt)
                equation = f'y = {popt[0]:.4f}ln(x) + {popt[1]:.4f}'
            elif args.fit == 'exp':
                popt, _ = curve_fit(exp_func, x_data, y_data, p0=(1, 0.1, 0))
                y_fit = exp_func(x_data, *popt)
                equation = f'y = {popt[0]:.4f}e^({popt[1]:.4f}x) + {popt[2]:.4f}'
            elif args.fit == 'sig':
                popt, _ = curve_fit(sigmoid_func, x_data, y_data, p0=(1, 1, np.mean(x_data), 0))
                y_fit = sigmoid_func(x_data, *popt)
                equation = f'y = {popt[0]:.4f} / (1 + e^(-{popt[1]:.4f}(x-{popt[2]:.4f}))) + {popt[3]:.4f}'
            r2 = r2_score(y_data, y_fit)
            ax.plot(x_data, y_fit, 'r-', label=f'{equation}\nR² = {r2:.4f}')
        except Exception as e:
            print(f"Error during fitting: {e}")

    # Labels and layout
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    if args.title:
        ax.set_title(args.title, fontsize=14)
    ax.grid(True)
    ax.legend()

    # Save or show output
    if args.outdir:
        fname = args.outdir
        if not (fname.endswith(".png") or fname.endswith(".svg")):
            fname += ".png"
        ext = os.path.splitext(fname)[-1].lower()
        dpi = 300 if ext == '.png' else None
        plt.savefig(fname, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to {fname}")
    else:
        plt.show()

if __name__ == '__main__':
    main()
