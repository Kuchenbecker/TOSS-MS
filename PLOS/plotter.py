import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# Define fitting functions
def linear_func(x, a, b):
    return a * x + b

def quadratic_func(x, a, b, c):
    return a * x**2 + b * x + c

def polynomial_func(x, *coeffs):
    return np.polyval(coeffs, x)

def log_func(x, a, b):
    return a * np.log10(x) + b

def ln_func(x, a, b):
    return a * np.log(x) + b

def exp_func(x, a, b, c):
    return a * np.exp(b * x) + c

def sigmoid_func(x, a, b, c, d):
    return a / (1 + np.exp(-b * (x - c))) + d

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Plot data from CSV file with optional fitting.')
    parser.add_argument('filename', help='Path to the CSV file')
    parser.add_argument('--fit', choices=['linear', 'quadratic', 'poly', 'log', 'ln', 'exp', 'sig'], 
                       help='Type of function to fit')
    parser.add_argument('--polydeg', type=int, default=2,
                       help='Degree of polynomial fit (only for poly fit)')
    parser.add_argument('--save', action='store_true',
                       help='Save plot as high-resolution PNG')
    parser.add_argument('--title', type=str,
                       help='Title for the plot')
    parser.add_argument('--yaxis', type=int, default=2,
                       help='Column number for y-axis data (1-based index, default: 2)')
    parser.add_argument('--xlabel', type=str, default=None,
                       help='Custom x-axis label (overrides CSV header)')
    parser.add_argument('--ylabel', type=str, default=None,
                       help='Custom y-axis label (overrides CSV header)')
    parser.add_argument('--dlabel', type=str, default=None,
                       help='Label for data points (empty for no label)')
    
    args = parser.parse_args()

    # Read CSV file with headers
    try:
        data = pd.read_csv(args.filename)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Check if we have at least 2 columns
    if len(data.columns) < 2:
        print("Error: CSV file must have at least 2 columns")
        return

    # Extract x and y data
    y_col = args.yaxis - 1  # convert to 0-based index
    x_col = 0 if y_col != 0 else 1  # x is the other column
    
    if y_col >= len(data.columns) or y_col < 0:
        print(f"Error: yaxis value {args.yaxis} is out of range for the CSV columns")
        print(f"Available columns (1-{len(data.columns)}): {list(data.columns)}")
        return
    
    x_data = data.iloc[:, x_col].values
    y_data = data.iloc[:, y_col].values

    # Determine axis labels
    x_label = args.xlabel if args.xlabel is not None else data.columns[x_col]
    y_label = args.ylabel if args.ylabel is not None else data.columns[y_col]
    
    # Determine data label
    data_label = args.dlabel if args.dlabel is not None else 'Data' if args.dlabel != '' else None

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, label=data_label, color='blue')

    # Perform fitting if requested
    if args.fit:
        try:
            if args.fit == 'linear':
                popt, pcov = curve_fit(linear_func, x_data, y_data)
                y_fit = linear_func(x_data, *popt)
                equation = f'y = {popt[0]:.4f}x + {popt[1]:.4f}'
            elif args.fit == 'quadratic':
                popt, pcov = curve_fit(quadratic_func, x_data, y_data)
                y_fit = quadratic_func(x_data, *popt)
                equation = f'y = {popt[0]:.4f}x² + {popt[1]:.4f}x + {popt[2]:.4f}'
            elif args.fit == 'poly':
                coeffs = np.polyfit(x_data, y_data, args.polydeg)
                y_fit = polynomial_func(x_data, *coeffs)
                equation = 'y = ' + ' + '.join([f'{coeffs[i]:.4f}x^{args.polydeg-i}' for i in range(args.polydeg+1)])
            elif args.fit == 'log':
                popt, pcov = curve_fit(log_func, x_data, y_data)
                y_fit = log_func(x_data, *popt)
                equation = f'y = {popt[0]:.4f}log(x) + {popt[1]:.4f}'
            elif args.fit == 'ln':
                popt, pcov = curve_fit(ln_func, x_data, y_data)
                y_fit = ln_func(x_data, *popt)
                equation = f'y = {popt[0]:.4f}ln(x) + {popt[1]:.4f}'
            elif args.fit == 'exp':
                popt, pcov = curve_fit(exp_func, x_data, y_data, p0=(1, 0.1, 0))
                y_fit = exp_func(x_data, *popt)
                equation = f'y = {popt[0]:.4f}e^({popt[1]:.4f}x) + {popt[2]:.4f}'
            elif args.fit == 'sig':
                popt, pcov = curve_fit(sigmoid_func, x_data, y_data, p0=(1, 1, np.mean(x_data), 0))
                y_fit = sigmoid_func(x_data, *popt)
                equation = f'y = {popt[0]:.4f} / (1 + e^(-{popt[1]:.4f}(x-{popt[2]:.4f}))) + {popt[3]:.4f}'
            
            r2 = r2_score(y_data, y_fit)
            plt.plot(x_data, y_fit, 'r-', label=f'Fit: {equation}\nR² = {r2:.4f}')
        except Exception as e:
            print(f"Error during fitting: {e}")
            print("Plotting data without fit.")

    # Add labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if args.title:
        plt.title(args.title)
    
    if data_label is not None or args.fit:
        plt.legend()
    plt.grid(True)

    # Save or show plot
    if args.save:
        output_filename = args.filename.replace('.csv', '_plot.png')
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {output_filename}")
    else:
        plt.show()

if __name__ == '__main__':
    main()