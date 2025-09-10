import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import os

# ---- Configurable font sizes ----
eq_fontsize = 20   # equation/R² text drawn above the plot (LaTeX-like)
label_fontsize = 12
title_fontsize = 14
# ---------------------------------

# ===== Basic models =====
def linear_func(x, a, b): return a * x + b
def quadratic_func(x, a, b, c): return a * x**2 + b * x + c
def polynomial_func(x, *coeffs): return np.polyval(coeffs, x)

def log_func(x, a, b):
    x = np.where(x <= 0, np.nan, x)
    return a * np.log10(x) + b

def ln_func(x, a, b):
    x = np.where(x <= 0, np.nan, x)
    return a * np.log(x) + b

def exp_func(x, a, b, c): return a * np.exp(b * x) + c

def sigmoid_func(x, a, b, c, d):
    return a / (1 + np.exp(-b * (x - c))) + d

# ===== Extra models =====
def four_pl(x, A1, A2, x0, p):
    """4-parameter logistic (decreasing)"""
    return A2 + (A1 - A2) / (1.0 + (x / x0) ** p)

def exp_decay(x, A, k, C):
    """Exponential decay to asymptote C"""
    return A * np.exp(-k * x) + C

def weibull_surv(x, A, lam, k, C):
    """Weibull-like decay"""
    lam = max(lam, 1e-12)
    return A * np.exp(- (x / lam) ** k) + C

def gompertz(x, A, b, c, C):
    """Gompertz-type decay"""
    c = max(c, 1e-12)
    return A * np.exp(-b * (c ** x)) + C

def gaussian_peak(x, A, mu, sigma, C):
    """Gaussian peak"""
    sigma = max(sigma, 1e-12)
    return C + A * np.exp(- (x - mu) ** 2 / (2.0 * sigma ** 2))

def lognormal_peak(x, A, mu, sigma, C):
    """Log-normal peak (x>0)"""
    sigma = max(sigma, 1e-12)
    x = np.maximum(x, 1e-12)
    return C + A * np.exp(- (np.log(x) - mu) ** 2 / (2.0 * sigma ** 2))

# ---- Heuristic initial guesses ----
def _span(v):
    v = np.asarray(v)
    return float(np.nanmax(v) - np.nanmin(v) if len(v) else 0.0)

def heuristics_xy(x, y):
    return (float(np.nanmin(x)), float(np.nanmax(x)), max(_span(x), 1e-12),
            float(np.nanmin(y)), float(np.nanmax(y)), max(_span(y), 1e-12))

def p0_4pl(x, y):
    xmin, xmax, xspan, ymin, ymax, yspan = heuristics_xy(x, y)
    return [ymax, ymin, xmin + 0.5 * xspan, 1.0]

def p0_expdecay(x, y):
    xmin, xmax, xspan, ymin, ymax, yspan = heuristics_xy(x, y)
    C = ymin; A = max(ymax - C, 1e-6); k = 1.0 / max(xspan, 1e-6)
    return [A, k, C]

def p0_weibull(x, y):
    xmin, xmax, xspan, ymin, ymax, yspan = heuristics_xy(x, y)
    C = ymin; A = max(ymax - C, 1e-6); lam = max(xspan / 2.0, 1e-6); k = 1.5
    return [A, lam, k, C]

def p0_gompertz(x, y):
    xmin, xmax, xspan, ymin, ymax, yspan = heuristics_xy(x, y)
    C = ymin; A = max(ymax - C, 1e-6); b = 1.0; c = 1.05
    return [A, b, c, C]

def p0_gauss(x, y):
    xmin, xmax, xspan, ymin, ymax, yspan = heuristics_xy(x, y)
    mu = float(x[np.nanargmax(y)]) if len(x) else 0.0
    sigma = max(float(np.std(x)), xspan / 6.0, 1e-6)
    C = ymin; A = max(ymax - C, 1e-6)
    return [A, mu, sigma, C]

def p0_lognormal(x, y):
    m = x > 0
    xx, yy = x[m], y[m]
    if len(xx) == 0:
        xx, yy = np.array([1.0, 2.0]), np.array([0.0, 0.0])
    xmin, xmax, xspan, ymin, ymax, yspan = heuristics_xy(xx, yy)
    mu = float(np.log(xx[np.nanargmax(yy)]) if len(xx) else 0.0)
    sigma = 1.0; C = ymin; A = max(ymax - C, 1e-6)
    return [A, mu, sigma, C]

# ---- LaTeX-like equation builders (mathtext) ----
def fmt(v, sig=4):
    try:
        s = f"{float(v):.{sig}g}"
        if s.startswith("."): s = "0" + s
        if s.startswith("-."): s = "-0" + s[1:]
        return s
    except Exception:
        return str(v)

def eq_text_tex(name, params, degree=None):
    # Return TeX math string WITHOUT surrounding $...$
    p = [float(v) for v in (params if isinstance(params, (list, tuple, np.ndarray)) else [params])]
    if name == 'linear':
        a, b = p; return rf"y = {fmt(a)}\,x + {fmt(b)}"
    if name == 'quadratic':
        a, b, c = p; return rf"y = {fmt(a)}\,x^2 + {fmt(b)}\,x + {fmt(c)}"
    if name == 'poly':
        coeffs = p; deg = degree
        terms = []
        for i, c in enumerate(coeffs):
            power = deg - i
            if power > 1:
                terms.append(rf"{fmt(c)}\,x^{power}")
            elif power == 1:
                terms.append(rf"{fmt(c)}\,x")
            else:
                terms.append(rf"{fmt(c)}")
        return r"y = " + " + ".join(terms).replace("+ -", "- ")
    if name == 'log':
        a, b = p; return rf"y = {fmt(a)}\,\log_{{10}}(x) + {fmt(b)}"
    if name == 'ln':
        a, b = p; return rf"y = {fmt(a)}\,\ln(x) + {fmt(b)}"
    if name == 'exp':
        a, b, c = p; return rf"y = {fmt(a)}\,e^{{{fmt(b)}\,x}} + {fmt(c)}"
    if name == 'sig':
        a,b,c,d = p; return rf"y = \frac{{{fmt(a)}}}{{1 + e^{{-{fmt(b)}\,(x - {fmt(c)})}}}} + {fmt(d)}"
    if name == '4pl':
        A1,A2,x0,pw = p; return rf"y = {fmt(A2)} + \frac{{{fmt(A1)} - {fmt(A2)}}}{{1 + (x/{fmt(x0)})^{{{fmt(pw)}}}}}"
    if name == 'expdecay':
        A,k,C = p; return rf"y = {fmt(A)}\,e^{{-{fmt(k)}\,x}} + {fmt(C)}"
    if name == 'weibull':
        A,lam,k,C = p; return rf"y = {fmt(A)}\,\exp\!\left(-\left(\frac{{x}}{{{fmt(lam)}}}\right)^{{{fmt(k)}}}\right) + {fmt(C)}"
    if name == 'gompertz':
        A,b,c,C = p; return rf"y = {fmt(A)}\,\exp\!\left(-{fmt(b)}\,{fmt(c)}^x\right) + {fmt(C)}"
    if name == 'gauss':
        A,mu,sig,C = p; return rf"y = {fmt(C)} + {fmt(A)}\,\exp\!\left(-\frac{{(x - {fmt(mu)})^2}}{{2\,{fmt(sig)}^2}}\right)"
    if name == 'lognormal':
        A,mu,sig,C = p; return rf"y = {fmt(C)} + {fmt(A)}\,\exp\!\left(-\frac{{(\ln x - {fmt(mu)})^2}}{{2\,{fmt(sig)}^2}}\right)"
    return "y = f(x)"

def main():
    parser = argparse.ArgumentParser(description='Plot CSV data with optional curve fitting.')
    parser.add_argument('filename', help='Path to the CSV file')
    parser.add_argument('--fit', choices=[
        'linear','quadratic','poly','log','ln','exp','sig',
        '4pl','expdecay','weibull','gompertz','gauss','lognormal'
    ], help='Type of function to fit')
    parser.add_argument('--polydeg', type=int, default=2, help='Degree of polynomial fit')
    parser.add_argument('--outdir', type=str, default=None, help='Path to save output file (supports .png or .svg)')
    parser.add_argument('--title', type=str, help='Title for the plot')
    parser.add_argument('--yaxis', type=int, default=2, help='Column number for y-axis (1-based)')
    parser.add_argument('--xaxis', type=int, default=None, help='Column number for x-axis (1-based)')
    parser.add_argument('--xlabel', type=str, default=None, help='Custom x-axis label')
    parser.add_argument('--ylabel', type=str, default=None, help='Custom y-axis label')
    parser.add_argument('--dlabel', type=str, default=None, help='Column number (1-based) for right-side y-axis labels')
    parser.add_argument('--show', action='store_true', help='Show Y-values above each point')
    parser.add_argument('--connect', action='store_true', help='Connect points with a line (no fitting), ascending X')

    args = parser.parse_args()

    # --- Load data
    try:
        data = pd.read_csv(args.filename)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    if len(data.columns) < 2:
        print("Error: CSV file must have at least 2 columns")
        return

    # --- Resolve columns
    y_col = args.yaxis - 1
    if not (0 <= y_col < len(data.columns)):
        print(f"Error: yaxis value {args.yaxis} is out of range.")
        return

    if args.xaxis is not None:
        x_col = args.xaxis - 1
        if not (0 <= x_col < len(data.columns)):
            print(f"Error: xaxis value {args.xaxis} is out of range.")
            return
        if x_col == y_col:
            print("Error: --xaxis and --yaxis refer to the same column.")
            return
    else:
        x_col = 0 if y_col != 0 else (1 if len(data.columns) > 1 else 0)

    # --- Extract numeric arrays
    x = pd.to_numeric(data.iloc[:, x_col], errors="coerce").to_numpy()
    y = pd.to_numeric(data.iloc[:, y_col], errors="coerce").to_numpy()
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    x_label = args.xlabel if args.xlabel else str(data.columns[x_col])
    y_label = args.ylabel if args.ylabel else str(data.columns[y_col])

    # Optional right-axis labels aligned to Y
    dlabel_values = None
    if args.dlabel and str(args.dlabel).isdigit():
        dlabel_col = int(args.dlabel) - 1
        if 0 <= dlabel_col < len(data.columns):
            dlabel_values = data.iloc[:, dlabel_col].astype(str).to_numpy()[mask]
        else:
            print(f"Warning: dlabel column {args.dlabel} out of range")

    # --- Figure & axes
    fig, ax = plt.subplots(figsize=(10, 6))
    # room for legend (right) and LaTeX banner (top)
    plt.subplots_adjust(right=0.78, top=0.82)

    # Scatter
    ax.scatter(x, y, label='Data')

    # Optional point labels
    if args.show and len(y) > 0:
        sorted_idx = np.argsort(y)
        seen = {}
        y_span = float(np.nanmax(y) - np.nanmin(y)) if len(y) else 1.0
        for idx in sorted_idx:
            yv = y[idx]
            offn = seen.get(yv, 0); off = offn * 0.03 * y_span
            seen[yv] = offn + 1
            ax.annotate(f'{yv:.2f}', (x[idx], yv + off),
                        fontsize=12, ha='center', va='bottom',
                        xytext=(0, 5), textcoords='offset points',
                        color='darkgreen')

    # Right-side Y-axis labels
    if dlabel_values is not None and len(dlabel_values) == len(y):
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(y)
        ax2.set_yticklabels(dlabel_values, fontsize=label_fontsize)
        ax2.tick_params(axis='y', which='both', length=0)
        ax2.set_ylabel("Labels", fontsize=label_fontsize)

    # Connect (no fit)
    if args.connect and len(x) > 1:
        order = np.argsort(x)
        ax.plot(x[order], y[order], linestyle='-', marker=None, label='Connected points')

    # Fit + LaTeX top banner
    banner = None
    if args.fit:
        try:
            fit = args.fit
            xx, yy = x.copy(), y.copy()

            if fit == 'linear':
                popt, _ = curve_fit(linear_func, xx, yy); y_fit = linear_func(xx, *popt)
            elif fit == 'quadratic':
                popt, _ = curve_fit(quadratic_func, xx, yy); y_fit = quadratic_func(xx, *popt)
            elif fit == 'poly':
                popt = np.polyfit(xx, yy, args.polydeg); y_fit = polynomial_func(xx, *popt)
            elif fit == 'log':
                m = xx > 0; popt, _ = curve_fit(log_func, xx[m], yy[m]); y_fit = log_func(xx, *popt)
            elif fit == 'ln':
                m = xx > 0; popt, _ = curve_fit(ln_func, xx[m], yy[m]); y_fit = ln_func(xx, *popt)
            elif fit == 'exp':
                popt, _ = curve_fit(exp_func, xx, yy, p0=(1, 0.1, 0)); y_fit = exp_func(xx, *popt)
            elif fit == 'sig':
                popt, _ = curve_fit(sigmoid_func, xx, yy, p0=(1, 1, np.nanmean(xx), 0)); y_fit = sigmoid_func(xx, *popt)
            elif fit == '4pl':
                p0 = p0_4pl(xx, yy); bounds = ([-np.inf, -np.inf, 1e-9, 1e-6], [np.inf, np.inf, np.inf, np.inf])
                popt, _ = curve_fit(four_pl, xx, yy, p0=p0, bounds=bounds, maxfev=20000); y_fit = four_pl(xx, *popt)
            elif fit == 'expdecay':
                p0 = p0_expdecay(xx, yy); bounds = ([-np.inf, 0.0, -np.inf], [np.inf, np.inf, np.inf])
                popt, _ = curve_fit(exp_decay, xx, yy, p0=p0, bounds=bounds, maxfev=20000); y_fit = exp_decay(xx, *popt)
            elif fit == 'weibull':
                p0 = p0_weibull(xx, yy); bounds = ([-np.inf, 1e-9, 1e-9, -np.inf], [np.inf, np.inf, np.inf, np.inf])
                popt, _ = curve_fit(weibull_surv, xx, yy, p0=p0, bounds=bounds, maxfev=20000); y_fit = weibull_surv(xx, *popt)
            elif fit == 'gompertz':
                p0 = p0_gompertz(xx, yy); bounds = ([-np.inf, 1e-12, 1e-12, -np.inf], [np.inf, np.inf, np.inf, np.inf])
                popt, _ = curve_fit(gompertz, xx, yy, p0=p0, bounds=bounds, maxfev=20000); y_fit = gompertz(xx, *popt)
            elif fit == 'gauss':
                p0 = p0_gauss(xx, yy); bounds = ([-np.inf, -np.inf, 1e-9, -np.inf], [np.inf, np.inf, np.inf, np.inf])
                popt, _ = curve_fit(gaussian_peak, xx, yy, p0=p0, bounds=bounds, maxfev=20000); y_fit = gaussian_peak(xx, *popt)
            elif fit == 'lognormal':
                m = xx > 0; p0 = p0_lognormal(xx[m], yy[m]); bounds = ([-np.inf, -np.inf, 1e-9, -np.inf], [np.inf, np.inf, np.inf, np.inf])
                popt, _ = curve_fit(lognormal_peak, xx[m], yy[m], p0=p0, bounds=bounds, maxfev=20000); y_fit = lognormal_peak(xx, *popt)
            else:
                raise ValueError("Unknown fit type")

            ax.plot(xx, y_fit, 'r-', label='Fitted curve')

            # R² on points used for fit (ignore non-finite y_fit)
            valid = np.isfinite(y_fit)
            r2 = r2_score(yy[valid], y_fit[valid])

            # Build LaTeX-like banner (mathtext)
            fit_key = 'poly' if fit == 'poly' else fit
            eq_tex = eq_text_tex(fit_key, popt, degree=(args.polydeg if fit == 'poly' else None))
            banner = rf"${eq_tex}$" + "\n" + rf"$R^2 = {r2:.4f}$"

            # Also print to console
            print("Equation:", eq_tex)
            print(f"R^2: {r2:.6f}")

        except Exception as e:
            print(f"Error during fitting: {e}")

    # Labels/title/grid
    ax.set_xlabel(x_label, fontsize=label_fontsize)
    ax.set_ylabel(y_label, fontsize=label_fontsize)
    if args.title:
        ax.set_title(args.title, fontsize=title_fontsize)
    ax.grid(True)

    # Legend outside right-center
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc='center left',
                  bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, frameon=True)

    # Top banner (LaTeX equation + R²), above and outside the axes
    if banner:
        fig.text(0.5, 0.98, banner, ha='center', va='top',
                 fontsize=eq_fontsize,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, linewidth=0.5))

    # Save / show
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

