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

# ---------- Helper: load & plot one CSV ----------
def _load_xy_from_csv(path, x_col_idx, y_col_idx):
    df = pd.read_csv(path)
    if len(df.columns) < 2:
        raise ValueError(f"{path}: CSV must have at least 2 columns.")
    x = pd.to_numeric(df.iloc[:, x_col_idx], errors="coerce").to_numpy()
    y = pd.to_numeric(df.iloc[:, y_col_idx], errors="coerce").to_numpy()
    m = np.isfinite(x) & np.isfinite(y)
    return df, x[m], y[m]

def _plot_single_series(ax, x, y, label, show_vals=False, connect=False, color=None):
    sc = ax.scatter(x, y, label=label, color=color)
    if connect and len(x) > 1:
        order = np.argsort(x)
        ax.plot(x[order], y[order], linestyle='-', marker=None, label=f"{label} (connected)", color=color)
    if show_vals and len(y) > 0:
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
    return sc

def main():
    parser = argparse.ArgumentParser(description='Plot CSV data with optional curve fitting.')
    # filename becomes optional if --multi is used
    parser.add_argument('filename', nargs='?', help='Path to the CSV file')
    parser.add_argument('--multi', type=str,
                        help='Comma-separated list of CSVs to overlay (e.g., a.csv,b.csv,c.csv)')
    parser.add_argument('--fit', choices=[
        'linear','quadratic','poly','log','ln','exp','sig',
        '4pl','expdecay','weibull','gompertz','gauss','lognormal'
    ], help='Type of function to fit (applied to each series independently)')
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

    # --- Determine input mode
    series_files = None
    if args.multi:
        # split by comma and strip whitespace
        series_files = [p.strip() for p in args.multi.split(',') if p.strip()]
        if len(series_files) == 0:
            print("Error: --multi provided but no CSV paths parsed.")
            return
    elif args.filename:
        series_files = [args.filename]
    else:
        print("Error: provide a CSV filename or use --multi with a list of CSVs.")
        return

    # --- We'll use the SAME column selections for all series
    # Resolve column indexes later per-file after reading first file to name axes
    # Provisional labels
    x_label = args.xlabel
    y_label = args.ylabel

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(right=0.78, top=0.82)

    # plot each series
    banners = []  # one per series if fitting
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', None)

    for idx, path in enumerate(series_files):
        try:
            # Read dataframe to get column names
            df = pd.read_csv(path)
        except Exception as e:
            print(f"Error reading '{path}': {e}")
            continue

        # Resolve columns indices
        y_col = args.yaxis - 1
        if not (0 <= y_col < len(df.columns)):
            print(f"Error: yaxis {args.yaxis} out of range for '{path}'.")
            continue
        if args.xaxis is not None:
            x_col = args.xaxis - 1
            if not (0 <= x_col < len(df.columns)):
                print(f"Error: xaxis {args.xaxis} out of range for '{path}'.")
                continue
            if x_col == y_col:
                print(f"Error: --xaxis and --yaxis refer to the same column in '{path}'.")
                continue
        else:
            x_col = 0 if y_col != 0 else (1 if len(df.columns) > 1 else 0)

        # Labels (use first file to set default axis labels if not provided)
        if x_label is None: x_label = str(df.columns[x_col])
        if y_label is None: y_label = str(df.columns[y_col])

        # Extract numeric arrays with same masking logic
        x = pd.to_numeric(df.iloc[:, x_col], errors="coerce").to_numpy()
        y = pd.to_numeric(df.iloc[:, y_col], errors="coerce").to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

        color = None if color_cycle is None else color_cycle[idx % len(color_cycle)]
        label = os.path.splitext(os.path.basename(path))[0]

        _plot_single_series(ax, x, y, label=label, show_vals=args.show, connect=args.connect, color=color)

        # Right-axis labels (desaconselhado com multi, mas suportado se houver apenas 1 série)
        if args.dlabel and len(series_files) == 1 and str(args.dlabel).isdigit():
            dlabel_col = int(args.dlabel) - 1
            if 0 <= dlabel_col < len(df.columns):
                dvals = df.iloc[:, dlabel_col].astype(str).to_numpy()[mask]
                ax2 = ax.twinx()
                ax2.set_ylim(ax.get_ylim())
                ax2.set_yticks(y)
                ax2.set_yticklabels(dvals, fontsize=label_fontsize)
                ax2.tick_params(axis='y', which='both', length=0)
                ax2.set_ylabel("Labels", fontsize=label_fontsize)
            else:
                print(f"Warning: dlabel column {args.dlabel} out of range for '{path}'")

        # Optional fitting per series (independent)
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

                # plot fitted curve for the series with matching color
                ax.plot(xx, y_fit, '-', label=f'{label} (fit: {args.fit})', color=color)

                # R² only printed to console for each series
                valid = np.isfinite(y_fit)
                r2 = r2_score(yy[valid], y_fit[valid])
                fit_key = 'poly' if fit == 'poly' else fit
                eq_tex = eq_text_tex(fit_key, popt, degree=(args.polydeg if fit == 'poly' else None))
                print(f"[{label}] Equation:", eq_tex)
                print(f"[{label}] R^2: {r2:.6f}")

                # If there is a single series, keep the top banner like before.
                if len(series_files) == 1:
                    banners.append(rf"${eq_tex}$" + "\n" + rf"$R^2 = {r2:.4f}$")
            except Exception as e:
                print(f"Error during fitting for '{path}': {e}")

    # Labels/title/grid
    ax.set_xlabel(x_label if x_label else "X", fontsize=label_fontsize)
    ax.set_ylabel(y_label if y_label else "Y", fontsize=label_fontsize)
    if args.title:
        ax.set_title(args.title, fontsize=title_fontsize)
    ax.grid(True)

    # Legend outside right-center
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc='center left',
                  bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, frameon=True)

    # Top banner only for o modo single (para não poluir quando há muitas séries)
    if len(banners) == 1:
        fig.text(0.5, 0.98, banners[0], ha='center', va='top',
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

