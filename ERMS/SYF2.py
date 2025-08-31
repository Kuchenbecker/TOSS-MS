import os
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from pyteomics import mzml

# -----------------------------
# Utilities
# -----------------------------

def get_unique_filename(folder, base_name, extension):
    counter = 1
    while True:
        filename = f"{base_name}.{extension}" if counter == 1 else f"{base_name}_{counter}.{extension}"
        full_path = os.path.join(folder, filename)
        if not os.path.exists(full_path):
            return full_path, filename
        counter += 1

def hcd_to_ce(hcd_value):
    return 0.1742 * hcd_value + 3.8701

def calculate_cecom(ce_value, precursor_mass):
    return ce_value * (28.0134 / (precursor_mass + 28.0134))

def format_equation_param(value):
    return f"{value:.4g}".replace("--", "-")

def extract_hcd_value(filename):
    match = re.search(r'HCD(\d+)', filename, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None

# -----------------------------
# Peak aggregation with tolerance
# -----------------------------

def aggregate_intensity_for_target(spectrum, target_mz, mtol, use_ppm=False,
                                   agg='sum', gauss_sigma_frac=0.5):
    """
    Agrega a intensidade de todos os pontos em uma janela em torno de target_mz.
    - mtol: tolerância (Da por padrão; ppm se use_ppm=True)
    - agg: 'sum' (padrão), 'mean', 'max', 'gauss' (ponderação gaussiana centrada no alvo)
    - gauss_sigma_frac: fração de mtol usada como sigma do peso gaussiano (para 'gauss')
    """
    mz = spectrum.get('m/z array', np.array([]))
    I = spectrum.get('intensity array', np.array([]))
    if mz is None or I is None or len(mz) == 0:
        return 0.0

    tol = (target_mz * mtol / 1e6) if use_ppm else mtol
    if tol <= 0:
        return 0.0

    lo, hi = target_mz - tol, target_mz + tol
    mask = (mz >= lo) & (mz <= hi)
    if not np.any(mask):
        return 0.0

    mz_w = mz[mask]
    I_w = I[mask].astype(float)

    if agg == 'max':
        return float(np.max(I_w))
    elif agg == 'mean':
        return float(np.mean(I_w))
    elif agg == 'gauss':
        sigma = max(1e-12, gauss_sigma_frac * tol)
        w = np.exp(-0.5 * ((mz_w - target_mz) / sigma) ** 2)
        num = float(np.sum(w * I_w))
        den = float(np.sum(w))
        return num / den if den > 0 else 0.0
    else:
        # 'sum' padrão
        return float(np.sum(I_w))

# -----------------------------
# Model definitions for --fit
# -----------------------------

def four_pl(x, A1, A2, x0, p):
    """Decreasing 4-parameter logistic (Hill-type).
    y = A2 + (A1 - A2) / (1 + (x/x0)**p)
    """
    return A2 + (A1 - A2) / (1 + (x / x0) ** p)

def exp_decay(x, A, k, C):
    """Simple exponential decay to asymptote C: y = A * exp(-k x) + C"""
    return A * np.exp(-k * x) + C

def weibull_surv(x, A, lam, k, C):
    """Weibull survival-like decay: y = A * exp(-(x/lam)**k) + C"""
    return A * np.exp(- (x / lam) ** k) + C

def gompertz(x, A, b, c, C):
    """Gompertz-type decay: y = A * exp(-b * c**x) + C"""
    return A * np.exp(-b * (c ** x)) + C

# NEW: peak-shaped models
def gaussian_peak(x, A, mu, sigma, C):
    """Gaussian peak in energy: y = C + A * exp(-(x-mu)^2 / (2*sigma^2))"""
    sigma = max(sigma, 1e-12)
    return C + A * np.exp(- (x - mu) ** 2 / (2.0 * sigma ** 2))

def lognormal_peak(x, A, mu, sigma, C):
    """Log-normal 'shape' (skewed peak): y = C + A * exp(-(ln x - mu)^2 / (2*sigma^2))"""
    sigma = max(sigma, 1e-12)
    x = np.maximum(x, 1e-12)  # estabilidade numérica
    return C + A * np.exp(- (np.log(x) - mu) ** 2 / (2.0 * sigma ** 2))

MODEL_SPECS = {
    '4PL': {
        'fn': four_pl,
        'p0': lambda x, y: [float(np.nanmax(y)), float(np.nanmin(y)), np.nanmedian(x), 3.0],
        'bounds': ([0, -np.inf, 0, 0.1], [np.inf, np.inf, np.inf, 20.0]),
        'latex': lambda p: (r"$y= %s + \frac{%s-%s}{1+(x/%s)^{%s}}$"
                            % tuple(map(format_equation_param, [p[1], p[0], p[1], p[2], p[3]])))
    },
    'Exponential': {
        'fn': exp_decay,
        'p0': lambda x, y: [float(np.nanmax(y)), 1.0, float(np.nanmin(y))],
        'bounds': ([0.0, 0.0, -np.inf], [np.inf, 10.0, np.inf]),
        'latex': lambda p: (r"$y= %s\,e^{-%s x}+%s$" % tuple(map(format_equation_param, p)))
    },
    'WeibullSurv': {
        'fn': weibull_surv,
        'p0': lambda x, y: [float(np.nanmax(y)), max(1e-6, float(np.nanmedian(x))), 3.0, float(np.nanmin(y))],
        'bounds': ([0.0, 1e-6, 0.1, -np.inf], [np.inf, 10.0, 20.0, np.inf]),
        'latex': lambda p: (r"$y= %s\,e^{-(x/%s)^{%s}}+%s$" % tuple(map(format_equation_param, p)))
    },
    'Gompertz': {
        'fn': gompertz,
        'p0': lambda x, y: [float(np.nanmax(y)), 0.5, 1.5, float(np.nanmin(y))],
        'bounds': ([0.0, 1e-3, 1e-3, -np.inf], [np.inf, 10.0, 10.0, np.inf]),
        'latex': lambda p: (r"$y= %s\,e^{-%s\,%s^{x}}+%s$" % tuple(map(format_equation_param, p)))
    },
    # NEW models in --fit
    'GaussianPeak': {
        'fn': gaussian_peak,
        'p0': lambda x, y: [
            max(1e-9, float(np.nanmax(y) - np.nanmin(y))),                  # A
            float(x[np.nanargmax(y)]) if np.any(np.isfinite(y)) else float(np.nanmedian(x)),  # mu
            max(1e-6, float((np.nanmax(x) - np.nanmin(x)) / 6.0)),         # sigma
            float(np.nanmin(y))                                            # C
        ],
        'bounds': ([0.0, 0.0, 1e-6, -np.inf], [np.inf, np.inf, 10.0, np.inf]),
        'latex': lambda p: (r"$y= %s + %s\,e^{-\frac{(x-%s)^2}{2\,%s^2}}$"
                            % tuple(map(format_equation_param, [p[3], p[0], p[1], p[2]])))
    },
    'LogNormalPeak': {
        'fn': lognormal_peak,
        'p0': lambda x, y: [
            max(1e-9, float(np.nanmax(y) - np.nanmin(y))),     # A
            float(np.log(max(1e-12, x[np.nanargmax(y)]))),     # mu (em ln x)
            0.3,                                               # sigma
            float(np.nanmin(y))                                # C
        ],
        'bounds': ([0.0, -10.0, 1e-6, -np.inf], [np.inf, 10.0, 5.0, np.inf]),
        'latex': lambda p: (r"$y= %s + %s\,\exp\!\Big(-\frac{(\ln x - %s)^2}{2\,%s^2}\Big)$"
                            % tuple(map(format_equation_param, [p[3], p[0], p[1], p[2]])))
    }
}

def fit_and_score(x, y, model_key):
    spec = MODEL_SPECS[model_key]
    fn = spec['fn']
    p0 = spec['p0'](x, y)
    bounds = spec['bounds']
    try:
        popt, _ = curve_fit(fn, x, y, p0=p0, bounds=bounds, maxfev=20000)
        yhat = fn(x, *popt)
        resid = y - yhat
        rss = float(np.nansum(resid ** 2))
        n = int(np.sum(~np.isnan(y)))
        k = len(popt)
        sst = float(np.nansum((y - np.nanmean(y)) ** 2)) if n > 0 else np.nan
        r2 = 1 - rss / sst if sst > 0 else np.nan
        aic = n * np.log(rss / n) + 2 * k if n > k and rss > 0 else np.inf
        return {
            'ok': True,
            'model': model_key,
            'popt': popt,
            'r2': r2,
            'aic': aic,
            'rss': rss,
            'latex': spec['latex'](popt)
        }
    except Exception as e:
        return {'ok': False, 'model': model_key, 'error': str(e)}

def choose_best_model(x, y):
    results = [fit_and_score(x, y, m) for m in MODEL_SPECS.keys()]
    ok = [r for r in results if r['ok']]
    if not ok:
        return {'ok': False, 'error': '; '.join([r.get('error', 'fit failed') for r in results])}
    ok.sort(key=lambda r: (r['aic'], -r['r2']))
    return ok[0]

# -----------------------------
# Core data extraction
# -----------------------------

def process_mzml_files(folder_path, target_ions, use_ce=False, use_com=False,
                       mtol=0.01, ppm_mode=False, agg_mode='sum', gauss_sigma=0.5):
    """
    Lê todos os .mzML com padrão HCDXX e agrega intensidades dos íons-alvo com tolerância.
    Retorna dicionários para intensidades absolutas e relativas e o rótulo de energia (HCD/CE/CECOM).
    """
    abs_results = {}
    rel_results = {}
    precursor_mass = max(target_ions) if target_ions else 0

    for filename in os.listdir(folder_path):
        if not (filename.endswith('.mzML') or filename.endswith('.mzml')):
            continue

        hcd_value = extract_hcd_value(filename)
        if hcd_value is None:
            continue

        ce_value = hcd_to_ce(hcd_value) if use_ce else hcd_value
        energy_value = calculate_cecom(ce_value, precursor_mass) if (use_ce and use_com) else ce_value
        energy_key = 'CECOM' if (use_ce and use_com) else ('CE' if use_ce else 'HCD')

        filepath = os.path.join(folder_path, filename)
        print(f"Processing {filename} ({energy_key}{energy_value:.1f} eV)...")

        result_entry = {
            'HCD': hcd_value,
            'CE': ce_value if use_ce else None,
            'CECOM': calculate_cecom(ce_value, precursor_mass) if (use_ce and use_com) else None,
            energy_key: energy_value
        }

        abs_results[energy_value] = result_entry.copy()
        rel_results[energy_value] = result_entry.copy()

        abs_intensities = {ion_mz: 0.0 for ion_mz in target_ions}
        total_base_intensity = 0.0

        with mzml.read(filepath) as reader:
            for spectrum in reader:
                if spectrum.get('ms level') != 2:
                    continue

                # base peak de cada espectro para normalização relativa
                spec_I = spectrum.get('intensity array', np.array([]))
                spectrum_max_intensity = float(np.max(spec_I)) if len(spec_I) > 0 else 0.0
                total_base_intensity += spectrum_max_intensity

                # agrega cada íon na janela especificada
                for ion_mz in target_ions:
                    intensity = aggregate_intensity_for_target(
                        spectrum, ion_mz, mtol, use_ppm=ppm_mode,
                        agg=agg_mode, gauss_sigma_frac=gauss_sigma
                    )
                    abs_intensities[ion_mz] += intensity

        for ion_mz in target_ions:
            abs_intensity = abs_intensities.get(ion_mz, 0.0)
            abs_results[energy_value][ion_mz] = abs_intensity
            rel_intensity = (abs_intensity / total_base_intensity * 100.0) if total_base_intensity > 0 else 0.0
            rel_results[energy_value][ion_mz] = rel_intensity

    return abs_results, rel_results, energy_key, precursor_mass

# -----------------------------
# Plotting (SHOW only; no auto-save)
# -----------------------------

def show_combined_plot(df, energy_label, y_label_suffix=''):
    fig = plt.figure(figsize=(10, 6))
    ion_columns = [col for col in df.columns if col not in ['HCD', 'CE', 'CECOM']]
    for ion in ion_columns:
        plt.plot(df[energy_label], df[ion], 'o-', label=f'm/z {ion}')
    xlabel = r'CE$_{\mathrm{COM}}$ (eV)' if energy_label == 'CECOM' else ('CE (eV)' if energy_label == 'CE' else 'HCD')
    plt.xlabel(xlabel)
    plt.ylabel(f'Intensity{y_label_suffix}')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout()
    return fig

def show_individual_plots(df, energy_label, y_label_suffix='', normalize=False, auto_fit=False):
    figs = []
    ion_columns = [col for col in df.columns if col not in ['HCD', 'CE', 'CECOM']]
    for ion in ion_columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        x_data = df[energy_label].values.astype(float)
        y_data = df[ion].values.astype(float)

        # Normalização apenas visual
        if normalize:
            max_intensity = np.nanmax(y_data)
            if max_intensity > 0:
                y_plot = y_data / max_intensity * 100.0
                y_label = f'Normalized Intensity{y_label_suffix} (%)'
            else:
                y_plot = y_data.copy()
                y_label = f'Intensity{y_label_suffix}'
        else:
            y_plot = y_data.copy()
            y_label = f'Intensity{y_label_suffix}'

        data_line, = ax.plot(x_data, y_plot, 'o', label='Data')

        fit_line = None

        if auto_fit:
            best = choose_best_model(x_data, y_data)
            if best.get('ok', False):
                x_fit = np.linspace(np.nanmin(x_data), np.nanmax(x_data), 300)
                y_fit_raw = MODEL_SPECS[best['model']]['fn'](x_fit, *best['popt'])
                if normalize and np.nanmax(y_data) > 0:
                    y_fit = y_fit_raw / np.nanmax(y_data) * 100.0
                else:
                    y_fit = y_fit_raw

                fit_line, = ax.plot(x_fit, y_fit, '-', label=f"{best['model']} fit")

                # EC50 apenas quando aplicável
                ec50 = None
                if best['model'] == 'WeibullSurv':
                    A, lam, k, C = best['popt']
                    if lam > 0 and k > 0:
                        ec50 = lam * (np.log(2.0)) ** (1.0 / k)
                elif best['model'] == '4PL':
                    ec50 = best['popt'][2]

                # Texto da equação/estatísticas
                eq_text = best['latex'] + ("\n$R^2$ = %.4f" % best['r2']) + ("\nAIC = %.2f" % best['aic'])
                if ec50 is not None and np.isfinite(ec50):
                    eq_text += ("\n$EC_{50}$ ≈ %s" % format_equation_param(ec50))

                # >>>> LEGENDA E EQUAÇÃO FORA, À DIREITA <<<<
                plt.subplots_adjust(left=0.12, right=0.65)

                fig.text(0.98, 0.5, eq_text,
                         fontsize=9, va='center', ha='right',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

                legend_handles = [data_line] + ([fit_line] if fit_line is not None else [])
                legend_labels = ['Data'] + ([f"{best['model']} fit"] if fit_line is not None else [])
                fig.legend(legend_handles, legend_labels,
                           loc='center right', bbox_to_anchor=(0.98, 0.8), frameon=True)
            else:
                # Falha no fit
                plt.subplots_adjust(left=0.12, right=0.65)
                fig.text(0.98, 0.5, f"Fit failed: {best.get('error','unknown')}",
                         fontsize=9, va='center', ha='right',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
                fig.legend([data_line], ['Data'],
                           loc='center right', bbox_to_anchor=(0.98, 0.8), frameon=True)

        # Quando NÃO usa --fit, mantém legenda interna
        if not auto_fit:
            ax.legend(loc='upper left')

        xlabel = r'CE$_{\mathrm{COM}}$ (eV)' if energy_label == 'CECOM' else ('CE (eV)' if energy_label == 'CE' else 'HCD')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(y_label)
        title = rf'Ion Intensity vs. CE$_{{\mathrm{{COM}}}}$ (m/z {ion})' if energy_label == 'CECOM' else f'Ion Intensity vs. {energy_label} (m/z {ion})'
        ax.set_title(title)
        ax.grid(True)

        figs.append(fig)
    return figs

# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Extract ion intensities from .mzML files and plot them. Optionally save CSV.'
    )
    parser.add_argument('folder_path', help='Path to the folder containing .mzML files')
    parser.add_argument('ions', help='List of target ion masses (comma-separated)')
    # REMOVED --plot
    parser.add_argument('--csv', action='store_true',
                        help='Save results to CSV (default: do not save)')
    parser.add_argument('--r', action='store_true',
                        help='Use relative intensities instead of absolute')
    parser.add_argument('--s', action='store_true',
                        help='Open a separate graph window for each ion (in addition to combined plot)')
    parser.add_argument('--n', action='store_true',
                        help='Normalize each ion to 100% in separate graphs (requires --s)')
    parser.add_argument('--fit', action='store_true',
                        help='Auto-fit best model (includes GaussianPeak and LogNormalPeak) on individual ion plots (requires --s)')
    parser.add_argument('--CE', action='store_true',
                        help='Convert HCD values to Collision Energy (eV)')
    parser.add_argument('--COM', action='store_true',
                        help='Use Center of Mass collision energy (requires --CE)')
    parser.add_argument('--tog',
                        help='Only plot the combined graph for specified ion masses (comma-separated)')

    # Mass tolerance options
    parser.add_argument('--mtol', type=float, default=0.01,
                        help='Mass tolerance for peak aggregation (Da by default) [default: 0.01]')
    parser.add_argument('--ppm', action='store_true',
                        help='Interpret --mtol as ppm instead of Da')
    parser.add_argument('--agg', choices=['sum', 'mean', 'max', 'gauss'],
                        default='sum', help='Aggregation mode within tolerance window [default: sum]')
    parser.add_argument('--gauss-sigma', type=float, default=0.5,
                        help='Relative width (fraction of --mtol) for Gaussian weighting when --agg gauss [default: 0.5]')

    args = parser.parse_args()

    if args.n and not args.s:
        parser.error("--n requires --s to be specified")
    if args.fit and not args.s:
        parser.error("--fit requires --s to be specified")
    if args.COM and not args.CE:
        parser.error("--COM requires --CE to be specified")

    if args.tog:
        args.s = False
        args.n = False
        args.fit = False
        ions_for_plotting = [float(mass.strip()) for mass in args.tog.split(',')]
        full_ion_list = [float(mass.strip()) for mass in args.ions.split(',')]
        target_ions = full_ion_list
    else:
        ions_for_plotting = None
        target_ions = [float(mass.strip()) for mass in args.ions.split(',')]

    if not target_ions:
        parser.error("At least one ion mass must be specified")

    abs_results, rel_results, energy_label, precursor_mass = process_mzml_files(
        args.folder_path, target_ions, args.CE, args.COM,
        mtol=args.mtol, ppm_mode=args.ppm, agg_mode=args.agg, gauss_sigma=args.gauss_sigma
    )

    if not abs_results:
        print("No valid .mzML files with HCD values found.")
        return

    abs_df = pd.DataFrame.from_dict(abs_results, orient='index')
    abs_df.sort_values(energy_label, inplace=True)

    rel_df = pd.DataFrame.from_dict(rel_results, orient='index')
    rel_df.sort_values(energy_label, inplace=True)

    energy_cols = ['HCD', 'CE', 'CECOM'] if args.COM else (['HCD', 'CE'] if args.CE else ['HCD'])
    other_cols = [col for col in abs_df.columns if col not in energy_cols]
    abs_df = abs_df[energy_cols + other_cols]
    rel_df = rel_df[energy_cols + other_cols]

    # Save CSV only if requested
    if args.csv:
        output_folder = 'SYF_output'
        os.makedirs(output_folder, exist_ok=True)
        output_basename = f'ion_intensities_by_{energy_label.lower()}{"_relative" if args.r else ""}'
        csv_path, csv_filename = get_unique_filename(output_folder, output_basename, 'csv')
        (rel_df if args.r else abs_df).to_csv(csv_path, index=False)
        print(f"Results saved to {csv_filename}")

    print(f"Precursor mass used for CECOM calculation: {precursor_mass:.4f} m/z")

    # Always plot (open windows)
    plot_df = rel_df if args.r else abs_df
    y_label_suffix = ' (Relative)' if args.r else ' (Absolute)'

    if args.tog:
        selected_cols = energy_cols + [ion for ion in ions_for_plotting if ion in plot_df.columns]
        plot_df = plot_df[selected_cols]
        show_combined_plot(plot_df, energy_label, y_label_suffix)
    else:
        show_combined_plot(plot_df, energy_label, y_label_suffix)
        if args.s:
            show_individual_plots(plot_df, energy_label, y_label_suffix, args.n, args.fit)

    plt.show()

if __name__ == "__main__":
    main()
