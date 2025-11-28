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
        filename = f"{base_name}.{extension}"
        full_path = os.path.join(folder, filename)
        if not os.path.exists(full_path):
            return full_path, filename
        filename = f"{base_name}_{counter}.{extension}"
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

def aggregate_intensity_for_target(spectrum, target_mz, mtol,
                                   agg='sum', gauss_sigma=0.5):
    """
    Agrega a intensidade de todos os pontos em uma janela em torno de target_mz.
    - mtol: tolerância (Da)
    - agg: 'sum', 'mean', 'max' ou 'gauss' (peso gaussiano em torno de target_mz)
    - gauss_sigma: fração de mtol usada como sigma no peso gaussiano.
    """
    mz = spectrum.get('m/z array', np.array([]))
    I = spectrum.get('intensity array', np.array([]))

    if len(mz) == 0 or len(I) == 0:
        return 0.0

    da_tol = mtol
    lower = target_mz - da_tol
    upper = target_mz + da_tol

    mask = (mz >= lower) & (mz <= upper)
    if not np.any(mask):
        return 0.0

    mz_w = mz[mask]
    I_w = I[mask]

    if agg == 'max':
        return float(np.max(I_w))
    elif agg == 'mean':
        return float(np.mean(I_w))
    elif agg == 'gauss':
        sigma = max(gauss_sigma * da_tol, 1e-12)
        weights = np.exp(-0.5 * ((mz_w - target_mz) / sigma) ** 2)
        return float(np.sum(I_w * weights))
    else:
        return float(np.sum(I_w))

# -----------------------------
# Curve models for fitting
# -----------------------------

def four_pl(x, A, B, EC50, s):
    """4-parameter logistic (4PL)"""
    return B + (A - B) / (1.0 + (x / EC50) ** s)

def exp_decay(x, A, k, C):
    """Exponential decay: y = A exp(-k x) + C."""
    return A * np.exp(-k * x) + C

def weibull_surv(x, A, lam, k, C):
    """
    Weibull-based survival function (monotonic decay):
    y = A * exp(-(x/lam)^k) + C
    """
    lam = max(lam, 1e-12)
    return A * np.exp(- (x / lam) ** k) + C

def gompertz(x, A, b, c, C):
    """
    Gompertz function for asymmetric decay:
    y = A * exp(-b * exp(c x)) + C
    """
    return A * np.exp(-b * np.exp(c * x)) + C

def gauss_peak(x, A, mu, sigma, C):
    """Simple Gaussian peak: y = C + A * exp(-(x-mu)^2/(2 sigma^2))"""
    sigma = max(sigma, 1e-12)
    return C + A * np.exp(- (x - mu) ** 2 / (2.0 * sigma ** 2))

def lognormal_peak(x, A, mu, sigma, C):
    """Log-normal 'shape' (skewed peak): y = C + A * exp(-(ln x - mu)^2 / (2*sigma^2))"""
    sigma = max(sigma, 1e-12)
    x = np.maximum(x, 1e-12)
    return C + A * np.exp(- (np.log(x) - mu) ** 2 / (2.0 * sigma ** 2))

MODEL_SPECS = {
    '4PL': {
        'fn': four_pl,
        'p0': lambda x, y: [float(np.nanmax(y)), float(np.nanmin(y)), np.nanmedian(x), 3.0],
        'bounds': ([0, -np.inf, 0, 0.1], [np.inf, np.inf, np.inf, 20.0]),
        'latex': lambda p: (
            r"$y= %s + \frac{%s-%s}{1+(x/%s)^{%s}}$"
            % tuple(map(format_equation_param, [p[1], p[0], p[1], p[2], p[3]]))
        )
    },
    'Exponential': {
        'fn': exp_decay,
        'p0': lambda x, y: [float(np.nanmax(y)), 1.0, float(np.nanmin(y))],
        'bounds': ([0.0, 0.0, -np.inf], [np.inf, 10.0, np.inf]),
        'latex': lambda p: (
            r"$y= %s\,e^{-%s x}+%s$" % tuple(map(format_equation_param, p))
        )
    },
    'WeibullSurv': {
        'fn': weibull_surv,
        'p0': lambda x, y: [
            float(np.nanmax(y)),
            max(1e-6, float(np.nanmedian(x))),
            3.0,
            float(np.nanmin(y))
        ],
        'bounds': ([0.0, 1e-6, 0.1, -np.inf], [np.inf, 10.0, 20.0, np.inf]),
        'latex': lambda p: (
            r"$y= %s\,e^{-(x/%s)^{%s}}+%s$"
            % tuple(map(format_equation_param, p))
        )
    },
    'Gompertz': {
        'fn': gompertz,
        'p0': lambda x, y: [
            float(np.nanmax(y)), 1.0, 1.0, float(np.nanmin(y))
        ],
        'bounds': ([0.0, 1e-6, -10.0, -np.inf], [np.inf, 100.0, 10.0, np.inf]),
        'latex': lambda p: (
            r"$y= %s\,e^{-%s e^{%s x}}+%s$"
            % tuple(map(format_equation_param, p))
        )
    },
    'GaussPeak': {
        'fn': gauss_peak,
        'p0': lambda x, y: [
            float(np.nanmax(y) - np.nanmin(y)),
            float(x[np.nanargmax(y)] if np.any(~np.isnan(y)) else np.nanmedian(x)),
            max((np.nanmax(x) - np.nanmin(x)) / 10.0, 1e-6),
            float(np.nanmin(y))
        ],
        'bounds': ([0.0, 0.0, 1e-6, -np.inf], [np.inf, np.inf, 10.0, np.inf]),
        'latex': lambda p: (
            r"$y= %s + %s \exp\left(-\frac{(x-%s)^2}{2 %s^2}\right)$"
            % tuple(map(format_equation_param, [p[3], p[0], p[1], p[2]]))
        )
    },
    'LogNormalPeak': {
        'fn': lognormal_peak,
        'p0': lambda x, y: [
            max(1e-9, float(np.nanmax(y) - np.nanmin(y))),
            float(np.log(max(1e-12, x[np.nanargmax(y)]))),
            0.3,
            float(np.nanmin(y))
        ],
        'bounds': ([0.0, -10.0, 1e-6, -np.inf], [np.inf, 10.0, 5.0, np.inf]),
        'latex': lambda p: (
            r"$y= %s + %s \exp\left(-\frac{(\ln x - %s)^2}{2 %s^2}\right)$"
            % tuple(map(format_equation_param, [p[3], p[0], p[1], p[2]]))
        )
    }
}

def compute_rsquared(y, y_fit):
    ss_res = np.nansum((y - y_fit) ** 2)
    ss_tot = np.nansum((y - np.nanmean(y)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

def compute_aic(y, y_fit, num_params):
    n = np.count_nonzero(~np.isnan(y))
    if n <= num_params:
        return np.inf
    ss_res = np.nansum((y - y_fit) ** 2)
    if ss_res <= 0:
        return np.inf
    return n * np.log(ss_res / n) + 2 * num_params

def fit_all_models(x, y):
    """
    Ajusta todos os modelos em MODEL_SPECS aos dados (x, y)
    e retorna o melhor modelo (menor AIC).
    """
    results = []
    for name, spec in MODEL_SPECS.items():
        fn = spec['fn']
        try:
            p0 = spec['p0'](x, y)
            bounds = spec['bounds']
            params, _ = curve_fit(fn, x, y, p0=p0, bounds=bounds, maxfev=10000)
            y_fit = fn(x, *params)
            r2 = compute_rsquared(y, y_fit)
            aic = compute_aic(y, y_fit, len(params))
            results.append({
                'name': name,
                'params': params,
                'y_fit': y_fit,
                'r2': r2,
                'aic': aic,
                'latex': spec['latex'](params)
            })
        except Exception as e:
            results.append({
                'name': name,
                'error': str(e),
                'ok': False
            })
    ok = [r for r in results if r.get('error') is None]
    if not ok:
        return {'ok': False, 'error': '; '.join([r.get('error', 'fit failed') for r in results])}
    ok.sort(key=lambda r: (r['aic'], -r['r2']))
    return ok[0]

# -----------------------------
# Text rendering fallback helpers
# -----------------------------

def _sanitize_for_plain(s: str) -> str:
    """Remove comandos TeX para um fallback texto simples."""
    s = s.replace('$', '')
    s = s.replace(r'\exp', 'exp').replace(r'\ln', 'ln')
    s = s.replace(r'\left', '').replace(r'\right', '')
    s = s.replace(r'\,', ' ')
    s = re.sub(r'\\[a-zA-Z]+', '', s)
    s = re.sub(r'\s{2,}', ' ', s)
    return s

def safe_axtext(ax, x, y, text, **kwargs):
    """Tenta desenhar TeX no ax; se der erro, usa versão plain."""
    try:
        return ax.text(x, y, text, **kwargs)
    except Exception:
        plain = _sanitize_for_plain(text) + "\n[plain text]"
        return ax.text(x, y, plain, **kwargs)

# -----------------------------
# Core data extraction
# -----------------------------

def process_mzml_files(folder_path, target_ions, use_ce=False, use_com=False,
                       mtol=0.01, agg_mode='sum',
                       gauss_sigma=0.5, precursor_mz=None):
    abs_results = {}
    rel_results = {}

    if precursor_mz is not None:
        precursor_mass = float(precursor_mz)
    else:
        precursor_mass = max(target_ions) if target_ions else 0
        if use_com and precursor_mass > 0:
            print(f"[WARNING] Using max(target_ions)={precursor_mass:.4f} as precursor mass. "
                  f"Use --COM <precursor_mz> for correct CECOM.")

    for filename in os.listdir(folder_path):
        if not (filename.endswith('.mzML') or filename.endswith('.mzml')):
            continue

        hcd_value = extract_hcd_value(filename)
        if hcd_value is None:
            continue

        filepath = os.path.join(folder_path, filename)

        if use_ce:
            ce_value = hcd_to_ce(hcd_value)
            if use_com:
                energy_value = calculate_cecom(ce_value, precursor_mass)
                energy_key = 'CECOM'
            else:
                energy_value = ce_value
                energy_key = 'CE'
        else:
            energy_value = hcd_value
            energy_key = 'HCD'

        if energy_value in abs_results:
            continue

        result_entry = {'HCD': hcd_value}
        if use_ce:
            result_entry['CE'] = ce_value
        if use_ce and use_com:
            result_entry['CECOM'] = energy_value

        abs_results[energy_value] = result_entry.copy()
        rel_results[energy_value] = result_entry.copy()

        abs_intensities = {ion_mz: 0.0 for ion_mz in target_ions}
        total_base_intensity = 0.0

        with mzml.read(filepath) as reader:
            for spectrum in reader:
                if spectrum.get('ms level') != 2:
                    continue

                spec_I = spectrum.get('intensity array', np.array([]))
                spectrum_max_intensity = float(np.max(spec_I)) if len(spec_I) > 0 else 0.0
                total_base_intensity += spectrum_max_intensity

                for ion_mz in target_ions:
                    intensity = aggregate_intensity_for_target(
                        spectrum, ion_mz, mtol,
                        agg=agg_mode, gauss_sigma=gauss_sigma
                    )
                    abs_intensities[ion_mz] += intensity

        for ion_mz in target_ions:
            abs_intensity = abs_intensities.get(ion_mz, 0.0)
            abs_results[energy_value][ion_mz] = abs_intensity
            rel_intensity = (abs_intensity / total_base_intensity * 100.0) if total_base_intensity > 0 else 0.0
            rel_results[energy_value][ion_mz] = rel_intensity

    return abs_results, rel_results, energy_key, precursor_mass

# -----------------------------
# Plotting
# -----------------------------

def show_combined_plot(df, energy_label, y_label_suffix=''):
    fig = plt.figure(figsize=(10, 6))
    ion_columns = [col for col in df.columns if col not in ['HCD', 'CE', 'CECOM']]
    for ion in ion_columns:
        plt.plot(df[energy_label], df[ion], 'o-', label=f'm/z {ion}')
    xlabel = (
        r'CE$_{\mathrm{COM}}$ (eV)' if energy_label == 'CECOM'
        else ('CE (eV)' if energy_label == 'CE' else 'HCD')
    )
    plt.xlabel(xlabel)
    plt.ylabel(f'Intensity{y_label_suffix}')
    plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    plt.grid(True)
    plt.tight_layout()
    return fig

def show_individual_plots(df, energy_label, y_label_suffix='', auto_fit=False):
    """
    Plots individuais por íon.
    - Sem fit: legenda padrão dentro do gráfico.
    - Com --fit:
        * sem legenda
        * curva experimental + curva ajustada
        * nome do modelo no título
        * caixa de equação + R² + AIC abaixo do gráfico
    """
    figs = []
    ion_columns = [col for col in df.columns if col not in ['HCD', 'CE', 'CECOM']]

    for ion in ion_columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        x_data = df[energy_label].values.astype(float)
        y_data = df[ion].values.astype(float)

        y_plot = y_data
        y_label = f'Intensity{y_label_suffix}'

        data_line, = ax.plot(x_data, y_plot, 'o', label=f'Experimental m/z {ion}')

        model_name_for_title = None
        box_text = None

        if auto_fit:
            # margem inferior maior para caber a caixa
            fig.subplots_adjust(bottom=0.3)

            best_model = fit_all_models(x_data, y_data)
            if best_model.get('name'):
                model_name = best_model['name']
                model_name_for_title = model_name
                params = best_model['params']
                y_fit = best_model['y_fit']

                ax.plot(x_data, y_fit, '-', label=f'Best fit ({model_name})')

                box_text = best_model['latex'] + f"\n$R^2 = {best_model['r2']:.4f}$\nAIC = {best_model['aic']:.2f}"

                ec50 = None
                if model_name == 'WeibullSurv':
                    A, lam, k, C = params
                    if lam > 0 and k > 0:
                        ec50 = lam * (np.log(2.0)) ** (1.0 / k)
                elif model_name == '4PL':
                    ec50 = params[2]

                if ec50 is not None and np.isfinite(ec50):
                    box_text += f"\n$EC_{{50}}$ ≈ {format_equation_param(ec50)}"

                # sem legenda nessas figuras de fit
                # caixa de equação abaixo do gráfico (y negativo), centralizada
                if box_text is not None:
                    safe_axtext(
                        ax,
                        0.5,
                        -0.35,
                        box_text,
                        transform=ax.transAxes,
                        fontsize=9,
                        va='center',
                        ha='center',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85),
                        clip_on=False
                    )
            else:
                # se o fit falhar, pelo menos mostra legenda simples dentro
                ax.legend([data_line], [f'Data (m/z {ion})'], loc='upper left')
        else:
            # modo clássico, sem fit: legenda simples dentro
            ax.legend([data_line], [f'Data (m/z {ion})'], loc='upper left')

        xlabel = (
            r'CE$_{\mathrm{COM}}$ (eV)' if energy_label == 'CECOM'
            else ('CE (eV)' if energy_label == 'CE' else 'HCD')
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(y_label)

        base_title = (
            rf'Ion Intensity vs. CE$_{{\mathrm{{COM}}}}$ (m/z {ion})'
            if energy_label == 'CECOM'
            else f'Ion Intensity vs. {energy_label} (m/z {ion})'
        )

        if auto_fit and model_name_for_title:
            ax.set_title(f'{base_title} - {model_name_for_title} fit')
        else:
            ax.set_title(base_title)

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
    parser.add_argument('--csv', action='store_true',
                        help='Save results to CSV (default: do not save)')
    parser.add_argument('--r', action='store_true',
                        help='Use relative intensities instead of absolute')
    parser.add_argument('--fit', action='store_true',
                        help='Auto-fit best model on individual ion plots')
    parser.add_argument('--CE', action='store_true',
                        help='Convert HCD values to Collision Energy (eV)')
    parser.add_argument(
        '--COM',
        type=float,
        help='Use Center-of-Mass collision energy (requires --CE); '
             'value is precursor m/z, e.g. --COM 632.2'
    )
    parser.add_argument('--tog',
                        help='Only plot the combined graph for specified ion masses (comma-separated)')

    parser.add_argument('--mtol', type=float, default=0.01,
                        help='Mass tolerance for peak aggregation (Da) [default: 0.01]')
    parser.add_argument('--agg', choices=['sum', 'mean', 'max', 'gauss'],
                        default='sum',
                        help='Aggregation mode within tolerance window [default: sum]')
    parser.add_argument('--gauss-sigma', type=float, default=0.5,
                        help='Relative width (fraction of --mtol) for Gaussian weighting when --agg gauss [default: 0.5]')

    args = parser.parse_args()

    # COM logic: if COM is used, CE must be used, and COM holds precursor m/z (float)
    if args.COM is not None and not args.CE:
        parser.error("--COM requires --CE to be specified (COM is computed from CE).")

    if args.tog:
        # When --tog is used, only combined plot is produced, no fit/individual
        args.fit = False
        ions_for_plotting = [float(mass.strip()) for mass in args.tog.split(',')]
        full_ion_list = [float(mass.strip()) for mass in args.ions.split(',')]
        target_ions = full_ion_list
    else:
        ions_for_plotting = None
        target_ions = [float(mass.strip()) for mass in args.ions.split(',')]

    if not target_ions:
        parser.error("At least one ion mass must be specified")

    use_com = args.COM is not None
    precursor_mz = args.COM if use_com else None

    abs_results, rel_results, energy_label, precursor_mass = process_mzml_files(
        args.folder_path, target_ions, args.CE, use_com,
        mtol=args.mtol,
        agg_mode=args.agg, gauss_sigma=args.gauss_sigma,
        precursor_mz=precursor_mz
    )

    if not abs_results:
        print("No valid .mzML files with HCD values found.")
        return

    abs_df = pd.DataFrame.from_dict(abs_results, orient='index')
    abs_df.sort_values(energy_label, inplace=True)

    rel_df = pd.DataFrame.from_dict(rel_results, orient='index')
    rel_df.sort_values(energy_label, inplace=True)

    if use_com:
        energy_cols = ['HCD', 'CE', 'CECOM']
    elif args.CE:
        energy_cols = ['HCD', 'CE']
    else:
        energy_cols = ['HCD']

    other_cols = [col for col in abs_df.columns if col not in energy_cols]
    abs_df = abs_df[energy_cols + other_cols]
    rel_df = rel_df[energy_cols + other_cols]

    if args.csv:
        output_folder = 'SYF_output'
        os.makedirs(output_folder, exist_ok=True)
        output_basename = f'ion_intensities_by_{energy_label.lower()}{"_relative" if args.r else ""}'
        csv_path, csv_filename = get_unique_filename(output_folder, output_basename, 'csv')
        (rel_df if args.r else abs_df).to_csv(csv_path, index=False)
        print(f"Results saved to {csv_filename}")

    if use_com:
        print(f"Precursor mass used for CECOM calculation: {precursor_mass:.4f} m/z")

    plot_df = rel_df if args.r else abs_df
    y_label_suffix = ' (Relative)' if args.r else ' (Absolute)'

    if args.tog:
        selected_cols = energy_cols + [ion for ion in ions_for_plotting if ion in plot_df.columns]
        plot_df = plot_df[selected_cols]
        show_combined_plot(plot_df, energy_label, y_label_suffix)
    else:
        show_combined_plot(plot_df, energy_label, y_label_suffix)

        # Plots individuais:
        # - se tiver mais de um íon, ou
        # - se --fit foi requisitado (mesmo para 1 íon)
        ion_columns = [col for col in plot_df.columns if col not in energy_cols]
        if len(ion_columns) > 1 or args.fit:
            show_individual_plots(plot_df, energy_label, y_label_suffix, auto_fit=args.fit)

    plt.show()

if __name__ == "__main__":
    main()

