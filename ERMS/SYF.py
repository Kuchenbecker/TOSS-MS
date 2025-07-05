import os
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from pyteomics import mzml

def get_unique_filename(folder, base_name, extension):
    counter = 1
    while True:
        if counter == 1:
            filename = f"{base_name}.{extension}"
        else:
            filename = f"{base_name}_{counter}.{extension}"
        full_path = os.path.join(folder, filename)
        if not os.path.exists(full_path):
            return full_path, filename
        counter += 1

def hcd_to_ce(hcd_value):
    return 0.1742 * hcd_value + 3.8701

def calculate_cecom(ce_value, precursor_mass):
    return ce_value * (28.0134 / (precursor_mass + 28.0134))

def sigmoid(x, L, x0, k, b):
    return L / (1 + np.exp(-k * (x - x0))) + b

def format_equation_param(value):
    return f"{value:.2f}".replace("--", "-")

def extract_hcd_value(filename):
    match = re.search(r'HCD(\d+)', filename)
    return int(match.group(1)) if match else None

def find_closest_peak(spectrum, target_mz, tolerance=0.01):
    mz_array = spectrum['m/z array']
    intensity_array = spectrum['intensity array']
    diffs = abs(mz_array - target_mz)
    min_diff_idx = diffs.argmin()
    if diffs[min_diff_idx] <= tolerance:
        return intensity_array[min_diff_idx]
    return 0

def process_mzml_files(folder_path, target_ions, use_ce=False, use_com=False):
    abs_results = {}
    rel_results = {}
    precursor_mass = max(target_ions) if target_ions else 0

    for filename in os.listdir(folder_path):
        if filename.endswith('.mzML') or filename.endswith('.mzml'):
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

            abs_intensities = {}
            total_base_intensity = 0

            with mzml.read(filepath) as reader:
                for spectrum in reader:
                    if spectrum['ms level'] != 2:
                        continue

                    spectrum_max_intensity = max(spectrum['intensity array']) if len(spectrum['intensity array']) > 0 else 0
                    total_base_intensity += spectrum_max_intensity

                    for ion_mz in target_ions:
                        intensity = find_closest_peak(spectrum, ion_mz)
                        abs_intensities[ion_mz] = abs_intensities.get(ion_mz, 0) + intensity

            for ion_mz in target_ions:
                abs_intensity = abs_intensities.get(ion_mz, 0)
                abs_results[energy_value][ion_mz] = abs_intensity
                rel_intensity = (abs_intensity / total_base_intensity * 100) if total_base_intensity > 0 else 0
                rel_results[energy_value][ion_mz] = rel_intensity

    return abs_results, rel_results, energy_key, precursor_mass

def save_combined_plot(df, energy_label, y_label_suffix='', output_folder='SYF_output'):
    plt.figure(figsize=(10, 6))
    ion_columns = [col for col in df.columns if col not in ['HCD', 'CE', 'CECOM']]
    for ion in ion_columns:
        plt.plot(df[energy_label], df[ion], 'o-', label=f'm/z {ion}')
    xlabel = r'CE$_{\mathrm{COM}}$ (eV)' if energy_label == 'CECOM' else ('CE (eV)' if energy_label == 'CE' else 'HCD')
    plt.xlabel(xlabel)
    plt.ylabel(f'Intensity{y_label_suffix}')
    title = r'Ion Intensities vs. CE$_{\mathrm{COM}}$' if energy_label == 'CECOM' else f'Ion Intensities vs. {energy_label}'
    plt.title(title)
    plt.legend()
    plt.grid(True)
    os.makedirs(output_folder, exist_ok=True)
    plot_basename = f'ion_intensities_combined_{energy_label.lower()}{y_label_suffix.replace(" ", "_").lower()}'
    svg_path, svg_filename = get_unique_filename(output_folder, plot_basename, 'svg')
    plt.savefig(svg_path, bbox_inches='tight')
    print(f"Combined plot saved to {svg_filename}")
    plt.close()

def save_individual_plots(df, energy_label, y_label_suffix='', normalize=False, sigmoidal_fit=False, output_folder='SYF_output'):
    ion_columns = [col for col in df.columns if col not in ['HCD', 'CE', 'CECOM']]
    os.makedirs(output_folder, exist_ok=True)
    for ion in ion_columns:
        plt.figure(figsize=(10, 6))
        x_data = df[energy_label].values
        y_data = df[ion].values
        if normalize:
            max_intensity = y_data.max()
            if max_intensity > 0:
                y_data = y_data / max_intensity * 100
            y_label = f'Normalized Intensity{y_label_suffix} (%)'
        else:
            y_label = f'Intensity{y_label_suffix}'
        plt.plot(x_data, y_data, 'o', color='blue', label='Data')
        if sigmoidal_fit:
            try:
                p0 = [max(y_data), np.median(x_data), 1, min(y_data)]
                popt, _ = curve_fit(sigmoid, x_data, y_data, p0, method='dogbox', maxfev=10000)
                residuals = y_data - sigmoid(x_data, *popt)
                r_squared = 1 - (np.sum(residuals**2) / np.sum((y_data - np.mean(y_data))**2))
                if r_squared >= 0.9:
                    x_fit = np.linspace(min(x_data), max(x_data), 100)
                    y_fit = sigmoid(x_fit, *popt)
                    plt.plot(x_fit, y_fit, 'r-', label='Sigmoidal Fit')
                    equation = (
                        r'$y = \frac{' + format_equation_param(popt[0]) + r'}{1 + e^{' +
                        format_equation_param(-popt[2]) + r'(x-' + format_equation_param(popt[1]) +
                        r')}} + ' + format_equation_param(popt[3]) + '$\n' +
                        f'$R^2 = {r_squared:.4f}$'
                    )
                    plt.text(0.98, 0.5, equation, transform=plt.gca().transAxes,
                             fontsize=10, verticalalignment='center', horizontalalignment='right',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                else:
                    plt.text(0.98, 0.5, f"$R^2 = {r_squared:.4f}$ (No good fit)",
                             transform=plt.gca().transAxes,
                             fontsize=10, verticalalignment='center', horizontalalignment='right',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            except Exception as e:
                print(f"Could not fit sigmoidal curve for m/z {ion}: {str(e)}")
                plt.text(0.98, 0.5, "Sigmoidal fit failed", transform=plt.gca().transAxes,
                         fontsize=10, verticalalignment='center', horizontalalignment='right',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        xlabel = r'CE$_{\mathrm{COM}}$ (eV)' if energy_label == 'CECOM' else ('CE (eV)' if energy_label == 'CE' else 'HCD')
        plt.xlabel(xlabel)
        plt.ylabel(y_label)
        title = rf'Ion Intensity vs. CE$_{{\mathrm{{COM}}}}$ (m/z {ion})' if energy_label == 'CECOM' else f'Ion Intensity vs. {energy_label} (m/z {ion})'
        plt.title(title)
        plt.grid(True)
        if sigmoidal_fit:
            plt.legend(loc='upper left')
        plt.subplots_adjust(right=0.75)
        plot_basename = f'ion_intensity_mz_{ion}_{energy_label.lower()}{y_label_suffix.replace(" ", "_").lower()}{"_normalized" if normalize else ""}'
        svg_path, svg_filename = get_unique_filename(output_folder, plot_basename, 'svg')
        plt.savefig(svg_path, bbox_inches='tight')
        print(f"Individual plot saved to {svg_filename}")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Extract ion intensities from .mzML files and optionally plot them.')
    parser.add_argument('folder_path', help='Path to the folder containing .mzML files')
    parser.add_argument('ions', help='List of target ion masses (comma-separated)')
    parser.add_argument('--plot', action='store_true', help='Generate plots of the results')
    parser.add_argument('--r', action='store_true', help='Use relative intensities instead of absolute')
    parser.add_argument('--s', action='store_true', help='Save separate graph for each ion (in addition to combined plot)')
    parser.add_argument('--n', action='store_true', help='Normalize each ion to 100%% in separate graphs (requires --s)')
    parser.add_argument('--sig', action='store_true', help='Fit sigmoidal curve to individual ion plots (requires --s)')
    parser.add_argument('--CE', action='store_true', help='Convert HCD values to Collision Energy (eV)')
    parser.add_argument('--COM', action='store_true', help='Use Center of Mass collision energy (requires --CE)')
    parser.add_argument('--tog', help='Only generate a combined plot for specified ion masses (comma-separated)')
    args = parser.parse_args()

    if args.n and not args.s:
        parser.error("--n requires --s to be specified")
    if args.sig and not args.s:
        parser.error("--sig requires --s to be specified")
    if args.COM and not args.CE:
        parser.error("--COM requires --CE to be specified")

    if args.tog:
        args.s = False
        args.n = False
        args.sig = False
        ions_for_plotting = [float(mass.strip()) for mass in args.tog.split(',')]
        full_ion_list = [float(mass.strip()) for mass in args.ions.split(',')]
        target_ions = full_ion_list
    else:
        ions_for_plotting = None
        target_ions = [float(mass.strip()) for mass in args.ions.split(',')]

    if not target_ions:
        parser.error("At least one ion mass must be specified")

    abs_results, rel_results, energy_label, precursor_mass = process_mzml_files(
        args.folder_path, target_ions, args.CE, args.COM
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

    output_folder = 'SYF_output'
    os.makedirs(output_folder, exist_ok=True)

    output_basename = f'ion_intensities_by_{energy_label.lower()}{"_relative" if args.r else ""}'
    csv_path, csv_filename = get_unique_filename(output_folder, output_basename, 'csv')
    (rel_df if args.r else abs_df).to_csv(csv_path, index=False)
    print(f"Results saved to {csv_filename}")
    print(f"Precursor mass used for CECOM calculation: {precursor_mass:.4f} m/z")

    if args.plot:
        plot_df = rel_df if args.r else abs_df
        y_label_suffix = ' (Relative)' if args.r else ' (Absolute)'

        if args.tog:
            selected_cols = energy_cols + [ion for ion in ions_for_plotting if ion in plot_df.columns]
            plot_df = plot_df[selected_cols]
            save_combined_plot(plot_df, energy_label, y_label_suffix, output_folder)
        else:
            save_combined_plot(plot_df, energy_label, y_label_suffix, output_folder)
            if args.s:
                save_individual_plots(plot_df, energy_label, y_label_suffix, args.n, args.sig, output_folder)

if __name__ == "__main__":
    main()
