import json
from pyteomics import mzml, mass
from itertools import product
import numpy as np
import matplotlib.pyplot as plt

PPM_TOL = 5  # ppm tolerance
REL_INTENSITY_THRESHOLD = 0.10  # 10%

def load_constraints(json_file):
    with open(json_file, 'r') as f:
        constraints = json.load(f)
    return constraints

def generate_candidate_formulas(constraints):
    elements = list(constraints.keys())
    ranges = [range(minmax[0], minmax[1]+1) for minmax in constraints.values()]
    formulas = []
    for counts in product(*ranges):
        formula_dict = {el: count for el, count in zip(elements, counts)}
        if sum(formula_dict.values()) == 0:
            continue  # skip empty formula
        try:
            mass_val = mass.calculate_mass(formula=formula_dict)
            formulas.append((formula_dict, mass_val))
        except:
            continue
    return formulas

def ppm_error(measured, theoretical):
    return abs(measured - theoretical) / theoretical * 1e6

def assign_formulas(spectrum, formulas):
    mzs = spectrum['m/z array']
    intensities = spectrum['intensity array']
    max_intensity = max(intensities)
    results = []

    for mz, intensity in zip(mzs, intensities):
        if intensity < REL_INTENSITY_THRESHOLD * max_intensity:
            continue
        best_match = None
        best_error = float('inf')

        for formula_dict, theo_mass in formulas:
            error = ppm_error(mz, theo_mass)
            if error < PPM_TOL and error < best_error:
                best_match = formula_dict
                best_error = error

        if best_match:
            results.append({
                'mz': mz,
                'intensity': intensity,
                'formula': best_match,
                'ppm_error': best_error
            })

    return mzs, intensities, results

def plot_spectrum(mzs, intensities, annotated_peaks):
    plt.figure(figsize=(12, 6))
    plt.vlines(mzs, 0, intensities, color='gray', lw=1)
    plt.xlabel("m/z")
    plt.ylabel("Intensity")
    plt.title("Annotated MS2 Spectrum")

    for peak in annotated_peaks:
        formula_str = ''.join(f"{el}{peak['formula'][el]}" for el in peak['formula'] if peak['formula'][el] > 0)
        label = f"{peak['mz']:.4f} {formula_str} ({peak['ppm_error']:.1f} ppm)"
        plt.text(peak['mz'], peak['intensity'] + max(intensities)*0.01, label,
                 rotation=90, ha='center', va='bottom', fontsize=8, color='blue')

    plt.tight_layout()
    plt.show()

def main(mzml_file, json_file):
    constraints = load_constraints(json_file)
    candidate_formulas = generate_candidate_formulas(constraints)
    print(f"Generated {len(candidate_formulas)} candidate formulas.")

    with mzml.read(mzml_file) as reader:
        for spectrum in reader:
            if spectrum.get('ms level') != 2:
                continue  # Only use MS2
            mzs, intensities, annotated_peaks = assign_formulas(spectrum, candidate_formulas)
            plot_spectrum(mzs, intensities, annotated_peaks)
            break  # Plot only the first MS2 spectrum

if __name__ == "__main__":
    main("PO3G_135MSMS_HCD10.mzML", "molform_par.json")
