import argparse
from pyteomics import mzxml
from molmass import Formula
from itertools import product

def parse_element_constraints(element_string):
    """
    Parses element constraints string like:
    "C,min=1,max=60;O,min=3,max=10;Na,min=1,max=1"
    Returns a dict like {'C': (1, 60), 'O': (3, 10), 'Na': (1, 1)}
    """
    element_dict = {}
    for item in element_string.split(";"):
        parts = item.split(",")
        element = parts[0].strip()
        min_val = int(parts[1].split("=")[1])
        max_val = int(parts[2].split("=")[1])
        element_dict[element] = (min_val, max_val)
    return element_dict

def generate_candidate_formulas(exact_mass, ppm_tolerance, element_constraints):
    candidates = []
    min_mass = exact_mass * (1 - ppm_tolerance / 1e6)
    max_mass = exact_mass * (1 + ppm_tolerance / 1e6)

    keys, ranges = zip(*element_constraints.items())
    for counts in product(*[range(r[0], r[1] + 1) for r in ranges]):
        formula_dict = dict(zip(keys, counts))
        formula_str = ''.join(f"{el}{n}" for el, n in formula_dict.items() if n > 0)
        try:
            f = Formula(formula_str)
            mass = f.isotope.mass
            if min_mass <= mass <= max_mass:
                candidates.append((formula_str, mass))
        except Exception:
            continue
    return sorted(candidates, key=lambda x: abs(x[1] - exact_mass))

def extract_precursor_masses(file_path):
    precursor_masses = []
    with mzxml.read(file_path) as reader:
        for spectrum in reader:
            if spectrum['msLevel'] == 2:
                if 'precursorMz' in spectrum:
                    mz_info = spectrum['precursorMz'][0]
                    precursor_masses.append(mz_info['precursorMz'])
    return precursor_masses

def main(mzxml_file, ppm_tolerance, element_string):
    element_constraints = parse_element_constraints(element_string)
    precursor_masses = extract_precursor_masses(mzxml_file)

    for idx, mz in enumerate(precursor_masses):
        print(f"\n--- Precursor Ion {idx + 1} ---")
        print(f"Exact Mass (m/z): {mz}")
        formulas = generate_candidate_formulas(mz, ppm_tolerance, element_constraints)
        for f, m in formulas[:10]:  # Show top 10 candidates
            error_ppm = (m - mz) / mz * 1e6
            print(f"  {f:25} - Calc Mass: {m:.5f}, Error: {error_ppm:.2f} ppm")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Molecular Formula Generator from mzXML MS/MS data")
    parser.add_argument("mzxml_file", help="Path to the mzXML file")
    parser.add_argument("--ppm", type=float, default=10, help="PPM tolerance for formula matching (default: 10)")
    parser.add_argument("--elements", required=True,
                        help='Element constraints in the format: "C,min=1,max=60;O,min=3,max=10;Na,min=1,max=1"')

    args = parser.parse_args()
    main(args.mzxml_file, args.ppm, args.elements)
