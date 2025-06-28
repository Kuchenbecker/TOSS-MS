import matplotlib.pyplot as plt
from pyteomics import mzml
import numpy as np
import os
import argparse
import glob

DEFAULT_MZ_TOLERANCE = 0.5  # binning tolerance in Da
SOLVENT_EXCLUSION_TOLERANCE = 0.2  # mass difference tolerance for exclusion

def get_average_ms1_spectrum(mzml_file, mz_tolerance=DEFAULT_MZ_TOLERANCE):
    all_mz = []
    all_intensities = []

    with mzml.read(mzml_file) as reader:
        ms1_scans = [s for s in reader if s['ms level'] == 1]
        if not ms1_scans:
            raise ValueError(f"No MS1 scans found in {mzml_file}")

        for scan in ms1_scans:
            mz_array = scan['m/z array']
            intensity_array = scan['intensity array']
            all_mz.extend(mz_array)
            all_intensities.extend(intensity_array)

    all_mz = np.array(all_mz)
    all_intensities = np.array(all_intensities)

    bins = np.arange(min(all_mz), max(all_mz) + mz_tolerance, mz_tolerance)
    digitized = np.digitize(all_mz, bins)

    avg_mz = []
    avg_intensity = []

    for i in range(1, len(bins)):
        bin_indices = np.where(digitized == i)[0]
        if len(bin_indices) > 0:
            avg_mz.append(np.mean(all_mz[bin_indices]))
            avg_intensity.append(np.mean(all_intensities[bin_indices]))

    return np.array(avg_mz), np.array(avg_intensity)

def prune_close_peaks(mz_array, intensity_array, tolerance=DEFAULT_MZ_TOLERANCE):
    if len(mz_array) == 0:
        return mz_array, intensity_array

    sort_idx = np.argsort(mz_array)
    mz_sorted = mz_array[sort_idx]
    intensity_sorted = intensity_array[sort_idx]

    filtered_mz, filtered_intensity = [], []
    current_cluster_mz, current_cluster_intensity = [mz_sorted[0]], [intensity_sorted[0]]

    for mz, intensity in zip(mz_sorted[1:], intensity_sorted[1:]):
        if mz - current_cluster_mz[-1] <= tolerance:
            current_cluster_mz.append(mz)
            current_cluster_intensity.append(intensity)
        else:
            max_idx = np.argmax(current_cluster_intensity)
            filtered_mz.append(current_cluster_mz[max_idx])
            filtered_intensity.append(current_cluster_intensity[max_idx])
            current_cluster_mz, current_cluster_intensity = [mz], [intensity]

    max_idx = np.argmax(current_cluster_intensity)
    filtered_mz.append(current_cluster_mz[max_idx])
    filtered_intensity.append(current_cluster_intensity[max_idx])

    return np.array(filtered_mz), np.array(filtered_intensity)

def exclude_solvent_peaks(sample_mz, sample_intensity, solvent_mz, tolerance=SOLVENT_EXCLUSION_TOLERANCE):
    if len(solvent_mz) == 0:
        return sample_mz, sample_intensity

    mask = np.ones(len(sample_mz), dtype=bool)

    for mz_solv in solvent_mz:
        matches = np.where(np.abs(sample_mz - mz_solv) <= tolerance)[0]
        mask[matches] = False

    filtered_mz = sample_mz[mask]
    filtered_intensity = sample_intensity[mask]

    return filtered_mz, filtered_intensity

def plot_ms1_spectra(main_data, overlap_data_list, title=None, intype='r',
                     outdir=None, clean_threshold=0.0, labelmin_threshold=0.0):
    fig = plt.figure(figsize=(12, 8))  # increased height for table
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])

    ax = fig.add_subplot(gs[0])

    # Always draw baseline
    ax.axhline(y=0, color='black', linewidth=1)

    def process_and_plot(mz_array, intensity_array, label, color=None):
        if intype.lower() == 'r':
            intensity_array = (intensity_array / np.max(intensity_array)) * 100

        mz_array, intensity_array = prune_close_peaks(mz_array, intensity_array)
        mask = intensity_array >= clean_threshold
        mz_array = mz_array[mask]
        intensity_array = intensity_array[mask]

        # Plot stem
        markerline, stemlines, baseline = ax.stem(
            mz_array, intensity_array,
            linefmt='-', markerfmt=' ', basefmt=' ', label=label)
        plt.setp(stemlines, 'linewidth', 1)
        if color:
            plt.setp(stemlines, 'color', color)

        # Label peaks above threshold
        for mz, intensity in zip(mz_array, intensity_array):
            if intensity >= labelmin_threshold:
                ax.text(mz, intensity + (0.02 * max(intensity_array)), f'{mz:.2f}',
                        rotation=90, va='bottom', ha='center',
                        fontsize=8)

        return mz_array, intensity_array

    # Plot main data
    sample_mz, sample_intensity = process_and_plot(main_data[0], main_data[1], 'Input mzML')

    # Plot overlaps
    colors = plt.cm.tab10.colors
    for i, (mz_array, intensity_array, label) in enumerate(overlap_data_list):
        color = colors[(i+1) % len(colors)]
        process_and_plot(mz_array, intensity_array, label, color=color)

    # Adjust y-limit to add headroom for labels
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.15)

    ax.set_xlabel('m/z', fontsize=12)
    ylabel = 'Relative Intensity (%)' if intype.lower() == 'r' else 'Absolute Intensity'
    ax.set_ylabel(ylabel, fontsize=12)

    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title('Average MS1 Spectrum', fontsize=14)

    ax.legend()

    # Prepare table of top 5 peaks from main sample
    top_indices = np.argsort(sample_intensity)[-5:][::-1]
    top_mz = sample_mz[top_indices]
    top_int = sample_intensity[top_indices]

    table_data = []
    for mz, inten in zip(top_mz, top_int):
        table_data.append([f"{mz:.4f}", f"{inten:.2f}"])

    ax_table = fig.add_subplot(gs[1])
    ax_table.axis('off')
    table = ax_table.table(cellText=table_data,
                           colLabels=["m/z", "Intensity"],
                           loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    plt.tight_layout()

    if outdir:
        os.makedirs(os.path.dirname(outdir) or '.', exist_ok=True)
        ext = os.path.splitext(outdir)[1].lower()
        if ext not in ['.png', '.svg', '.pdf']:
            raise ValueError("Output file extension must be .png, .svg, or .pdf")
        plt.savefig(outdir, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {outdir}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot averaged MS1 spectrum from mzML file with optional solvent exclusion and overlaps')
    parser.add_argument('mzml_file', help='Path to sample MS1 mzML file')
    parser.add_argument('--solvent', help='Path to solvent MS1 mzML file to exclude peaks from')
    parser.add_argument('--overlap', help='Path to mzML file or folder to overlap')
    parser.add_argument('--title', help='Custom plot title (optional)')
    parser.add_argument('--intype', choices=['a', 'r'], default='r',
                        help='Intensity type: a=absolute, r=relative (default)')
    parser.add_argument('--outdir',
                        help='Output file path (with .png, .svg, or .pdf). If not provided, shows the plot.')
    parser.add_argument('--clean', type=float, default=0.0,
                        help='Minimum intensity threshold to plot peaks (default: 0)')
    parser.add_argument('--labelmin', type=float, default=0.0,
                        help='Minimum intensity threshold to label peaks (default: 0)')
    args = parser.parse_args()

    # Process main sample
    mz_array, intensity_array = get_average_ms1_spectrum(args.mzml_file)

    # If solvent provided, process and exclude peaks
    if args.solvent:
        solvent_mz, solvent_intensity = get_average_ms1_spectrum(args.solvent)
        mz_array, intensity_array = exclude_solvent_peaks(mz_array, intensity_array, solvent_mz)

    main_data = (mz_array, intensity_array)

    # Process overlaps
    overlap_data_list = []
    if args.overlap:
        if os.path.isdir(args.overlap):
            files = glob.glob(os.path.join(args.overlap, '*.mzML'))
        else:
            files = [args.overlap]

        for file in files:
            try:
                mz_o, int_o = get_average_ms1_spectrum(file)
                label = os.path.splitext(os.path.basename(file))[0]
                overlap_data_list.append((mz_o, int_o, label))
            except Exception as e:
                print(f"Warning: Could not process {file}: {e}")

    plot_ms1_spectra(main_data, overlap_data_list,
                     title=args.title,
                     intype=args.intype,
                     outdir=args.outdir,
                     clean_threshold=args.clean,
                     labelmin_threshold=args.labelmin)
