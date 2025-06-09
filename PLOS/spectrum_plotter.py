import matplotlib.pyplot as plt
from pyteomics import mzml
import numpy as np
import os

DEFAULT_MZ_TOLERANCE = 0.5  # in Da

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

def plot_msms_with_labels(mzml_file, scan_num=None, title=None, intype='r',
                          outdir=None, clean_threshold=0.0, labelmin_threshold=0.0):
    
    with mzml.read(mzml_file) as reader:
        
        if scan_num is not None:
            scan = next((s for s in reader if s['index'] + 1 == scan_num and s['ms level'] == 2), None)
            if not scan:
                raise ValueError(f"MS2 scan number {scan_num} not found")
            
            mz_array = scan['m/z array']
            intensity_array = scan['intensity array']
            scan_label = f"Scan {scan['index'] + 1}"
        
        else:
            all_mz = []
            all_intensities = []
            ms2_scans = [s for s in reader if s['ms level'] == 2]
            
            if not ms2_scans:
                raise ValueError("No MS2 scans found in file.")
            
            print(f"Found {len(ms2_scans)} MS2 scans. Averaging them...")
            
            for scan in ms2_scans:
                mz_array = scan['m/z array']
                intensity_array = scan['intensity array']
                
                all_mz.extend(mz_array)
                all_intensities.extend(intensity_array)
            
            bins = np.arange(min(all_mz), max(all_mz) + DEFAULT_MZ_TOLERANCE, DEFAULT_MZ_TOLERANCE)
            digitized = np.digitize(all_mz, bins)
            avg_mz = []
            avg_intensity = []
            
            for i in range(1, len(bins)):
                bin_indices = np.where(digitized == i)[0]
                if len(bin_indices) > 0:
                    avg_mz.append(np.mean(np.array(all_mz)[bin_indices]))
                    avg_intensity.append(np.mean(np.array(all_intensities)[bin_indices]))
            
            mz_array = np.array(avg_mz)
            intensity_array = np.array(avg_intensity)
            scan_label = "Average of all MS2 scans"
        
        if intype.lower() == 'r':
            intensity_display = (intensity_array / np.max(intensity_array)) * 100
            ylabel = 'Relative Intensity (%)'
        else:
            intensity_display = intensity_array
            ylabel = 'Absolute Intensity'
        
        mz_filtered, intensity_filtered = prune_close_peaks(mz_array, intensity_display)
        
        # Apply --clean threshold → affects plotting only
        mask = intensity_filtered >= clean_threshold
        mz_filtered = mz_filtered[mask]
        intensity_filtered = intensity_filtered[mask]
        
        x_min, x_max = np.min(mz_array), np.max(mz_array)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        markerline, stemlines, baseline = ax.stem(mz_filtered, intensity_filtered, linefmt='b-', markerfmt=' ', basefmt='k-')
        
        # ---- Control peak thickness here ----
        plt.setp(stemlines, linewidth=1.5, color='steelblue')  # You can adjust linewidth here
        
        rel_offset = 5.0  # Vertical offset for relative intensity label (fixes overlap)
        
        for mz, intensity in zip(mz_filtered, intensity_filtered):
            if intensity >= labelmin_threshold:
                # First intensity label (always on top)
                if intype.lower() == 'r':
                    intensity_text = f"{intensity:.1f}%"
                else:
                    intensity_text = f"{intensity:.1e}"
                
                ax.text(mz, intensity + rel_offset, intensity_text,
                        ha='center', va='bottom',
                        fontsize=10, color='dimgray')  # <-- Font size for intensity label
                
                # Then mz label under it
                ax.text(mz, intensity, f"{mz:.4f}",
                        ha='center', va='bottom',
                        fontsize=12, color='black')  # <-- Font size for mz label
        
        ax.set_xlim(x_min - 2, x_max + 2)
        ax.set_ylim(0, np.max(intensity_filtered) * 1.2)
        
        # ---- Axes labels font size ----
        ax.set_xlabel('m/z', fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        
        if title:
            # ---- Plot title font size ----
            ax.set_title(title, fontsize=16)
        
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        if outdir:
            # outdir must be full path + filename with extension
            outdir_dir = os.path.dirname(outdir)
            if outdir_dir:
                os.makedirs(outdir_dir, exist_ok=True)
            plt.savefig(outdir, dpi=300)
            print(f"Plot saved as {outdir}")
        else:
            plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Plot MS/MS spectrum with smart labeling')
    parser.add_argument('mzml_file', help='Path to mzML file')
    parser.add_argument('--scan', type=int, help='Specific scan number')
    parser.add_argument('--title', help='Custom plot title (optional)')
    parser.add_argument('--intype', choices=['a', 'r'], default='r',
                        help='Intensity type: a=absolute, r=relative (default)')
    parser.add_argument('--outdir',
                        help='Output file path (with extension, e.g. myplot.svg). If not provided, shows the plot.')
    parser.add_argument('--clean', type=float, default=0.0,
                        help='Minimum intensity threshold to plot peaks (default: 0)')
    parser.add_argument('--labelmin', type=float, default=0.0,
                        help='Minimum intensity threshold to label peaks (default: 0)')
    args = parser.parse_args()
    
    plot_msms_with_labels(
        args.mzml_file,
        args.scan,
        args.title,
        args.intype,
        args.outdir,
        args.clean,
        args.labelmin
    )

