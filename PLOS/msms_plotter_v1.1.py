import matplotlib.pyplot as plt
from pyteomics import mzml  # Changed from mzxml to mzml
import numpy as np
import os

def prune_close_peaks(mz_array, intensity_array, tolerance=0.5):
    """Remove nearby peaks, keeping only the highest intensity one in each cluster."""
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

def plot_msms_with_labels(mzml_file, scan_num=None, title=None, mz_tolerance=0.5, intype='r', save_plot=False):
    """
    Plot MS/MS spectrum from mzML file with:
    - Option for absolute (a) or relative (r) intensities
    - Clean peak labels without borders
    - Intensity percentage shown below m/z values
    - Auto-adjusted axes
    
    Parameters:
    - save_plot: If True, saves the plot as PNG instead of showing it
    """
    with mzml.read(mzml_file) as reader:  # Changed to mzml.read
        # Find target scan
        scan = next((s for s in reader if 
                    (scan_num is None and s['ms level'] == 2) or  # mzML uses 'ms level' instead of 'msLevel'
                    (scan_num is not None and s['index'] + 1 == scan_num)), None)  # mzML uses 'index' (0-based)
        
        if not scan:
            raise ValueError("No suitable MS2 scan found")

        # Extract data - mzML uses slightly different keys
        mz_array = scan['m/z array']
        intensity_array = scan['intensity array']
        if intensity_array is None:
            raise KeyError("No intensity data found")

        # Process intensity data based on type
        if intype.lower() == 'r':  # Relative intensities
            intensity_display = (intensity_array / np.max(intensity_array)) * 100
            ylabel = 'Relative Intensity (%)'
        else:  # Absolute intensities
            intensity_display = intensity_array
            ylabel = 'Absolute Intensity'

        # Prune nearby peaks
        mz_filtered, intensity_filtered = prune_close_peaks(mz_array, intensity_display, mz_tolerance)

        # Auto-calculate axis limits
        x_min, x_max = np.min(mz_array), np.max(mz_array)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Modern stem plot
        markerline, stemlines, baseline = ax.stem(
            mz_array, intensity_display, 
            linefmt='b-', markerfmt=' ', basefmt='k-')
        
        # Customize stem lines
        plt.setp(stemlines, linewidth=0.7, color='steelblue')

        # Label peaks >10% intensity (or top 10% for absolute)
        threshold = np.max(intensity_display) * 0.1 if intype.lower() == 'a' else 10
        for mz, intensity in zip(mz_filtered, intensity_filtered):
            if intensity > threshold:
                # Main m/z label (no border)
                ax.text(mz, intensity, f"{mz:.4f}", 
                        ha='center', va='bottom', fontsize=8, color='black')
                
                # Intensity percentage label below
                if intype.lower() == 'r':
                    intensity_text = f"{intensity:.1f}%"
                else:
                    intensity_text = f"{intensity:.1e}"
                
                ax.text(mz, intensity * 0.7, intensity_text,
                        ha='center', va='top', fontsize=7, color='dimgray')

        # Formatting with tight axis limits
        ax.set_xlim(x_min - 2, x_max + 2)
        ax.set_ylim(0, np.max(intensity_display) * 1.1)
        ax.set_xlabel('m/z', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        
        title_text = title or f"MS/MS Spectrum (Scan {scan['index'] + 1})"  # mzML uses 0-based index
        if 'collision energy' in scan:  # mzML uses lowercase with space
            title_text += f", CE: {scan['collision energy']}"
        ax.set_title(title_text, fontsize=14)
        
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        if save_plot:
            # Generate filename
            base_name = os.path.splitext(os.path.basename(mzml_file))[0]
            scan_part = f"_scan{scan['index'] + 1}" if scan_num is not None else ""
            filename = f"{base_name}{scan_part}_msms.png"
            plt.savefig(filename, dpi=300)
            print(f"Plot saved as {filename}")
        else:
            plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Plot MS/MS spectrum with smart labeling')
    parser.add_argument('mzml_file', help='Path to mzML file')  # Changed from mzxml_file to mzml_file
    parser.add_argument('--scan', type=int, help='Specific scan number')
    parser.add_argument('--title', help='Custom plot title')
    parser.add_argument('--tolerance', type=float, default=0.5, 
                       help='m/z tolerance for peak clustering (default: 0.5)')
    parser.add_argument('--intype', choices=['a', 'r'], default='r',
                       help='Intensity type: a=absolute, r=relative (default)')
    parser.add_argument('--save', action='store_true', 
                       help='Save plot as PNG instead of showing it')
    args = parser.parse_args()
    
    plot_msms_with_labels(
        args.mzml_file,  # Changed from args.mzxml_file
        args.scan, 
        args.title, 
        args.tolerance, 
        args.intype,
        args.save
    )