import matplotlib.pyplot as plt
from pyteomics import mzml
import numpy as np
import os
import re

DEFAULT_MZ_TOLERANCE = 0.5  # in Da

def get_hcd_energy_from_filename(filename):
    """Extract HCD energy from filename (pattern ...HCDXX.mzML where XX is the energy)"""
    match = re.search(r'HCD(\d+)\.mzML$', filename, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None

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

def get_spectrum_data(mzml_file, scan_num=None):
    with mzml.read(mzml_file) as reader:
        if scan_num is not None:
            scan = next((s for s in reader if s['index'] + 1 == scan_num and s['ms level'] == 2), None)
            if not scan:
                raise ValueError(f"MS2 scan number {scan_num} not found in {mzml_file}")
            
            mz_array = scan['m/z array']
            intensity_array = scan['intensity array']
            scan_label = f"Scan {scan['index'] + 1}"
        
        else:
            all_mz = []
            all_intensities = []
            ms2_scans = [s for s in reader if s['ms level'] == 2]
            
            if not ms2_scans:
                raise ValueError(f"No MS2 scans found in {mzml_file}")
            
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
    
    return mz_array, intensity_array, scan_label

def plot_msms_with_labels(mzml_file=None, scan_num=None, intype='r',
                         outdir=None, clean_threshold=0.0, labelmin_threshold=0.0,
                         overlay_files=None):
    if overlay_files:
        # 2D overlay mode
        fig, ax = plt.subplots(figsize=(12, 6))
        global_max_intensity = 0.0
        
        # Process all overlay files
        for overlay_file in overlay_files:
            try:
                overlay_mz, overlay_intensity, overlay_label = get_spectrum_data(
                    overlay_file, scan_num)
                
                if intype.lower() == 'r':
                    overlay_intensity = (overlay_intensity / np.max(overlay_intensity)) * 100
                
                overlay_mz, overlay_intensity = prune_close_peaks(overlay_mz, overlay_intensity)
                mask = overlay_intensity >= clean_threshold
                overlay_mz = overlay_mz[mask]
                overlay_intensity = overlay_intensity[mask]
                
                # Track global max for ylim headroom
                if overlay_intensity.size:
                    global_max_intensity = max(global_max_intensity, float(np.max(overlay_intensity)))
                
                # Draw vertical lines from 0 to intensity
                for mz, intensity in zip(overlay_mz, overlay_intensity):
                    ax.vlines(mz, 0, intensity, linewidth=1)
                    if intensity >= labelmin_threshold:
                        ax.text(mz, float(intensity), f'{mz:.4f}', 
                               rotation=90, va='bottom', ha='center',
                               fontsize=8)
                
            except Exception as e:
                print(f"Error processing {overlay_file}: {str(e)}")
                continue
        
        ax.set_xlabel('m/z', fontsize=12)
        if intype.lower() == 'r':
            ax.set_ylabel('Relative Intensity (%)', fontsize=12)
        else:
            ax.set_ylabel('Absolute Intensity', fontsize=12)
        
        # Ensure visible baseline at 0 and add a small headroom on top
        top = global_max_intensity * 1.05 if global_max_intensity > 0 else None
        if top is not None:
            ax.set_ylim(0, top)
        else:
            ax.set_ylim(bottom=0)
        
        # Remove top border
        ax.spines['top'].set_visible(False)
        
        plt.tight_layout()
        
        if outdir:
            os.makedirs(os.path.dirname(outdir) or '.', exist_ok=True)
            plt.savefig(outdir, dpi=300, bbox_inches='tight')
            print(f"Plot saved as {outdir}")
        else:
            plt.show()
    else:
        # Single file mode
        mz_array, intensity_array, scan_label = get_spectrum_data(
            mzml_file, scan_num)
        
        if intype.lower() == 'r':
            intensity_array = (intensity_array / np.max(intensity_array)) * 100
        
        mz_array, intensity_array = prune_close_peaks(mz_array, intensity_array)
        mask = intensity_array >= clean_threshold
        mz_array = mz_array[mask]
        intensity_array = intensity_array[mask]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Draw vertical lines from 0 to intensity
        for mz, intensity in zip(mz_array, intensity_array):
            ax.vlines(mz, 0, float(intensity), linewidth=1)
            if intensity >= labelmin_threshold:
                ax.text(mz, float(intensity), f'{mz:.4f}', 
                       rotation=90, va='bottom', ha='center',
                       fontsize=8)
        
        ax.set_xlabel('m/z', fontsize=12)
        if intype.lower() == 'r':
            ax.set_ylabel('Relative Intensity (%)', fontsize=12)
        else:
            ax.set_ylabel('Absolute Intensity', fontsize=12)
        
        # Ensure baseline is visible as 0 and add headroom
        ymax = float(np.max(intensity_array)) if intensity_array.size else 1.0
        ax.set_ylim(0, ymax * 1.05 if ymax > 0 else 1.0)
        
        # Remove top border
        ax.spines['top'].set_visible(False)
        
        plt.tight_layout()
        
        if outdir:
            os.makedirs(os.path.dirname(outdir) or '.', exist_ok=True)
            plt.savefig(outdir, dpi=300, bbox_inches='tight')
            print(f"Plot saved as {outdir}")
        else:
            plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Plot MS/MS spectrum with smart labeling')
    parser.add_argument('mzml_file', nargs='+', help='Path(s) to mzML file(s)')
    parser.add_argument('--scan', type=int, help='Specific scan number')
    parser.add_argument('--intype', choices=['a', 'r'], default='r',
                       help='Intensity type: a=absolute, r=relative (default)')
    parser.add_argument('--outdir',
                       help='Output file path (with extension, e.g. myplot.svg). If not provided, shows the plot.')
    parser.add_argument('--clean', type=float, default=0.0, dest='clean_threshold',
                       help='Minimum intensity threshold to plot peaks (default: 0)')
    parser.add_argument('--labelmin', type=float, default=0.0, dest='labelmin_threshold',
                       help='Minimum intensity threshold to label peaks (default: 0)')
    args = parser.parse_args()
    
    if len(args.mzml_file) > 1:
        # Multiple files = overlay mode
        print(f"Found {len(args.mzml_file)} files to overlay")
        plot_msms_with_labels(
            overlay_files=args.mzml_file,
            scan_num=args.scan,
            intype=args.intype,
            outdir=args.outdir,
            clean_threshold=args.clean_threshold,
            labelmin_threshold=args.labelmin_threshold
        )
    else:
        plot_msms_with_labels(
            mzml_file=args.mzml_file[0],
            scan_num=args.scan,
            intype=args.intype,
            outdir=args.outdir,
            clean_threshold=args.clean_threshold,
            labelmin_threshold=args.labelmin_threshold,
            overlay_files=None
        )
