import matplotlib.pyplot as plt
from pyteomics import mzml
import numpy as np
import os
import argparse

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

def read_text_file(txt_file, intype='r'):
    """Read mass spec data from text file with m/z, intensity, rel.intensity columns."""
    try:
        data = np.loadtxt(txt_file, skiprows=1)  # Skip header row
        mz_array = data[:, 0]
        
        if intype.lower() == 'r':
            intensity_array = data[:, 2]  # Relative intensity
            ylabel = 'Relative Intensity (%)'
        else:
            intensity_array = data[:, 1]  # Absolute intensity
            ylabel = 'Absolute Intensity'
            
        return mz_array, intensity_array, ylabel, None  # No CE info in text files
        
    except Exception as e:
        raise ValueError(f"Error reading text file: {str(e)}")

def plot_spectrum(mz_array, intensity_array, ylabel, title, 
                 mz_tolerance=0.5, label_threshold=None, 
                 save_plot=False, output_file=None, decimals=4):
    """Core plotting function for both file types."""
    mz_filtered, intensity_filtered = prune_close_peaks(mz_array, intensity_array, mz_tolerance)

    fig, ax = plt.subplots(figsize=(12, 6))
    markerline, stemlines, baseline = ax.stem(
        mz_array, intensity_array, 
        linefmt='b-', markerfmt=' ', basefmt='k-')
    plt.setp(stemlines, linewidth=0.7, color='steelblue')

    # Label peaks based on threshold (if provided)
    if label_threshold is not None:
        for mz, intensity in zip(mz_filtered, intensity_filtered):
            if intensity >= label_threshold:
                ax.text(mz, intensity, f"{mz:.{decimals}f}", 
                       ha='center', va='bottom', fontsize=11, color='black')

    ax.set_xlim(np.min(mz_array) - 2, np.max(mz_array) + 2)
    ax.set_ylim(0, np.max(intensity_array) * 1.1)
    ax.set_xlabel('m/z', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(output_file, dpi=300)
        print(f"Plot saved as {output_file}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot MS/MS spectrum from mzML or text file')
    parser.add_argument('input_file', help='Path to input file (mzML or text)')
    parser.add_argument('--txt', action='store_true', help='Process as text file')
    parser.add_argument('--scan', type=int, help='Specific scan number (mzML only)')
    parser.add_argument('--title', help='Custom plot title')
    parser.add_argument('--tolerance', type=float, default=0.5, 
                       help='m/z tolerance for peak clustering (default: 0.5)')
    parser.add_argument('--intype', choices=['a', 'r'], default='r',
                       help='Intensity type: a=absolute, r=relative (default)')
    parser.add_argument('--label', type=float, 
                       help='Label peaks with intensity >= this value (5=5% for relative, 5=5 for absolute)')
    parser.add_argument('--save', action='store_true', 
                       help='Save plot as PNG instead of showing it')
    parser.add_argument('--f', type=int, default=4, choices=range(1, 7),
                       help='Number of decimal places for m/z labels (default: 4)')
    args = parser.parse_args()
    
    try:
        if args.txt:
            if not args.input_file.lower().endswith('.txt'):
                print("Warning: --txt flag used but file doesn't have .txt extension")
            mz_array, intensity_array, ylabel, _ = read_text_file(args.input_file, args.intype)
            title = args.title or "MS Spectrum from Text File"
            
            # Generate output filename if saving
            if args.save:
                base_name = os.path.splitext(os.path.basename(args.input_file))[0]
                output_file = f"{base_name}_ms.png"
            else:
                output_file = None
                
            plot_spectrum(mz_array, intensity_array, ylabel, title, 
                         args.tolerance, args.label, args.save, output_file, args.f)
            
        else:
            if args.input_file.lower().endswith('.txt'):
                raise ValueError("Text file provided without --txt flag. Use --txt for text files.")
                
            # Process as mzML file
            with mzml.read(args.input_file) as reader:
                scans = list(reader)
                scan = next((s for s in scans if 
                            (args.scan is None and s['ms level'] == 2) or
                            (args.scan is not None and s['index'] + 1 == args.scan)), None)
                
                if not scan:
                    raise ValueError(f"No suitable MS2 scan found {'with number '+str(args.scan) if args.scan else ''}")

                mz_array = scan['m/z array']
                intensity_array = scan['intensity array']
                if intensity_array is None:
                    raise KeyError("No intensity data found")

                if args.intype.lower() == 'r':
                    intensity_display = (intensity_array / np.max(intensity_array)) * 100
                    ylabel = 'Relative Intensity (%)'
                else:
                    intensity_display = intensity_array
                    ylabel = 'Absolute Intensity'

                title = args.title or f"MS/MS Spectrum (Scan {scan['index'] + 1})"
                
                # Generate output filename if saving
                if args.save:
                    base_name = os.path.splitext(os.path.basename(args.input_file))[0]
                    scan_part = f"_scan{scan['index'] + 1}" if args.scan is not None else ""
                    output_file = f"{base_name}{scan_part}_msms.png"
                else:
                    output_file = None
                    
                plot_spectrum(mz_array, intensity_display, ylabel, title, 
                             args.tolerance, args.label, args.save, output_file, args.f)
                
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    main()