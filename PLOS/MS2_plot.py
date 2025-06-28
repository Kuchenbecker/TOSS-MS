import matplotlib.pyplot as plt
from pyteomics import mzml
import numpy as np
import os
import glob
import re
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

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

def subtract_solvent_peaks(main_mz, main_intensity, solvent_mz, solvent_intensity, tolerance=DEFAULT_MZ_TOLERANCE):
    """
    Subtract solvent peaks from main spectrum.
    
    Args:
        main_mz: m/z array of main spectrum
        main_intensity: intensity array of main spectrum
        solvent_mz: m/z array of solvent spectrum
        solvent_intensity: intensity array of solvent spectrum
        tolerance: m/z tolerance for peak matching
        
    Returns:
        Tuple of (filtered_mz, filtered_intensity) with solvent peaks removed
    """
    if len(solvent_mz) == 0:
        return main_mz, main_intensity
    
    # Find peaks in main spectrum that match solvent peaks
    mask = np.ones(len(main_mz), dtype=bool)
    
    for mz_solvent, int_solvent in zip(solvent_mz, solvent_intensity):
        # Find peaks in main spectrum within tolerance of this solvent peak
        matches = np.where(np.abs(main_mz - mz_solvent) <= tolerance)[0]
        
        for idx in matches:
            # Subtract solvent intensity from main intensity
            main_intensity[idx] = max(0, main_intensity[idx] - int_solvent)
            
            # If intensity becomes zero or negative, mark for removal
            if main_intensity[idx] <= 0:
                mask[idx] = False
    
    # Apply mask to remove zero/negative intensity peaks
    filtered_mz = main_mz[mask]
    filtered_intensity = main_intensity[mask]
    
    return filtered_mz, filtered_intensity

def get_spectrum_data(mzml_file, scan_num=None, solvent_mz=None, solvent_intensity=None):
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
    
    # Subtract solvent peaks if provided
    if solvent_mz is not None and solvent_intensity is not None:
        mz_array, intensity_array = subtract_solvent_peaks(mz_array, intensity_array, 
                                                          solvent_mz, solvent_intensity)
    
    return mz_array, intensity_array, scan_label

def plot_3d_spectrum(files_data, title=None, intype='r', outdir=None, clean_threshold=0.0):
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Sort files by HCD energy
    files_data.sort(key=lambda x: x['hcd_energy'] if x['hcd_energy'] is not None else 0)
    
    # Collect all unique m/z values across all spectra
    all_mz = np.unique(np.concatenate([fd['mz_filtered'] for fd in files_data]))
    
    # Create a color map for m/z values
    norm = plt.Normalize(min(all_mz), max(all_mz))
    cmap = cm.get_cmap('viridis', len(all_mz))
    
    # Create a dictionary to map m/z to color
    mz_to_color = {mz: cmap(norm(mz)) for mz in all_mz}
    
    # Plot each m/z across all collision energies
    for mz in all_mz:
        # Collect intensities for this m/z across all collision energies
        hcd_energies = []
        intensities = []
        
        for file_data in files_data:
            idx = np.where(np.isclose(file_data['mz_filtered'], mz, atol=DEFAULT_MZ_TOLERANCE))[0]
            if len(idx) > 0:
                hcd_energies.append(file_data['hcd_energy'])
                intensities.append(file_data['intensity_filtered'][idx[0]])
        
        if len(hcd_energies) > 1:
            # Plot connecting line
            ax.plot([mz]*len(hcd_energies), hcd_energies, intensities,
                    color=mz_to_color[mz], alpha=0.5, linewidth=1)
    
    # Plot the stems for each file
    for file_data in files_data:
        mz_filtered = file_data['mz_filtered']
        intensity_filtered = file_data['intensity_filtered']
        hcd_energy = file_data['hcd_energy'] or 0  # Default to 0 if None
        
        for mz, intensity in zip(mz_filtered, intensity_filtered):
            # Plot stem
            ax.plot([mz, mz], [hcd_energy, hcd_energy], [0, intensity],
                   color=mz_to_color[mz], linewidth=2, alpha=0.8)
            
            # Plot point at top of stem
            ax.scatter([mz], [hcd_energy], [intensity],
                      color=mz_to_color[mz], s=30, alpha=0.8)
    
    # Add colorbar for m/z values
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label('m/z', fontsize=12)
    
    ax.set_xlabel('m/z', fontsize=14, labelpad=10)
    ax.set_ylabel('HCD Collision Energy (%)', fontsize=14, labelpad=10)
    if intype.lower() == 'r':
        ax.set_zlabel('Relative Intensity (%)', fontsize=14, labelpad=10)
    else:
        ax.set_zlabel('Absolute Intensity', fontsize=14, labelpad=10)
    
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    
    if title:
        ax.set_title(title, fontsize=16, pad=20)
    else:
        ax.set_title('MS/MS Spectrum vs HCD Collision Energy', fontsize=16, pad=20)
    
    # Adjust viewing angle for better visibility
    ax.view_init(elev=25, azim=-45)
    
    plt.tight_layout()
    
    if outdir:
        os.makedirs(os.path.dirname(outdir) or '.', exist_ok=True)
        plt.savefig(outdir, dpi=300, bbox_inches='tight')
        print(f"3D plot saved as {outdir}")
    else:
        plt.show()

def plot_msms_with_labels(mzml_file=None, scan_num=None, title=None, intype='r',
                         outdir=None, clean_threshold=0.0, labelmin_threshold=0.0,
                         overlay_files=None, three_d=False, solvent_spectrum=None):
    
    # Load solvent spectrum if provided
    solvent_mz, solvent_intensity = None, None
    if solvent_spectrum:
        try:
            solvent_mz, solvent_intensity, _ = get_spectrum_data(solvent_spectrum, scan_num)
            if intype.lower() == 'r':
                solvent_intensity = (solvent_intensity / np.max(solvent_intensity)) * 100
        except Exception as e:
            print(f"Warning: Could not process solvent spectrum {solvent_spectrum}: {str(e)}")
    
    if overlay_files and three_d:
        # 3D plot mode - only process overlay files
        files_data = []
        
        # Process all overlay files
        for overlay_file in overlay_files:
            try:
                # Get HCD energy from filename
                overlay_hcd = get_hcd_energy_from_filename(overlay_file)
                if overlay_hcd is None:
                    print(f"Warning: Could not determine HCD energy from filename {overlay_file}")
                
                overlay_mz, overlay_intensity, _ = get_spectrum_data(
                    overlay_file, scan_num, solvent_mz, solvent_intensity)
                
                if intype.lower() == 'r':
                    overlay_intensity = (overlay_intensity / np.max(overlay_intensity)) * 100
                
                overlay_mz, overlay_intensity = prune_close_peaks(overlay_mz, overlay_intensity)
                mask = overlay_intensity >= clean_threshold
                overlay_mz = overlay_mz[mask]
                overlay_intensity = overlay_intensity[mask]
                
                files_data.append({
                    'mz_filtered': overlay_mz,
                    'intensity_filtered': overlay_intensity,
                    'hcd_energy': overlay_hcd,
                    'label': os.path.basename(overlay_file)
                })
                
            except Exception as e:
                print(f"Error processing {overlay_file}: {str(e)}")
                continue
        
        if not files_data:
            raise ValueError("No valid files found to plot")
        
        plot_3d_spectrum(files_data, title, intype, outdir, clean_threshold)
        return
    
    elif overlay_files:
        # 2D overlay mode
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Process all overlay files
        for overlay_file in overlay_files:
            try:
                overlay_mz, overlay_intensity, overlay_label = get_spectrum_data(
                    overlay_file, scan_num, solvent_mz, solvent_intensity)
                
                if intype.lower() == 'r':
                    overlay_intensity = (overlay_intensity / np.max(overlay_intensity)) * 100
                
                overlay_mz, overlay_intensity = prune_close_peaks(overlay_mz, overlay_intensity)
                mask = overlay_intensity >= clean_threshold
                overlay_mz = overlay_mz[mask]
                overlay_intensity = overlay_intensity[mask]
                
                # Create stems
                markerline, stemlines, baseline = ax.stem(
                    overlay_mz, overlay_intensity,
                    linefmt='-', markerfmt=' ', basefmt=' ',
                    label=os.path.basename(overlay_file))
                
                # Color the stems
                plt.setp(stemlines, 'linewidth', 1)
                
                # Label peaks above threshold
                for mz, intensity in zip(overlay_mz, overlay_intensity):
                    if intensity >= labelmin_threshold:
                        ax.text(mz, intensity, f'{mz:.2f}', 
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
        
        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title('MS/MS Spectrum Overlay', fontsize=14)
        
        ax.legend()
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
            mzml_file, scan_num, solvent_mz, solvent_intensity)
        
        if intype.lower() == 'r':
            intensity_array = (intensity_array / np.max(intensity_array)) * 100
        
        mz_array, intensity_array = prune_close_peaks(mz_array, intensity_array)
        mask = intensity_array >= clean_threshold
        mz_array = mz_array[mask]
        intensity_array = intensity_array[mask]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create stems
        markerline, stemlines, baseline = ax.stem(
            mz_array, intensity_array,
            linefmt='-', markerfmt=' ', basefmt=' ')
        
        # Color the stems
        plt.setp(stemlines, 'linewidth', 1)
        
        # Label peaks above threshold
        for mz, intensity in zip(mz_array, intensity_array):
            if intensity >= labelmin_threshold:
                ax.text(mz, intensity, f'{mz:.2f}', 
                       rotation=90, va='bottom', ha='center',
                       fontsize=8)
        
        ax.set_xlabel('m/z', fontsize=12)
        if intype.lower() == 'r':
            ax.set_ylabel('Relative Intensity (%)', fontsize=12)
        else:
            ax.set_ylabel('Absolute Intensity', fontsize=12)
        
        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title(f'MS/MS Spectrum - {scan_label}', fontsize=14)
        
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
    parser.add_argument('mzml_file', nargs='?', help='Path to mzML file (optional if --overlay is used)')
    parser.add_argument('--scan', type=int, help='Specific scan number')
    parser.add_argument('--title', help='Custom plot title (optional)')
    parser.add_argument('--intype', choices=['a', 'r'], default='r',
                       help='Intensity type: a=absolute, r=relative (default)')
    parser.add_argument('--outdir',
                       help='Output file path (with extension, e.g. myplot.svg). If not provided, shows the plot.')
    parser.add_argument('--clean', type=float, default=0.0, dest='clean_threshold',
                       help='Minimum intensity threshold to plot peaks (default: 0)')
    parser.add_argument('--labelmin', type=float, default=0.0, dest='labelmin_threshold',
                       help='Minimum intensity threshold to label peaks (default: 0)')
    parser.add_argument('--overlay', type=str,
                       help='Directory containing mzML files to overlay (all .mzML files will be included)')
    parser.add_argument('--3d', action='store_true',
                       help='Create a 3D plot with HCD collision energy (%) on the y-axis')
    parser.add_argument('--solventspectrum', type=str,
                       help='Path to solvent spectrum mzML file to subtract from main spectra')
    args = parser.parse_args()
    
    overlay_files = None
    if args.overlay:
        if not os.path.isdir(args.overlay):
            raise ValueError(f"Directory not found: {args.overlay}")
        overlay_files = glob.glob(os.path.join(args.overlay, '*.mzML'))
        print(f"Found {len(overlay_files)} files to overlay")
        
        plot_msms_with_labels(
            overlay_files=overlay_files,
            scan_num=args.scan,
            title=args.title,
            intype=args.intype,
            outdir=args.outdir,
            clean_threshold=args.clean_threshold,
            labelmin_threshold=args.labelmin_threshold,
            three_d=args.__dict__.get('3d', False),
            solvent_spectrum=args.solventspectrum
        )
    else:
        if not args.mzml_file:
            parser.error("Either specify an mzML file or use --overlay")
        
        plot_msms_with_labels(
            mzml_file=args.mzml_file,
            scan_num=args.scan,
            title=args.title,
            intype=args.intype,
            outdir=args.outdir,
            clean_threshold=args.clean_threshold,
            labelmin_threshold=args.labelmin_threshold,
            overlay_files=None,
            three_d=args.__dict__.get('3d', False),
            solvent_spectrum=args.solventspectrum
        )