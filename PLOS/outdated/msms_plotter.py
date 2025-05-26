import matplotlib.pyplot as plt
from pyteomics import mzxml
import numpy as np

def prune_close_peaks(mz_array, intensity_array, tolerance=0.5):
    """
    Remove nearby peaks (within m/z tolerance), keeping only the highest intensity one in each cluster.
    Returns filtered m/z and intensity arrays.
    """
    if len(mz_array) == 0:
        return mz_array, intensity_array
    
    # Sort by m/z (just in case)
    sort_idx = np.argsort(mz_array)
    mz_sorted = mz_array[sort_idx]
    intensity_sorted = intensity_array[sort_idx]
    
    filtered_mz = []
    filtered_intensity = []
    
    current_cluster_mz = [mz_sorted[0]]
    current_cluster_intensity = [intensity_sorted[0]]
    
    for mz, intensity in zip(mz_sorted[1:], intensity_sorted[1:]):
        if mz - current_cluster_mz[-1] <= tolerance:
            # Add to current cluster
            current_cluster_mz.append(mz)
            current_cluster_intensity.append(intensity)
        else:
            # Close current cluster and keep highest intensity peak
            max_idx = np.argmax(current_cluster_intensity)
            filtered_mz.append(current_cluster_mz[max_idx])
            filtered_intensity.append(current_cluster_intensity[max_idx])
            
            # Start new cluster
            current_cluster_mz = [mz]
            current_cluster_intensity = [intensity]
    
    # Add the last cluster
    max_idx = np.argmax(current_cluster_intensity)
    filtered_mz.append(current_cluster_mz[max_idx])
    filtered_intensity.append(current_cluster_intensity[max_idx])
    
    return np.array(filtered_mz), np.array(filtered_intensity)

def plot_msms_with_labels(mzxml_file, scan_num=None, title=None, mz_tolerance=0.5):
    """
    Plot MS/MS spectrum with:
    - Relative intensities (%)
    - Labels for peaks >10% intensity
    - Nearby peaks pruned (within mz_tolerance)
    """
    # Read mzXML file
    with mzxml.read(mzxml_file) as reader:
        # Find the target scan
        scan = None
        if scan_num is not None:
            for s in reader:
                if s['num'] == scan_num:
                    scan = s
                    break
        else:
            for s in reader:
                if s['msLevel'] == 2:
                    scan = s
                    break
        
        if scan is None:
            raise ValueError("No suitable MS2 scan found in the file")

        # Extract m/z and intensity data
        mz_array = scan['m/z array']
        if 'intensity array' in scan:
            intensity_array = scan['intensity array']
        elif 'i array' in scan:
            intensity_array = scan['i array']
        else:
            raise KeyError("No intensity data found. Available keys: " + str(scan.keys()))

        # Convert to relative intensity (%)
        intensity_percent = (intensity_array / np.max(intensity_array)) * 100

        # Prune nearby peaks
        mz_filtered, intensity_filtered = prune_close_peaks(mz_array, intensity_percent, mz_tolerance)

        # Get collision energy if available
        collision_energy = scan.get('collisionEnergy', 'N/A')

        # Create the plot
        plt.figure(figsize=(12, 6))
        
        # Stem plot
        stemlines = plt.stem(
            mz_array, intensity_percent,
            linefmt='b-', markerfmt=' ', basefmt='k-')
        
        # Customize stem lines
        plt.setp(stemlines[0], linewidth=0.7, color='steelblue')

        # Label peaks >10% intensity (using filtered data)
        threshold = 10
        for mz, intensity in zip(mz_filtered, intensity_filtered):
            if intensity > threshold:
                plt.text(
                    mz, intensity, f"{mz:.4f}", 
                    ha='center', va='bottom', 
                    fontsize=8,
                    bbox=dict(
                        facecolor='white', 
                        alpha=0.7, 
                        edgecolor='none',
                        boxstyle='round,pad=0.2'
                    )
                )

        # Add plot decorations
        plt.xlabel('m/z', fontsize=12)
        plt.ylabel('Relative Intensity (%)', fontsize=12)
        
        # Dynamic title
        plot_title = title or f"MS/MS Spectrum (Scan {scan['num']})"
        if collision_energy != 'N/A':
            plot_title += f", CE: {collision_energy}"
        plt.title(plot_title, fontsize=14)
        
        # Adjust axes
        plt.xlim(left=0, right=max(mz_array) * 1.1)
        plt.ylim(bottom=0, top=105)
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Plot MS/MS spectrum with labeled peaks')
    parser.add_argument('mzxml_file', help='Path to mzXML file')
    parser.add_argument('--scan', type=int, help='Specific scan number to plot')
    parser.add_argument('--title', help='Custom plot title')
    parser.add_argument('--tolerance', type=float, default=0.5, 
                       help='m/z tolerance for peak pruning (default: 0.5)')
    args = parser.parse_args()
    
    plot_msms_with_labels(args.mzxml_file, args.scan, args.title, args.tolerance)