import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path

def load_kitti_bin(bin_path: str) -> np.ndarray:
    """
    Load KITTI .bin file.

    Returns:
        np.ndarray: (N, 4) array with columns [x, y, z, intensity]
    """
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points

def analyze_intensity(bin_path: str, output_dir: str = None):
    """
    Analyze intensity distribution of KITTI .bin file.

    Parameters:
        bin_path: Path to .bin file
        output_dir: Directory to save plots (optional)
    """
    print(f"Analyzing: {bin_path}")
    print("=" * 70)

    # Load points
    points = load_kitti_bin(bin_path)
    intensities = points[:, 3]

    total_points = len(intensities)

    # Analyze zero intensity points
    zero_intensity_mask = intensities == 0.0
    zero_count = np.sum(zero_intensity_mask)
    zero_percentage = (zero_count / total_points) * 100

    print(f"\nTotal points: {total_points:,}")
    print(f"Points with intensity = 0: {zero_count:,} ({zero_percentage:.2f}%)")
    print(f"Points with intensity > 0: {total_points - zero_count:,} ({100 - zero_percentage:.2f}%)")

    # Intensity statistics (for non-zero intensities)
    non_zero_intensities = intensities[intensities > 0]
    if len(non_zero_intensities) > 0:
        print(f"\nNon-zero intensity statistics:")
        print(f"  Min:    {np.min(non_zero_intensities):.6f}")
        print(f"  Max:    {np.max(non_zero_intensities):.6f}")
        print(f"  Mean:   {np.mean(non_zero_intensities):.6f}")
        print(f"  Median: {np.median(non_zero_intensities):.6f}")
        print(f"  Std:    {np.std(non_zero_intensities):.6f}")

    # All intensity statistics (including zeros)
    print(f"\nAll intensity statistics (including zeros):")
    print(f"  Min:    {np.min(intensities):.6f}")
    print(f"  Max:    {np.max(intensities):.6f}")
    print(f"  Mean:   {np.mean(intensities):.6f}")
    print(f"  Median: {np.median(intensities):.6f}")
    print(f"  Std:    {np.std(intensities):.6f}")

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Intensity Analysis: {os.path.basename(bin_path)}', fontsize=14, fontweight='bold')

    # 1. Histogram of all intensities
    ax1 = axes[0, 0]
    ax1.hist(intensities, bins=100, alpha=0.7, edgecolor='black', color='steelblue')
    ax1.set_xlabel('Intensity')
    ax1.set_ylabel('Count')
    ax1.set_title(f'All Intensities (n={total_points:,})')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='intensity=0')
    ax1.legend()

    # 2. Histogram of non-zero intensities
    ax2 = axes[0, 1]
    if len(non_zero_intensities) > 0:
        ax2.hist(non_zero_intensities, bins=100, alpha=0.7, edgecolor='black', color='forestgreen')
        ax2.set_xlabel('Intensity')
        ax2.set_ylabel('Count')
        ax2.set_title(f'Non-zero Intensities (n={len(non_zero_intensities):,})')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No non-zero intensities', ha='center', va='center')
        ax2.set_title('Non-zero Intensities')

    # 3. Pie chart of zero vs non-zero
    ax3 = axes[1, 0]
    labels = [f'Intensity = 0\n({zero_percentage:.2f}%)',
              f'Intensity > 0\n({100 - zero_percentage:.2f}%)']
    sizes = [zero_count, total_points - zero_count]
    colors = ['#ff9999', '#66b3ff']
    explode = (0.05, 0)

    ax3.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90, textprops={'fontsize': 10})
    ax3.set_title('Zero vs Non-zero Intensity Distribution')

    # 4. Cumulative distribution
    ax4 = axes[1, 1]
    sorted_intensities = np.sort(intensities)
    cumulative = np.arange(1, len(sorted_intensities) + 1) / len(sorted_intensities) * 100
    ax4.plot(sorted_intensities, cumulative, linewidth=2, color='purple')
    ax4.set_xlabel('Intensity')
    ax4.set_ylabel('Cumulative Percentage (%)')
    ax4.set_title('Cumulative Distribution Function')
    ax4.grid(True, alpha=0.3)
    ax4.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.5)

    plt.tight_layout()

    # Save plot if output directory is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        bin_basename = Path(bin_path).stem
        output_path = os.path.join(output_dir, f"{bin_basename}_intensity_analysis.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved plot to: {output_path}")

    plt.show()

    # Additional analysis: intensity bins
    print("\nIntensity distribution by bins:")
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, bin_edges = np.histogram(intensities, bins=bins)
    for i in range(len(hist)):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i+1]
        count = hist[i]
        percentage = (count / total_points) * 100
        print(f"  [{bin_start:.1f}, {bin_end:.1f}): {count:8,} points ({percentage:6.2f}%)")

    print("=" * 70)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Analyze intensity distribution in KITTI .bin files"
    )
    parser.add_argument("bin_path", type=str, help="Path to KITTI .bin file")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save analysis plots")

    args = parser.parse_args()

    if not os.path.exists(args.bin_path):
        print(f"Error: File not found: {args.bin_path}")
        exit(1)

    analyze_intensity(args.bin_path, args.output_dir)
