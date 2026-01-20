#!/usr/bin/env python3
"""
Analyze intensity distribution in nuScenes-style point clouds.
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from pathlib import Path

def load_kitti_bin(bin_path):
    """Load KITTI format .bin file (x, y, z, intensity)."""
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points

def analyze_intensity_distribution(bin_dir, num_samples=10, output_dir='./intensity_analysis'):
    """
    Analyze intensity distribution from multiple point cloud samples.

    Parameters:
    -----------
    bin_dir : str
        Directory containing .bin files
    num_samples : int
        Number of files to sample
    output_dir : str
        Directory to save output graphs
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find all .bin files
    bin_files = sorted(glob.glob(os.path.join(bin_dir, '*.bin')))

    if not bin_files:
        print(f"No .bin files found in {bin_dir}")
        return

    print(f"Found {len(bin_files)} .bin files")

    # Sample files
    if len(bin_files) > num_samples:
        step = len(bin_files) // num_samples
        sampled_files = bin_files[::step][:num_samples]
    else:
        sampled_files = bin_files

    print(f"Sampling {len(sampled_files)} files for analysis")

    # Collect all intensities
    all_intensities = []
    file_intensities = []
    file_names = []

    for bin_file in sampled_files:
        points = load_kitti_bin(bin_file)
        intensities = points[:, 3]

        all_intensities.extend(intensities)
        file_intensities.append(intensities)
        file_names.append(Path(bin_file).stem)

        print(f"  {Path(bin_file).name}: {len(intensities)} points, "
              f"intensity range [{intensities.min():.2f}, {intensities.max():.2f}], "
              f"mean={intensities.mean():.2f}, std={intensities.std():.2f}")

    all_intensities = np.array(all_intensities)

    # Statistics
    print(f"\n{'=' * 80}")
    print("OVERALL STATISTICS")
    print(f"{'=' * 80}")
    print(f"Total points: {len(all_intensities)}")
    print(f"Intensity range: [{all_intensities.min():.4f}, {all_intensities.max():.4f}]")
    print(f"Mean: {all_intensities.mean():.4f}")
    print(f"Std: {all_intensities.std():.4f}")
    print(f"Median: {np.median(all_intensities):.4f}")
    print(f"Zero intensity points: {np.sum(all_intensities == 0.0)} ({100 * np.sum(all_intensities == 0.0) / len(all_intensities):.2f}%)")

    # Percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\nPercentiles:")
    for p in percentiles:
        val = np.percentile(all_intensities, p)
        print(f"  {p:3d}th: {val:.4f}")

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Overall histogram
    ax = axes[0, 0]
    ax.hist(all_intensities, bins=100, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Intensity', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Overall Intensity Distribution (n={len(all_intensities)})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = f"Mean: {all_intensities.mean():.2f}\n"
    stats_text += f"Std: {all_intensities.std():.2f}\n"
    stats_text += f"Min: {all_intensities.min():.2f}\n"
    stats_text += f"Max: {all_intensities.max():.2f}"
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)

    # 2. Log-scale histogram
    ax = axes[0, 1]
    ax.hist(all_intensities, bins=100, color='coral', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Intensity', fontsize=12)
    ax.set_ylabel('Frequency (log scale)', fontsize=12)
    ax.set_yscale('log')
    ax.set_title('Intensity Distribution (Log Scale)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')

    # 3. Per-file comparison (box plot)
    ax = axes[1, 0]
    bp = ax.boxplot(file_intensities, labels=[f[:6] for f in file_names],
                     patch_artist=True, showfliers=False)
    for patch in bp['boxes']:
        patch.set_facecolor('lightgreen')
        patch.set_alpha(0.7)
    ax.set_xlabel('Sample Files', fontsize=12)
    ax.set_ylabel('Intensity', fontsize=12)
    ax.set_title(f'Intensity Distribution Across {len(sampled_files)} Samples', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Cumulative distribution
    ax = axes[1, 1]
    sorted_intensities = np.sort(all_intensities)
    cumulative = np.arange(1, len(sorted_intensities) + 1) / len(sorted_intensities)
    ax.plot(sorted_intensities, cumulative, linewidth=2, color='purple')
    ax.set_xlabel('Intensity', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title('Cumulative Distribution Function', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add percentile markers
    for p in [25, 50, 75, 95]:
        val = np.percentile(all_intensities, p)
        ax.axvline(val, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(val, 0.5, f'{p}th', rotation=90, verticalalignment='center', fontsize=8)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'intensity_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved visualization to: {output_path}")
    plt.close()

    # Create detailed histogram with zero/non-zero split
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: All data
    ax = axes[0]
    ax.hist(all_intensities, bins=100, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Intensity', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'All Intensities (n={len(all_intensities)})', fontsize=14, fontweight='bold')
    ax.axvline(all_intensities.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {all_intensities.mean():.2f}')
    ax.axvline(np.median(all_intensities), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(all_intensities):.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Non-zero only
    ax = axes[1]
    non_zero = all_intensities[all_intensities > 0]
    ax.hist(non_zero, bins=100, color='coral', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Intensity', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Non-Zero Intensities Only (n={len(non_zero)})', fontsize=14, fontweight='bold')
    ax.axvline(non_zero.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {non_zero.mean():.2f}')
    ax.axvline(np.median(non_zero), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(non_zero):.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path2 = os.path.join(output_dir, 'intensity_distribution_detailed.png')
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"✅ Saved detailed visualization to: {output_path2}")
    plt.close()

    print(f"\n{'=' * 80}")
    print("Analysis complete!")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Analyze intensity distribution in point clouds.")
    parser.add_argument('--bin-dir', type=str, required=True,
                        help='Directory containing .bin files')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of files to sample (default: 10)')
    parser.add_argument('--output-dir', type=str, default='./intensity_analysis',
                        help='Output directory for graphs (default: ./intensity_analysis)')

    args = parser.parse_args()

    analyze_intensity_distribution(
        bin_dir=args.bin_dir,
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )
