import numpy as np
import argparse

def analyze_horizontal_resolution(bin_file_path: str):
    """
    Analyze the actual horizontal (azimuth) resolution from KITTI point cloud data.
    """

    # Load point cloud
    raw_data = np.fromfile(bin_file_path, dtype=np.float32)

    # Detect format
    if raw_data.size % 4 == 0:
        points = raw_data.reshape(-1, 4)[:, :3]
    else:
        points = raw_data.reshape(-1, 5)[:, :3]

    print(f"Loaded {len(points):,} points from {bin_file_path}")

    # Calculate azimuth angles
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    azimuth_deg = np.rad2deg(np.arctan2(x, y))

    # Normalize to [0, 360)
    azimuth_deg = (azimuth_deg + 360) % 360

    # Sort by azimuth
    sorted_indices = np.argsort(azimuth_deg)
    sorted_azimuth = azimuth_deg[sorted_indices]

    # Calculate differences between consecutive points
    azimuth_diffs = np.diff(sorted_azimuth)

    # Handle wrap-around at 360/0 boundary
    # If difference is large (> 180), it's likely a wrap-around
    azimuth_diffs = np.where(azimuth_diffs > 180, 360 - azimuth_diffs, azimuth_diffs)
    azimuth_diffs = np.where(azimuth_diffs < -180, 360 + azimuth_diffs, azimuth_diffs)

    # Filter out very small differences (same azimuth, different elevation)
    # and very large differences (different scan lines)
    meaningful_diffs = azimuth_diffs[(azimuth_diffs > 0.01) & (azimuth_diffs < 1.0)]

    print(f"\n=== Azimuth Difference Statistics ===")
    print(f"Minimum azimuth step: {meaningful_diffs.min():.6f}°")
    print(f"Maximum azimuth step: {meaningful_diffs.max():.6f}°")
    print(f"Mean azimuth step:    {meaningful_diffs.mean():.6f}°")
    print(f"Median azimuth step:  {np.median(meaningful_diffs):.6f}°")
    print(f"Std deviation:        {meaningful_diffs.std():.6f}°")

    # Find the most common azimuth step (mode)
    # Bin the differences
    bins = np.arange(0, 1.0, 0.001)  # 0.001° bins
    hist, bin_edges = np.histogram(meaningful_diffs, bins=bins)
    mode_bin_idx = np.argmax(hist)
    mode_azimuth_step = (bin_edges[mode_bin_idx] + bin_edges[mode_bin_idx + 1]) / 2

    print(f"Most common step (mode): {mode_azimuth_step:.6f}°")
    print(f"Occurrences:            {hist[mode_bin_idx]:,}")

    # Estimate horizontal resolution (steps per 360°)
    estimated_resolution = 360 / mode_azimuth_step
    print(f"\n=== Estimated Horizontal Resolution ===")
    print(f"Steps per 360°: {estimated_resolution:.1f}")
    print(f"Degrees per step: {360/estimated_resolution:.6f}°")

    # Check if it's close to common LiDAR resolutions
    common_resolutions = {
        900: "0.4° (VLP-16 @ 5Hz)",
        1800: "0.2° (VLP-16 @ 10Hz, HDL-64E @ 10Hz)",
        2400: "0.15° (VLP-32 @ 20Hz)",
        3600: "0.1° (HDL-64E @ 20Hz)",
    }

    closest_resolution = min(common_resolutions.keys(),
                            key=lambda x: abs(x - estimated_resolution))

    print(f"\nClosest standard resolution: {closest_resolution} steps ({common_resolutions[closest_resolution]})")
    print(f"Difference: {abs(closest_resolution - estimated_resolution):.1f} steps")

    # Analyze by elevation angle groups
    print(f"\n=== Resolution by Elevation Angle ===")
    distances = np.linalg.norm(points, axis=1)
    elevation_angles = np.rad2deg(np.arcsin(z / distances))

    elevation_bins = [(-30, -20), (-20, -10), (-10, 0), (0, 5)]
    for elev_min, elev_max in elevation_bins:
        mask = (elevation_angles >= elev_min) & (elevation_angles < elev_max)
        if mask.sum() > 100:  # Only if we have enough points
            bin_azimuth = azimuth_deg[mask]
            bin_sorted = np.sort(bin_azimuth)
            bin_diffs = np.diff(bin_sorted)
            bin_diffs = bin_diffs[(bin_diffs > 0.01) & (bin_diffs < 1.0)]

            if len(bin_diffs) > 0:
                print(f"Elevation {elev_min:3d}° to {elev_max:3d}°: "
                      f"median step = {np.median(bin_diffs):.6f}°, "
                      f"points = {mask.sum():6d}")

    # Create visualization
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Histogram of azimuth steps
    ax = axes[0]
    ax.hist(meaningful_diffs, bins=100, edgecolor='black', alpha=0.7)
    ax.axvline(mode_azimuth_step, color='red', linestyle='--', linewidth=2,
               label=f'Mode: {mode_azimuth_step:.4f}°')
    ax.axvline(0.2, color='green', linestyle='--', linewidth=2,
               label='Assumed: 0.2°')
    ax.set_xlabel('Azimuth Step (degrees)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Azimuth Steps Between Consecutive Points')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Azimuth vs Point Index
    ax = axes[1]
    sample_indices = np.linspace(0, len(sorted_azimuth)-1, min(5000, len(sorted_azimuth)), dtype=int)
    ax.plot(sample_indices, sorted_azimuth[sample_indices], '.', markersize=1, alpha=0.5)
    ax.set_xlabel('Point Index (sorted by azimuth)')
    ax.set_ylabel('Azimuth (degrees)')
    ax.set_title('Azimuth Distribution (sampling)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('horizontal_resolution_analysis.png', dpi=150)
    print(f"\n=== Plot saved to horizontal_resolution_analysis.png ===")

    return mode_azimuth_step, estimated_resolution


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze horizontal resolution from KITTI bin file")
    parser.add_argument("--bin-file", type=str, required=True, help="Path to .bin file")

    args = parser.parse_args()

    mode_step, resolution = analyze_horizontal_resolution(args.bin_file)

    print(f"\n=== Summary ===")
    print(f"Actual horizontal resolution: {mode_step:.6f}° per step")
    print(f"Hardcoded assumption in code: 0.2° per step")

    if abs(mode_step - 0.2) > 0.01:
        print(f"⚠️  WARNING: Mismatch detected! ({abs(mode_step - 0.2):.4f}° difference)")
        print(f"This could explain some reconstruction quality issues.")
    else:
        print(f"✅ The 0.2° assumption is correct.")
