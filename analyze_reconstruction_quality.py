import numpy as np
import blosc2
import json
import matplotlib.pyplot as plt
import argparse

def analyze_reconstruction_quality(dataset_path: str, original_bin_path: str):
    """
    Analyze reconstruction quality by distance and elevation angle.
    """

    # Load reconstructed data
    with open(f'{dataset_path}/config.json', 'r') as f:
        config = json.load(f)

    with open(f'{dataset_path}/answer_matrix.bl2', 'rb') as f:
        answer_matrix = blosc2.unpack_array(f.read())

    # Load original point cloud
    raw_data = np.fromfile(original_bin_path, dtype=np.float32)
    if raw_data.size % 4 == 0:
        original_points = raw_data.reshape(-1, 4)[:, :3]
    else:
        original_points = raw_data.reshape(-1, 5)[:, :3]

    print(f"Original points: {len(original_points):,}")
    print(f"Reconstructed points: {np.count_nonzero(answer_matrix):,}")

    # Calculate statistics for original points
    distances = np.linalg.norm(original_points, axis=1)
    x, y, z = original_points[:, 0], original_points[:, 1], original_points[:, 2]
    elevation_angles = np.rad2deg(np.arcsin(z / distances))
    azimuth_angles = np.rad2deg(np.arctan2(x, y))

    # Analyze by distance bins
    distance_bins = np.arange(0, 100, 5)  # 5m bins
    distance_hist, _ = np.histogram(distances, bins=distance_bins)

    # Count reconstructed points by reconstructing from answer_matrix
    vertical_angles = config['vertical_angles']
    time_resolution_ns = config['time_resolution_ns']
    initial_azimuth_offset = config['initial_azimuth_offset']
    fov = config['fov']

    channels, horizontal_resolution = answer_matrix.shape

    reconstructed_points = []
    reconstructed_distances = []
    reconstructed_elevations = []

    for v_idx in range(channels):
        for h_idx in range(horizontal_resolution):
            peak_time = answer_matrix[v_idx, h_idx]
            if peak_time > 0:
                distance_m = peak_time * time_resolution_ns * 0.15
                altitude_deg = vertical_angles[v_idx]
                azimuth_deg = (h_idx / horizontal_resolution) * fov + initial_azimuth_offset

                reconstructed_distances.append(distance_m)
                reconstructed_elevations.append(altitude_deg)

                alpha = np.deg2rad(azimuth_deg)
                omega = np.deg2rad(altitude_deg)
                x = distance_m * np.cos(omega) * np.sin(alpha)
                y = distance_m * np.cos(omega) * np.cos(alpha)
                z = distance_m * np.sin(omega)
                reconstructed_points.append([x, y, z])

    reconstructed_points = np.array(reconstructed_points)
    reconstructed_distances = np.array(reconstructed_distances)
    reconstructed_elevations = np.array(reconstructed_elevations)

    # Count reconstructed points by distance
    reconstructed_distance_hist, _ = np.histogram(reconstructed_distances, bins=distance_bins)

    # Calculate recovery rate by distance
    recovery_rate_by_distance = np.zeros_like(distance_hist, dtype=float)
    for i in range(len(distance_bins) - 1):
        if distance_hist[i] > 0:
            recovery_rate_by_distance[i] = reconstructed_distance_hist[i] / distance_hist[i] * 100

    # Analyze by elevation angle bins
    elevation_bins = np.arange(-30, 5, 2)  # 2-degree bins
    elevation_hist, _ = np.histogram(elevation_angles, bins=elevation_bins)
    reconstructed_elevation_hist, _ = np.histogram(reconstructed_elevations, bins=elevation_bins)

    recovery_rate_by_elevation = np.zeros_like(elevation_hist, dtype=float)
    for i in range(len(elevation_bins) - 1):
        if elevation_hist[i] > 0:
            recovery_rate_by_elevation[i] = reconstructed_elevation_hist[i] / elevation_hist[i] * 100

    # Analyze angular spacing
    print(f"\n=== Angular Spacing Analysis ===")
    print(f"Horizontal resolution: {horizontal_resolution} steps (360° / {360/horizontal_resolution:.2f}° per step)")
    print(f"Vertical channels: {channels}")
    sorted_v_angles = sorted(vertical_angles, reverse=True)
    v_angle_diffs = np.diff(sorted_v_angles)
    print(f"Vertical angle spacing: min={abs(v_angle_diffs).min():.3f}°, max={abs(v_angle_diffs).max():.3f}°, mean={abs(v_angle_diffs).mean():.3f}°")

    # Calculate point density (points per solid angle)
    print(f"\n=== Point Density Analysis ===")
    for dist_range in [(0, 20), (20, 40), (40, 60), (60, 100)]:
        mask_orig = (distances >= dist_range[0]) & (distances < dist_range[1])
        mask_recon = (reconstructed_distances >= dist_range[0]) & (reconstructed_distances < dist_range[1])

        if mask_orig.sum() > 0:
            orig_count = mask_orig.sum()
            recon_count = mask_recon.sum()
            recovery = recon_count / orig_count * 100 if orig_count > 0 else 0
            print(f"Distance {dist_range[0]:3d}-{dist_range[1]:3d}m: "
                  f"Original={orig_count:6d}, Reconstructed={recon_count:6d}, Recovery={recovery:5.1f}%")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Point count by distance
    ax = axes[0, 0]
    bin_centers = (distance_bins[:-1] + distance_bins[1:]) / 2
    ax.bar(bin_centers - 1, distance_hist, width=4, alpha=0.5, label='Original', color='blue')
    ax.bar(bin_centers + 1, reconstructed_distance_hist, width=4, alpha=0.5, label='Reconstructed', color='red')
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Point Count')
    ax.set_title('Point Distribution by Distance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Recovery rate by distance
    ax = axes[0, 1]
    ax.plot(bin_centers, recovery_rate_by_distance, 'o-', linewidth=2, markersize=6)
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Recovery Rate (%)')
    ax.set_title('Recovery Rate by Distance')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='70% threshold')
    ax.legend()

    # Plot 3: Point count by elevation
    ax = axes[1, 0]
    elevation_bin_centers = (elevation_bins[:-1] + elevation_bins[1:]) / 2
    ax.bar(elevation_bin_centers - 0.4, elevation_hist, width=0.8, alpha=0.5, label='Original', color='blue')
    ax.bar(elevation_bin_centers + 0.4, reconstructed_elevation_hist, width=0.8, alpha=0.5, label='Reconstructed', color='red')
    ax.set_xlabel('Elevation Angle (degrees)')
    ax.set_ylabel('Point Count')
    ax.set_title('Point Distribution by Elevation Angle')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Recovery rate by elevation
    ax = axes[1, 1]
    ax.plot(elevation_bin_centers, recovery_rate_by_elevation, 'o-', linewidth=2, markersize=6)
    ax.set_xlabel('Elevation Angle (degrees)')
    ax.set_ylabel('Recovery Rate (%)')
    ax.set_title('Recovery Rate by Elevation Angle')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='70% threshold')
    ax.legend()

    plt.tight_layout()
    plt.savefig('reconstruction_quality_analysis.png', dpi=150)
    print(f"\n=== Plot saved to reconstruction_quality_analysis.png ===")

    # Identify problem areas
    print(f"\n=== Problem Areas ===")
    poor_distance_bins = bin_centers[recovery_rate_by_distance < 70]
    if len(poor_distance_bins) > 0:
        print(f"Distance bins with <70% recovery: {poor_distance_bins}")

    poor_elevation_bins = elevation_bin_centers[recovery_rate_by_elevation < 70]
    if len(poor_elevation_bins) > 0:
        print(f"Elevation bins with <70% recovery: {poor_elevation_bins}")

    return {
        'distance_bins': bin_centers,
        'recovery_by_distance': recovery_rate_by_distance,
        'elevation_bins': elevation_bin_centers,
        'recovery_by_elevation': recovery_rate_by_elevation,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze reconstruction quality")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to dataset directory (e.g., ./lidar_datasets_hdl64e_extracted/000000)")
    parser.add_argument("--original-bin", type=str, required=True, help="Path to original .bin file")

    args = parser.parse_args()

    analyze_reconstruction_quality(args.dataset_path, args.original_bin)
