import numpy as np
import argparse
from sklearn.cluster import KMeans
from collections import defaultdict
import json

def extract_fire_angles_from_bin(bin_file_path: str, num_channels: int = 64, azimuth_resolution_deg: float = 0.2):
    """
    Extract fire angles (v_angle and h_offset) from a KITTI bin file.

    Parameters:
    -----------
    bin_file_path : str
        Path to the .bin file
    num_channels : int
        Number of laser channels (e.g., 64 for HDL-64E)
    azimuth_resolution_deg : float
        Expected azimuth resolution in degrees (e.g., 0.2 for HDL-64E)

    Returns:
    --------
    list of dict : List of fire angles with v_angle and h_offset
    """

    # Load point cloud
    raw_data = np.fromfile(bin_file_path, dtype=np.float32)

    # Detect format (4 or 5 elements per point)
    if raw_data.size % 4 == 0 and raw_data.size % 5 != 0:
        num_elements = 4
    elif raw_data.size % 5 == 0:
        num_elements = 5
    else:
        num_elements = 4
        raw_data = raw_data[:-(raw_data.size % 4)]

    points = raw_data.reshape(-1, num_elements)[:, :3]  # x, y, z
    print(f"Loaded {len(points)} points from {bin_file_path}")

    # Calculate spherical coordinates
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Distance (range)
    distance = np.linalg.norm(points, axis=1)

    # Filter out points at origin
    valid_mask = distance > 1e-6
    x, y, z = x[valid_mask], y[valid_mask], z[valid_mask]
    distance = distance[valid_mask]
    points = points[valid_mask]

    print(f"Valid points: {len(points)}")

    # Calculate elevation angle (vertical angle)
    elevation_deg = np.rad2deg(np.arcsin(z / distance))

    # Calculate azimuth angle
    azimuth_deg = np.rad2deg(np.arctan2(x, y))

    # Step 1: Cluster elevation angles to find vertical angles
    print(f"\nClustering elevation angles into {num_channels} channels...")
    kmeans = KMeans(n_clusters=num_channels, random_state=42, n_init=10)
    channel_labels = kmeans.fit_predict(elevation_deg.reshape(-1, 1))

    # Get cluster centers (vertical angles)
    vertical_angles = sorted(kmeans.cluster_centers_.flatten(), reverse=True)

    print(f"Detected vertical angles (degrees):")
    for i, v_angle in enumerate(vertical_angles):
        print(f"  Channel {i:2d}: {v_angle:8.4f}°")

    # Step 2: For each channel, analyze azimuth distribution to find h_offset
    print(f"\nAnalyzing azimuth offsets for each channel...")

    # Create mapping from cluster center to channel index
    center_to_channel = {center: i for i, center in enumerate(sorted(kmeans.cluster_centers_.flatten(), reverse=True))}

    fire_angles = []

    for cluster_idx in range(num_channels):
        # Get points in this cluster
        mask = channel_labels == cluster_idx
        cluster_points = points[mask]
        cluster_azimuths = azimuth_deg[mask]
        cluster_elevation = elevation_deg[mask]

        if len(cluster_points) == 0:
            print(f"  Channel {cluster_idx}: No points")
            continue

        # Get the vertical angle for this cluster
        v_angle = kmeans.cluster_centers_[cluster_idx, 0]

        # Find the channel index (sorted by descending v_angle)
        sorted_cluster_centers = sorted(kmeans.cluster_centers_.flatten(), reverse=True)
        # Find closest match (to handle floating point precision)
        channel_index = min(range(len(sorted_cluster_centers)),
                           key=lambda i: abs(sorted_cluster_centers[i] - v_angle))

        # Analyze azimuth distribution
        # Discretize azimuths to the expected resolution
        discretized_azimuths = np.round(cluster_azimuths / azimuth_resolution_deg) * azimuth_resolution_deg

        # Calculate the mode (most common azimuth offset pattern)
        # We look at the fractional part of azimuth / resolution
        azimuth_offsets = cluster_azimuths - discretized_azimuths

        # Use median offset as h_offset estimate
        h_offset = np.median(azimuth_offsets)

        # Alternative: use mode
        # hist, bin_edges = np.histogram(azimuth_offsets, bins=100)
        # h_offset = bin_edges[np.argmax(hist)]

        fire_angles.append({
            'channel': channel_index,
            'v_angle': float(v_angle),
            'h_offset': float(h_offset),
            'num_points': int(np.sum(mask)),
            'elevation_std': float(cluster_elevation.std())
        })

        print(f"  Channel {channel_index:2d}: v_angle={v_angle:8.4f}°, h_offset={h_offset:8.4f}°, points={np.sum(mask):6d}, std={cluster_elevation.std():.4f}°")

    # Sort by channel index
    fire_angles = sorted(fire_angles, key=lambda x: x['channel'])

    return fire_angles


def generate_python_code(fire_angles: list):
    """Generate Python code for fire_angles initialization"""
    print("\n" + "="*80)
    print("Generated Python code for fire_angles:")
    print("="*80)
    print("fire_angles: list[FireAngle] = [")

    for i, fa in enumerate(fire_angles):
        comment = f"  # Channel {fa['channel']}"
        print(f"    FireAngle({fa['v_angle']:14.10f}, {fa['h_offset']:14.10f}),{comment}")

    print("]")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract fire angles from point cloud bin file")
    parser.add_argument("--bin-file", type=str, required=True, help="Path to .bin file")
    parser.add_argument("--num-channels", type=int, default=64, help="Number of laser channels")
    parser.add_argument("--azimuth-resolution", type=float, default=0.2, help="Azimuth resolution in degrees")
    parser.add_argument("--output-json", type=str, default=None, help="Output JSON file path")

    args = parser.parse_args()

    fire_angles = extract_fire_angles_from_bin(
        args.bin_file,
        num_channels=args.num_channels,
        azimuth_resolution_deg=args.azimuth_resolution
    )

    # Generate Python code
    generate_python_code(fire_angles)

    # Save to JSON if requested
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(fire_angles, f, indent=2)
        print(f"\nSaved fire angles to {args.output_json}")

    # Statistics
    print(f"\nStatistics:")
    print(f"  Total channels: {len(fire_angles)}")
    print(f"  Vertical angle range: {min(fa['v_angle'] for fa in fire_angles):.2f}° to {max(fa['v_angle'] for fa in fire_angles):.2f}°")
    print(f"  H_offset range: {min(fa['h_offset'] for fa in fire_angles):.2f}° to {max(fa['h_offset'] for fa in fire_angles):.2f}°")
    print(f"  Total points analyzed: {sum(fa['num_points'] for fa in fire_angles)}")
