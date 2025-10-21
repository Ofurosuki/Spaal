import numpy as np
import blosc2
import json
import argparse

def find_missing_points(dataset_path: str, original_bin_path: str):
    """
    Find exactly where points are being lost in the reconstruction pipeline.
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
    total_loss = len(original_points) - np.count_nonzero(answer_matrix)
    print(f"Total loss: {total_loss:,} ({total_loss/len(original_points)*100:.2f}%)")

    # Calculate original point coordinates
    x, y, z = original_points[:, 0], original_points[:, 1], original_points[:, 2]
    distances = np.linalg.norm(original_points, axis=1)
    azimuth_deg = np.rad2deg(np.arctan2(x, y))
    elevation_deg = np.rad2deg(np.arcsin(z / distances))

    # Simulate the depth_map creation process (from dummy_lidar_hdl64e.py)
    vertical_angles = config['vertical_angles']
    initial_azimuth_offset = config['initial_azimuth_offset']

    # Key parameters
    horizontal_resolution_deg = 0.1
    horizontal_resolution = int(horizontal_resolution_deg * 100)  # 10

    print(f"\n=== Simulation Parameters ===")
    print(f"Horizontal resolution: {horizontal_resolution_deg}deg ({horizontal_resolution} in key units)")
    print(f"Initial azimuth offset: {initial_azimuth_offset:.6f}deg")
    print(f"Vertical angles (channels): {len(vertical_angles)}")

    # Find closest vertical angle for each point
    vertical_angles_np = np.array(vertical_angles)
    diffs = np.abs(elevation_deg[:, np.newaxis] - vertical_angles_np)
    closest_v_angle_indices = np.argmin(diffs, axis=1)
    closest_vertical_angles = vertical_angles_np[closest_v_angle_indices]

    # Create azimuth and altitude keys (matching _create_depth_map logic)
    azimuth_indices = np.round(azimuth_deg * 100 / horizontal_resolution)
    discretized_azimuths = azimuth_indices * horizontal_resolution
    azimuth_keys = (discretized_azimuths % 36000).astype(int)
    altitude_keys = (closest_vertical_angles * 100).astype(int)

    # Count unique keys
    keys = list(zip(azimuth_keys, altitude_keys))
    unique_keys = set(keys)

    print(f"\n=== Depth Map Creation Simulation ===")
    print(f"Original points: {len(original_points):,}")
    print(f"Unique grid keys: {len(unique_keys):,}")
    print(f"Points lost to grid collision: {len(original_points) - len(unique_keys):,} ({(len(original_points) - len(unique_keys))/len(original_points)*100:.2f}%)")

    # Now simulate the reconstruction (from hist_matrix_visualizer.py)
    time_resolution_ns = config['time_resolution_ns']
    fov = config['fov']
    channels, horizontal_steps = answer_matrix.shape

    print(f"\n=== Reconstruction Simulation ===")
    print(f"Answer matrix shape: {channels} channels x {horizontal_steps} steps")
    print(f"Expected horizontal steps from resolution: {int(360 / horizontal_resolution_deg)}")

    if horizontal_steps != int(360 / horizontal_resolution_deg):
        print(f"WARNING: Horizontal steps mismatch!")
        print(f"  Answer matrix has {horizontal_steps} steps")
        print(f"  Expected {int(360 / horizontal_resolution_deg)} steps for {horizontal_resolution_deg}deg resolution")

    # Create reconstructed keys
    reconstructed_keys = set()
    for v_idx in range(channels):
        v_angle = vertical_angles[v_idx]
        altitude_key_recon = int(v_angle * 100)

        for h_idx in range(horizontal_steps):
            if answer_matrix[v_idx, h_idx] > 0:
                # Calculate azimuth from reconstruction
                azimuth_deg_recon = (h_idx / horizontal_steps) * fov + initial_azimuth_offset

                # Discretize to match depth_map key generation
                azimuth_index_recon = round(azimuth_deg_recon * 100 / horizontal_resolution)
                discretized_azimuth_recon = (azimuth_index_recon * horizontal_resolution) % 36000
                azimuth_key_recon = int(discretized_azimuth_recon)

                reconstructed_keys.add((azimuth_key_recon, altitude_key_recon))

    print(f"Reconstructed unique keys: {len(reconstructed_keys):,}")

    # Find missing keys
    depth_map_keys = unique_keys
    missing_keys = depth_map_keys - reconstructed_keys
    extra_keys = reconstructed_keys - depth_map_keys

    print(f"\n=== Key Matching Analysis ===")
    print(f"Keys in depth_map (from original): {len(depth_map_keys):,}")
    print(f"Keys in reconstruction: {len(reconstructed_keys):,}")
    print(f"Missing keys (in depth_map but not reconstructed): {len(missing_keys):,}")
    print(f"Extra keys (reconstructed but not in depth_map): {len(extra_keys):,}")

    if len(missing_keys) > 0:
        print(f"\n=== Analysis of Missing Keys ===")

        # Analyze missing keys by altitude (channel)
        missing_altitudes = [k[1] for k in missing_keys]
        from collections import Counter
        altitude_counter = Counter(missing_altitudes)

        print(f"Missing keys by altitude (top 10):")
        for alt_key, count in altitude_counter.most_common(10):
            alt_deg = alt_key / 100
            print(f"  Altitude {alt_deg:6.2f}deg (key={alt_key}): {count:5d} missing keys")

        # Check if it's an azimuth pattern issue
        missing_azimuths = [k[0] for k in missing_keys]
        azimuth_range = max(missing_azimuths) - min(missing_azimuths)
        print(f"\nMissing azimuth range: {min(missing_azimuths)} to {max(missing_azimuths)} (span: {azimuth_range})")

    # Count points that would be lost due to missing keys
    points_in_missing_keys = sum(1 for k in keys if k in missing_keys)

    print(f"\n=== Point Loss Breakdown ===")
    print(f"Total original points: {len(original_points):,}")
    print(f"Lost to grid collision: {len(original_points) - len(unique_keys):,} ({(len(original_points) - len(unique_keys))/len(original_points)*100:.2f}%)")
    print(f"Lost to key mismatch: {points_in_missing_keys:,} ({points_in_missing_keys/len(original_points)*100:.2f}%)")
    print(f"Successfully reconstructed: {len(reconstructed_keys):,} ({len(reconstructed_keys)/len(original_points)*100:.2f}%)")
    print(f"Accounted for: {(len(original_points) - len(unique_keys)) + points_in_missing_keys + len(reconstructed_keys):,}")

    # Sample some missing keys to understand the pattern
    if len(missing_keys) > 0:
        print(f"\n=== Sample Missing Keys (first 20) ===")
        for i, (az_key, alt_key) in enumerate(sorted(missing_keys)[:20]):
            print(f"  ({az_key:6d}, {alt_key:5d}) -> azimuth={(az_key/100)%360:7.2f}deg, altitude={alt_key/100:6.2f}deg")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find where points are being lost")
    parser.add_argument("--dataset-path", type=str, required=True,
                       help="Path to dataset directory")
    parser.add_argument("--original-bin", type=str, required=True,
                       help="Path to original .bin file")

    args = parser.parse_args()
    find_missing_points(args.dataset_path, args.original_bin)
