import numpy as np
import blosc2
import json
import matplotlib.pyplot as plt
import argparse

def analyze_channel_recovery(dataset_path: str, original_bin_path: str):
    """
    Analyze reconstruction quality by vertical channel to identify systematic issues.
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

    # Calculate original point statistics
    distances = np.linalg.norm(original_points, axis=1)
    x, y, z = original_points[:, 0], original_points[:, 1], original_points[:, 2]
    elevation_angles = np.rad2deg(np.arcsin(z / distances))
    azimuth_angles = np.rad2deg(np.arctan2(x, y))

    # Get channel info
    vertical_angles = sorted(config['vertical_angles'], reverse=True)
    channels = len(vertical_angles)
    horizontal_resolution = answer_matrix.shape[1]

    print(f"\n=== Configuration ===")
    print(f"Vertical channels: {channels}")
    print(f"Horizontal resolution: {horizontal_resolution} steps")

    # For each vertical angle (channel), count original and reconstructed points
    channel_stats = []

    for ch_idx, v_angle in enumerate(vertical_angles):
        # Find original points closest to this channel
        angle_diffs = np.abs(elevation_angles - v_angle)

        # Points that would be assigned to this channel (closest match)
        # Use a threshold based on average channel spacing
        if ch_idx < len(vertical_angles) - 1:
            next_v_angle = vertical_angles[ch_idx + 1]
            threshold = abs(v_angle - next_v_angle) / 2
        else:
            threshold = 0.5  # Default threshold for last channel

        channel_mask = angle_diffs < threshold
        orig_count = channel_mask.sum()

        # Count reconstructed points in this channel
        recon_count = np.count_nonzero(answer_matrix[ch_idx, :])

        recovery_rate = (recon_count / orig_count * 100) if orig_count > 0 else 0

        channel_stats.append({
            'channel': ch_idx,
            'v_angle': v_angle,
            'orig_count': orig_count,
            'recon_count': recon_count,
            'recovery_rate': recovery_rate
        })

        if orig_count > 100:  # Only print channels with significant data
            status = "OK" if recovery_rate >= 80 else "WARN" if recovery_rate >= 60 else "POOR"
            print(f"Ch {ch_idx:2d} (v_angle={v_angle:6.2f}deg): "
                  f"Original={orig_count:5d}, Reconstructed={recon_count:5d}, "
                  f"Recovery={recovery_rate:5.1f}% {status}")

    # Analyze azimuth distribution
    print(f"\n=== Azimuth Analysis ===")
    azimuth_bins = np.arange(0, 360, 10)  # 10-degree bins
    azimuth_hist_orig, _ = np.histogram((azimuth_angles + 180) % 360, bins=azimuth_bins)

    # For reconstructed points
    time_resolution_ns = config['time_resolution_ns']
    initial_azimuth_offset = config['initial_azimuth_offset']
    fov = config['fov']

    reconstructed_azimuths = []
    for v_idx in range(channels):
        for h_idx in range(horizontal_resolution):
            peak_time = answer_matrix[v_idx, h_idx]
            if peak_time > 0:
                azimuth_deg = (h_idx / horizontal_resolution) * fov + initial_azimuth_offset
                reconstructed_azimuths.append(azimuth_deg)

    reconstructed_azimuths = np.array(reconstructed_azimuths)
    azimuth_hist_recon, _ = np.histogram((reconstructed_azimuths + 180) % 360, bins=azimuth_bins)

    azimuth_recovery = np.zeros_like(azimuth_hist_orig, dtype=float)
    for i in range(len(azimuth_bins) - 1):
        if azimuth_hist_orig[i] > 0:
            azimuth_recovery[i] = azimuth_hist_recon[i] / azimuth_hist_orig[i] * 100

    azimuth_bin_centers = (azimuth_bins[:-1] + azimuth_bins[1:]) / 2
    poor_azimuth_bins = azimuth_bin_centers[azimuth_recovery < 70]
    if len(poor_azimuth_bins) > 0:
        print(f"Azimuth ranges with <70% recovery: {poor_azimuth_bins}")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Recovery rate by channel
    ax = axes[0, 0]
    channel_indices = [s['channel'] for s in channel_stats]
    recovery_rates = [s['recovery_rate'] for s in channel_stats]
    ax.plot(channel_indices, recovery_rates, 'o-', linewidth=2, markersize=6)
    ax.axhline(y=80, color='g', linestyle='--', alpha=0.5, label='80% threshold')
    ax.axhline(y=60, color='orange', linestyle='--', alpha=0.5, label='60% threshold')
    ax.set_xlabel('Channel Index')
    ax.set_ylabel('Recovery Rate (%)')
    ax.set_title('Recovery Rate by Vertical Channel')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 2: Point count by channel
    ax = axes[0, 1]
    orig_counts = [s['orig_count'] for s in channel_stats]
    recon_counts = [s['recon_count'] for s in channel_stats]
    x = np.arange(len(channel_indices))
    width = 0.35
    ax.bar(x - width/2, orig_counts, width, label='Original', alpha=0.7)
    ax.bar(x + width/2, recon_counts, width, label='Reconstructed', alpha=0.7)
    ax.set_xlabel('Channel Index')
    ax.set_ylabel('Point Count')
    ax.set_title('Point Count by Vertical Channel')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Recovery rate by azimuth
    ax = axes[1, 0]
    ax.plot(azimuth_bin_centers, azimuth_recovery, 'o-', linewidth=2, markersize=4)
    ax.axhline(y=80, color='g', linestyle='--', alpha=0.5, label='80% threshold')
    ax.set_xlabel('Azimuth (degrees)')
    ax.set_ylabel('Recovery Rate (%)')
    ax.set_title('Recovery Rate by Azimuth')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 4: Vertical angle spacing
    ax = axes[1, 1]
    v_angle_diffs = np.abs(np.diff(vertical_angles))
    ax.plot(range(len(v_angle_diffs)), v_angle_diffs, 'o-', linewidth=2, markersize=4)
    ax.axhline(y=np.mean(v_angle_diffs), color='r', linestyle='--', alpha=0.5,
               label=f'Mean: {np.mean(v_angle_diffs):.3f}°')
    ax.set_xlabel('Channel Index')
    ax.set_ylabel('Vertical Angle Spacing (degrees)')
    ax.set_title('Vertical Angle Spacing Between Adjacent Channels')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig('channel_recovery_analysis.png', dpi=150)
    print(f"\n=== Plot saved to channel_recovery_analysis.png ===")

    # Identify problem channels
    print(f"\n=== Problem Channels (Recovery < 70%) ===")
    poor_channels = [s for s in channel_stats if s['recovery_rate'] < 70 and s['orig_count'] > 50]
    for s in poor_channels:
        print(f"Channel {s['channel']:2d} (v_angle={s['v_angle']:6.2f}deg): {s['recovery_rate']:.1f}%")

    # Check for patterns
    if len(poor_channels) > 0:
        poor_v_angles = [s['v_angle'] for s in poor_channels]
        print(f"\nV-angle range of problem channels: {min(poor_v_angles):.2f}deg to {max(poor_v_angles):.2f}deg")

        # Check if they're clustered
        if len(poor_channels) > 1:
            channel_indices_poor = [s['channel'] for s in poor_channels]
            consecutive_groups = []
            current_group = [channel_indices_poor[0]]
            for i in range(1, len(channel_indices_poor)):
                if channel_indices_poor[i] == channel_indices_poor[i-1] + 1:
                    current_group.append(channel_indices_poor[i])
                else:
                    if len(current_group) > 1:
                        consecutive_groups.append(current_group)
                    current_group = [channel_indices_poor[i]]
            if len(current_group) > 1:
                consecutive_groups.append(current_group)

            if consecutive_groups:
                print(f"\nWARNING: Consecutive problem channel groups detected:")
                for group in consecutive_groups:
                    print(f"   Channels {group[0]}-{group[-1]} ({len(group)} consecutive channels)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze reconstruction quality by channel")
    parser.add_argument("--dataset-path", type=str, required=True,
                       help="Path to dataset directory")
    parser.add_argument("--original-bin", type=str, required=True,
                       help="Path to original .bin file")

    args = parser.parse_args()
    analyze_channel_recovery(args.dataset_path, args.original_bin)
