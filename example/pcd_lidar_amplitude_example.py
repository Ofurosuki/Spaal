
import numpy as np
from spaal2.core.dummy_lidar.dummy_lidar_vlp16_pcd_amplitude import PcdLidarVLP16Amplitude
from spaal2.core import (
    PreciseDuration, 
    visualize_comparison,
    DummyOutdoor,
    apply_noise,
    DummySpooferAdaptiveHFR
)
import open3d as o3d
import matplotlib.pyplot as plt

def run_pcd_lidar_amplitude_example():
    # Path to the PCD file
    pcd_file_path = "C:/Users/nextr/spaal2-core/nuscenes_data/356d81f38dd9473ba590f39e266f54e5.pcd"

    # LiDAR position and orientation
    lidar_position = np.array([0.0, 0.0, 0.0])
    lidar_rotation = np.array([0.0, 0.0, 0.0])

    # Create a PcdLidarVLP16Amplitude instance with fingerprinting
    lidar = PcdLidarVLP16Amplitude(
        pcd_file_path=pcd_file_path,
        lidar_position=lidar_position,
        lidar_rotation=lidar_rotation,
        base_timestamp=PreciseDuration(nanoseconds=0),
        time_resolution_ns=0.2,
        # Fingerprint settings
        pulse_num=3,  # 3 pulses for fingerprinting
        min_interval=PreciseDuration(nanoseconds=50),
        max_interval=PreciseDuration(nanoseconds=100),
        consider_amp=True, # Use amplitude modulation
        min_amp_diff_ratio=0.3,
        max_amp_diff_ratio=0.5,
        max_torelance_error=PreciseDuration(nanoseconds=2)
    )
    
    # Setup outdoor environment and HFR spoofer
    outdoor = DummyOutdoor(50.0, 0.8)
    spoofer = DummySpooferAdaptiveHFR(
        frequency=20 * 1e6, 
        duration=PreciseDuration(milliseconds=20),
        spoofer_distance_m=10.0,
        pulse_width=PreciseDuration(nanoseconds=5),
        time_resolution_ns=lidar.time_resolution_ns,
    )

    print("Running PCD LiDAR with Amplitude Authentication example...")

    point_list = []
    echo_group_list = []
    # Perform a full scan
    for i in range(lidar.max_index):
        try:
            config, signal = lidar.scan()

            # Trigger the spoofer at a specific angle
            if config.altitude == 100 and abs(config.azimuth - 0) < 1000:
                spoofer.trigger(config, signal)

            # Apply outdoor conditions and noise
            signal = apply_noise(outdoor.apply(signal), ratio=0.1)
            external_signal = apply_noise(spoofer.get_range_signal(config.start_timestamp, config.accept_duration), ratio=0.01)
            
            signal_combined = np.maximum(signal, external_signal)
            signal_combined = np.clip(signal_combined, 0, 9)

            # Receive points and echo groups from the combined signal
            points, echo_groups = lidar.receive(config, [signal_combined])
            if points:
                point_list.extend(points)
                echo_group_list.extend(echo_groups)

        except StopIteration:
            print("End of scan.")
            break
    
    print(f"Detected {len(point_list)} points after authentication.")
    original_points = np.asarray(lidar.point_cloud.points)

    # Convert the list of VeloPoint objects to a NumPy array for visualization
    simulated_points_np = np.array([[p.x, p.y, p.z] for p in point_list])

    # Visualize the comparison between original and simulated points
    if simulated_points_np.any():
        visualize_comparison(original_points, simulated_points_np)
    else:
        print("No points were simulated.")

if __name__ == "__main__":
    run_pcd_lidar_amplitude_example()
