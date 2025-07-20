import numpy as np
from spaal2.core.dummy_lidar.dummy_lidar_vlp16_pcd import PcdLidarVLP16
from spaal2.core import (
    PreciseDuration, 
    visualize_comparison,
    DummyOutdoor,
    apply_noise,
    DummySpooferAdaptiveHFR
)

def run_pcd_lidar_example_with_hfr_spoofer():
    # Path to the PCD file
    # The user provided this path, assuming it exists.
    pcd_file_path = "C:/Users/nextr/spaal2-core/example/1464001237.670017000.pcd"

    # LiDAR position and orientation
    lidar_position = np.array([0.0, 0.0, 0.0])
    lidar_rotation = np.array([0.0, 0.0, 0.0])

    # Create a PcdLidarVLP16 instance
    lidar = PcdLidarVLP16(
        pcd_file_path=pcd_file_path,
        lidar_position=lidar_position,
        lidar_rotation=lidar_rotation,
        base_timestamp=PreciseDuration(nanoseconds=0),
        time_resolution_ns=0.2 # ここで分解能を設定
    )
    
    # Setup outdoor environment and HFR spoofer
    outdoor = DummyOutdoor(50.0, 0.8)
    spoofer = DummySpooferAdaptiveHFR(
        frequency=20 * 1e6, 
        duration=PreciseDuration(milliseconds=20),
        spoofer_distance_m=10.0,
        pulse_width=PreciseDuration(nanoseconds=5),
    )

    print("Running PCD LiDAR example with HFR spoofer...")

    point_list = []
    # Perform a full scan
    for i in range(lidar.max_index):
        try:
            config, signal = lidar.scan()

            # Trigger the spoofer at a specific angle (e.g., altitude 1 degree, azimuth ~180 degrees)
            # if config.altitude == 100 and abs(config.azimuth - 100) < 20:
            #     spoofer.trigger(config, signal)

            # # Apply outdoor conditions and noise to the legitimate signal
            # #signal = apply_noise(outdoor.apply(signal), ratio=0.1)
            
            # # Get the spoofer's signal and add noise
            # external_signal = apply_noise(spoofer.get_range_signal(config.start_timestamp, config.accept_duration), ratio=0.01)
            
            # # Combine the legitimate and spoofed signals
            # signal = np.maximum(signal, external_signal)
            # signal = np.clip(signal, 0, 9)

            # Receive points from the combined signal
            points = lidar.receive(config, signal)
            if points:
                point_list.extend(points)

        except StopIteration:
            print("End of scan.")
            break
    
    print(f"Detected {len(point_list)} points.")
    original_points = np.asarray(lidar.point_cloud.points)

    # --- Rotation Logic ---
    # Rotate the original point cloud by the calculated offset to check alignment.
    # We use the negative offset to align the PCD's starting angle with the simulation's 0-degree start.
    #angle_rad = np.deg2rad(-lidar.initial_azimuth_offset)
    angle_rad = np.deg2rad(90)  # Rotate by -90 degrees to align with the simulation's 0-degree start
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # Z-axis rotation matrix
    rotation_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,          0, 1]
    ])

    # Apply the rotation
    rotated_original_points = original_points.dot(rotation_matrix.T)
    # --- End Rotation Logic ---

    no_signal_points = lidar.get_no_signal_points()

    visualize_comparison(rotated_original_points, point_list, no_signal_points)

if __name__ == "__main__":
    run_pcd_lidar_example_with_hfr_spoofer()
