import numpy as np
from spaal2.core import PreciseDuration
from spaal2.core.dummy_lidar import PcdLidarHDL64E

# KITTI dataset bin file path
bin_file_path = 'D:/testing/velodyne/000800.bin'

# Initialize HDL-64E with KITTI data
lidar_position = np.array([0.0, 0.0, 0.0])
lidar_rotation = np.eye(3)

try:
    lidar = PcdLidarHDL64E(
        pcd_file_path=bin_file_path,
        lidar_position=lidar_position,
        lidar_rotation=lidar_rotation,
        base_timestamp=PreciseDuration(nanoseconds=0),
        amplitude=1.0,
        pulse_width=PreciseDuration(nanoseconds=10),
        time_resolution_ns=1.0,
        intensity_to_amplitude_ratio=1.0/255.0,
        initial_point_offset=0,
        scan_mode='horizontal'
    )

    print(f"HDL-64E initialized successfully")
    print(f"Total points loaded: {len(lidar.points)}")
    print(f"Total scan points: {lidar.get_point_count()}")
    print(f"Depth map size: {len(lidar.depth_map)}")
    print(f"Initial azimuth offset: {lidar.initial_azimuth_offset:.2f} degrees")

    # Test scanning a few measurements
    print("\nTesting first 10 scans:")
    for i in range(10):
        try:
            config, signal = lidar.scan()
            print(f"Scan {i}: azimuth={config.azimuth/100:.2f}°, altitude={config.altitude/100:.2f}°, "
                  f"timestamp={config.start_timestamp.in_nanoseconds}ns, signal_max={signal.max():.4f}")

            # Test receiving
            points = lidar.receive(config, signal)
            if points:
                point = points[0]
                print(f"  -> Detected point: distance={point.distance_m:.2f}m, "
                      f"x={point.x:.2f}, y={point.y:.2f}, z={point.z:.2f}, intensity={point.intensity}")
        except StopIteration:
            print("Scan completed")
            break

    print(f"\nDetected point indices: {len(lidar.detected_point_indices)}")

except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please make sure the KITTI bin file exists at the specified path")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
