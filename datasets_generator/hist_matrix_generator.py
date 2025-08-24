import numpy as np
import os
import argparse
import glob
from spaal2.core import (
    PreciseDuration, DummyOutdoor, apply_noise, gen_sunlight,
)
from spaal2.core.dummy_lidar.dummy_lidar_vlp16 import DummyLidarVLP16
from spaal2.core.dummy_lidar.dummy_lidar_vlp16_pcd import PcdLidarVLP16
from spaal2.core.dummy_lidar.dummy_lidar_vlp32_pcd import PcdLidarVLP32c
from spaal2.core.dummy_spoofer.dummy_spoofer_adaptive_hfr_with_perturbation import DummySpooferAdaptiveHFRWithPerturbation
from spaal2.core.dummy_spoofer.dummy_spoofer_off import DummySpooferOff

class LidarSignalDatasetGenerator:
    def __init__(self, 
                 lidar_type: str = "VLP16",
                 pcd_directory: str = None,
                 output_dir: str = "./datasets",
                 outdoor_distance: float = 50.0, outdoor_ratio: float = 0.8,
                 spoofer_type: str = "adaptive_hfr_perturbation",
                 spoofer_frequency: float = 10 * 1e6,
                 spoofer_duration_ms: float = 20,
                 spoofer_distance_m: float = 10.0,
                 spoofer_pulse_width_ns: float = 5,
                 spoofer_perturbation_ns: float = 20.0,
                 spoofer_amplitude_range: tuple[float, float] = (5.0, 9.0),
                 lidar_amplitude_range: tuple[float, float] = (1.0, 7.0),
                 lidar_pulse_width_ns: float = 5,
                 time_resolution_ns: float = 1.0,
                 noise_ratio: float = 0.1,
                 sunlight_mean: float = 0.5,
                 spoofer_angle_deg: float = 0.0, 
                 spoofer_altitude_deg: float = 8.0):

        self.lidar_type = lidar_type
        self.pcd_directory = pcd_directory
        self.time_resolution_ns = time_resolution_ns
        self.spoofer_angle_deg = spoofer_angle_deg
        self.spoofer_altitude_deg = spoofer_altitude_deg

        if self.lidar_type == "VLP16":
            self.lidar = DummyLidarVLP16(
                amplitude=lidar_amplitude_range[0], 
                pulse_width=PreciseDuration(nanoseconds=lidar_pulse_width_ns),
                time_resolution_ns=self.time_resolution_ns
            )
            self.channels = 16
            self.horizontal_resolution = 1800
        elif self.lidar_type == "PCD_VLP16" or self.lidar_type == "PCD_VLP32c":
            if not self.pcd_directory or not os.path.isdir(self.pcd_directory):
                raise ValueError(f"PCD directory path must be provided and valid for {self.lidar_type} lidar type. Provided: {self.pcd_directory}")
            
            self.pcd_files = sorted(glob.glob(os.path.join(self.pcd_directory, '*.pcd')))
            if not self.pcd_files:
                raise ValueError(f"No PCD files found in {self.pcd_directory}")

            if self.lidar_type == "PCD_VLP16":
                lidar_class = PcdLidarVLP16
                self.channels = 16
            else:
                lidar_class = PcdLidarVLP32c
                self.channels = 32

            self.lidar = lidar_class(
                pcd_file_path=None, # Initialized without a specific file
                lidar_position=np.array([0.0, 0.0, 0.0]),
                lidar_rotation=np.array([0.0, 0.0, 0.0]),
                amplitude=lidar_amplitude_range[0],
                pulse_width=PreciseDuration(nanoseconds=lidar_pulse_width_ns),
                time_resolution_ns=self.time_resolution_ns
            )
            self.lidar.set_pcd_files(self.pcd_files)
            
            # Load the first frame to determine horizontal_resolution
            self.lidar.new_frame(frame_num=0)
            self.horizontal_resolution = self.lidar.max_index // self.channels
        else:
            raise ValueError(f"Unknown LiDAR model: {lidar_type}")

        self.samples_per_scan = int(self.lidar.accept_window.in_nanoseconds / self.lidar.time_resolution_ns)
        
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.outdoor = DummyOutdoor(outdoor_distance, outdoor_ratio)
        self.noise_ratio = noise_ratio
        self.sunlight_mean = sunlight_mean
        self.lidar_amplitude_range = lidar_amplitude_range
        self.spoofer_amplitude_range = spoofer_amplitude_range
        self.spoofer_type = spoofer_type

        if self.spoofer_type == "adaptive_hfr_perturbation":
            self.spoofer = DummySpooferAdaptiveHFRWithPerturbation(
                frequency=spoofer_frequency,
                duration=PreciseDuration(milliseconds=spoofer_duration_ms),
                spoofer_distance_m=spoofer_distance_m,
                pulse_width=PreciseDuration(nanoseconds=spoofer_pulse_width_ns),
                perturbation_ns=spoofer_perturbation_ns,
                amplitude=spoofer_amplitude_range[0],
                time_resolution_ns=self.lidar.time_resolution_ns
            )
        elif self.spoofer_type == "off":
            self.spoofer = DummySpooferOff()
        else:
            raise ValueError(f"Spoofer type {self.spoofer_type} not implemented for this script yet.")

    def generate(self, num_frames: int, filename_prefix: str = "lidar_signal"):
        all_frames_data = np.zeros((num_frames, self.channels, self.horizontal_resolution, self.samples_per_scan), dtype=np.float32)
        answer_matrix = np.zeros((num_frames, self.channels, self.horizontal_resolution), dtype=np.int32)
        all_initial_azimuth_offsets = []

        for frame_num in range(num_frames):
            print(f"Generating frame {frame_num + 1}/{num_frames}...")
            
            if self.lidar_type in ["PCD_VLP16", "PCD_VLP32c"]:
                current_lidar = self.lidar.new_frame(frame_num=frame_num, base_timestamp=PreciseDuration(nanoseconds=frame_num * 10**9))
            else:
                current_lidar = self.lidar.new_frame(base_timestamp=PreciseDuration(nanoseconds=frame_num * 10**9))

            if hasattr(current_lidar, 'initial_azimuth_offset'):
                all_initial_azimuth_offsets.append(current_lidar.initial_azimuth_offset)
            else:
                all_initial_azimuth_offsets.append(0.0)

            actual_trigger_point = None
            if self.spoofer_type != "off" and hasattr(current_lidar, 'depth_map'):
                # Convert user-facing angle (0-front, CCW) to internal angle (0-right, CCW)
                # by adding 90 degrees. The result is wrapped to [0, 360).
                internal_angle_deg = (self.spoofer_angle_deg + 90) % 360
                target_azimuth = internal_angle_deg * 100
                target_altitude = self.spoofer_altitude_deg * 100
                target_point = (target_azimuth, target_altitude)

                available_points = list(current_lidar.depth_map.keys())
                
                if not available_points:
                    print("Warning: Cannot determine spoofer trigger point, depth map is empty.")
                elif target_point in current_lidar.depth_map:
                    actual_trigger_point = target_point
                else:
                    # Find the closest point by Euclidean distance
                    distances = [np.sqrt((az - target_azimuth)**2 + (alt - target_altitude)**2) for az, alt in available_points]
                    closest_index = np.argmin(distances)
                    actual_trigger_point = available_points[closest_index]
                    print(f"Target spoofer point at {self.spoofer_angle_deg} deg (front=0, ccw) not found. Using closest point: az={actual_trigger_point[0]/100}, alt={actual_trigger_point[1]/100} deg")
            
            frame_data = np.zeros((self.channels, self.horizontal_resolution, self.samples_per_scan), dtype=np.float32)

            try:
                for i in range(current_lidar.max_index):
                    config, signal = current_lidar.scan()
                    # find candidate peaks in the signal
                    raises = np.flatnonzero(
                        (signal[:-1] < 0.01) & (signal[1:] >= 0.01)
                    ) + 1
                    
                    if len(raises) == 0:
                        true_peak_index = 0
                    else:
                        peaks = np.empty_like(raises, dtype=np.float64)
                        for i in range(len(raises)):
                            peaks[i] = np.max(
                                signal[raises[i]:min(len(signal), raises[i] + 50)]
                            )
                        highest_peak_index = np.argmax(peaks)
                        highest_peak = peaks[highest_peak_index]
                        highest_peak_time = raises[highest_peak_index]

                        true_peak_index = highest_peak_time

                    lidar_amp = np.random.uniform(self.lidar_amplitude_range[0], self.lidar_amplitude_range[1])
                    current_lidar.set_amplitude(lidar_amp)
                    
                    if self.spoofer_type != "off":
                        spoofer_amp = np.random.uniform(self.spoofer_amplitude_range[0], self.spoofer_amplitude_range[1])
                        self.spoofer.set_amplitude(spoofer_amp)

                        # Check if the current scan config matches the determined trigger point
                        if actual_trigger_point and config.altitude == actual_trigger_point[1] and config.azimuth == actual_trigger_point[0]:
                            print(f"Spoofer triggered for azimuth: {config.azimuth}, altitude: {config.altitude}")
                            self.spoofer.trigger(config, signal)
                        
                        external_signal = apply_noise(self.spoofer.get_range_signal(config.start_timestamp, config.accept_duration), ratio=0.01)
                        signal = np.maximum(signal, external_signal)

                    signal = np.clip(signal, 0, 9)

                    horizontal_index = (current_lidar.index -1) // self.channels
                    vertical_index = (current_lidar.index - 1) % self.channels

                    if horizontal_index < self.horizontal_resolution:
                        frame_data[vertical_index, horizontal_index, :] = signal
                        answer_matrix[frame_num, vertical_index, horizontal_index] = true_peak_index

            except StopIteration:
                pass

            all_frames_data[frame_num, :, :, :] = frame_data

        vertical_angles = self.lidar.vertical_angles
        fov = 360.0  # FOV for VLP16 is 360 degrees

        output_filename = os.path.join(self.output_dir, f"{filename_prefix}.npz")
        np.savez(output_filename, 
                 signals=all_frames_data, 
                 answer_matrix=answer_matrix,
                 initial_azimuth_offsets=np.array(all_initial_azimuth_offsets), 
                 vertical_angles=vertical_angles,
                 fov=fov,
                 time_resolution_ns=self.time_resolution_ns)
        print(f"Saved all frames to {output_filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate LiDAR signal datasets.")
    parser.add_argument("--lidar-type", type=str, default="VLP16", choices=["VLP16", "PCD_VLP16", "PCD_VLP32c"],
                        help="Type of LiDAR to use.")
    parser.add_argument("--pcd-directory", type=str, default=None,
                        help="Path to the directory containing PCD files, required if lidar-type starts with PCD.")
    parser.add_argument("--num-frames", type=int, default=1,
                        help="Number of frames to generate.")
    parser.add_argument("--output-dir", type=str, default="./lidar_datasets",
                        help="Directory to save the generated dataset.")
    parser.add_argument("--time-resolution-ns", type=float, default=1.0,
                        help="Time resolution in nanoseconds for the simulation.")
    parser.add_argument("--spoofer-type", type=str, default="adaptive_hfr_perturbation", choices=["adaptive_hfr_perturbation", "off"],
                        help="Type of spoofer to use.")
    # New arguments for spoofer targeting
    parser.add_argument("--spoofer-angle", type=float, default=0.0,
                        help="The angle for the spoofer trigger, in degrees, counter-clockwise with 0 at the front.")
    parser.add_argument("--spoofer-altitude", type=float, default=8.0,
                        help="The altitude for the spoofer trigger, in degrees.")

    args = parser.parse_args()

    if (args.lidar_type.startswith("PCD")) and not args.pcd_directory:
        parser.error(f"--pcd-directory is required when --lidar-type is {args.lidar_type}")

    generator = LidarSignalDatasetGenerator(
        lidar_type=args.lidar_type,
        pcd_directory=args.pcd_directory,
        output_dir=args.output_dir,
        time_resolution_ns=args.time_resolution_ns,
        spoofer_type=args.spoofer_type,
        spoofer_angle_deg=args.spoofer_angle,
        spoofer_altitude_deg=args.spoofer_altitude
    )
    generator.generate(num_frames=args.num_frames)
