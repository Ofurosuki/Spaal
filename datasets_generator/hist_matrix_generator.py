import numpy as np
import os
import argparse
import glob
from typing import Tuple
from spaal2.core import (
    PreciseDuration, DummyOutdoor, apply_noise, gen_sunlight,
)
from spaal2.core.dummy_lidar.dummy_lidar_vlp16 import DummyLidarVLP16
from spaal2.core.dummy_lidar.dummy_lidar_vlp16_pcd import PcdLidarVLP16
from spaal2.core.dummy_lidar.dummy_lidar_vlp32_pcd import PcdLidarVLP32c
from spaal2.core.dummy_spoofer.dummy_spoofer_adaptive_hfr_with_perturbation import DummySpooferAdaptiveHFRWithPerturbation
from spaal2.core.dummy_spoofer.dummy_spoofer_off import DummySpooferOff

def get_peak_time_and_amplitude(signal: np.ndarray) -> Tuple[float, float]:
    """
    Finds the interpolated time and amplitude of the highest peak in a signal.
    Returns (0.0, 0.0) if no peak is found.
    """
    raises = np.flatnonzero((signal[:-1] < 0.01) & (signal[1:] >= 0.01)) + 1
    if len(raises) == 0:
        return 0.0, 0.0

    peak_values = np.array([np.max(signal[r:min(len(signal), r + 50)]) for r in raises])
    if len(peak_values) == 0:
        return 0.0, 0.0
    
    peak_amplitude = np.max(peak_values)
    if peak_amplitude < 0.01:
        return 0.0, 0.0

    highest_pulse_start_index = raises[np.argmax(peak_values)]
    pulse_region = signal[highest_pulse_start_index:min(len(signal), highest_pulse_start_index + 50)]
    
    if len(pulse_region) == 0:
        return 0.0, 0.0
        
    peak_idx_in_region = np.argmax(pulse_region)
    peak_idx_global = highest_pulse_start_index + peak_idx_in_region

    interpolated_time = float(peak_idx_global)
    if 0 < peak_idx_global < len(signal) - 1:
        y0, y1, y2 = signal[peak_idx_global - 1:peak_idx_global + 2]
        if y0 > 0 and y1 > 0 and y2 > 0:
            ln_y0, ln_y1, ln_y2 = np.log(y0), np.log(y1), np.log(y2)
            denominator = (ln_y0 - 2 * ln_y1 + ln_y2)
            if abs(denominator) > 1e-9:
                offset = (ln_y0 - ln_y2) / (2 * denominator)
                interpolated_time = peak_idx_global + offset
    
    return interpolated_time, peak_amplitude

class LidarSignalDatasetGenerator:
    def __init__(self, 
                 lidar_type: str = "PCD_VLP32c",
                 pcd_directory: str = None,
                 output_dir: str = "./datasets",
                 outdoor_distance: float = 50.0, outdoor_ratio: float = 0.8,
                 spoofer_type: str = "adaptive_hfr_perturbation",
                 spoofer_frequency: float = 10 * 1e6,
                 spoofer_duration_ms: float = 10,
                 spoofer_distance_m: float = 10.0,
                 spoofer_pulse_width_ns: float = 5,
                 spoofer_perturbation_ns: float = 20.0,
                 spoofer_amplitude_range: tuple[float, float] = (3.0, 8.0),
                 lidar_amplitude_range: tuple[float, float] = (3.0, 3.0),
                 lidar_pulse_width_ns: float = 5,
                 time_resolution_ns: float = 1.0,
                 noise_ratio: float = 0.1,
                 sunlight_mean: float = 0.5,
                 spoofer_angle_deg: float = 0.0, 
                 spoofer_altitude_deg: float = 8.0,
                 spoofer_width_deg: float = 90.0,
                 sync_angle_step_deg: float = 10, # Add new parameter
                 initial_point_offset: int = 0):

        self.lidar_type = lidar_type
        self.pcd_directory = pcd_directory
        self.time_resolution_ns = time_resolution_ns
        self.spoofer_angle_deg = spoofer_angle_deg
        self.spoofer_altitude_deg = spoofer_altitude_deg
        self.spoofer_width_deg = spoofer_width_deg
        self.sync_angle_step_deg = sync_angle_step_deg # Store the new parameter
        self.initial_point_offset = initial_point_offset

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
                time_resolution_ns=self.time_resolution_ns,
                initial_point_offset=self.initial_point_offset
            )
            if self.lidar_type == "PCD_VLP32c":
                self.lidar.set_sync_angle_step(self.sync_angle_step_deg)  # Use the new parameter
                print(f"Set VLP32c sync angle step to {self.sync_angle_step_deg} degrees.")
                #self.lidar.set_azimuth_time_perturbation([78,90,112],[20,20,20])
            self.lidar.set_pcd_files(self.pcd_files)
            
            # Load the first frame to determine horizontal_resolution
            self.lidar.new_frame(frame_num=0)
            self.horizontal_resolution = self.lidar.max_index // self.channels
        else:
            raise ValueError(f"Unknown LiDAR model: {lidar_type}")

        # Create a spatially sorted list of vertical angles for the output matrix
        self.sorted_vertical_angles = sorted(self.lidar.vertical_angles, reverse=True)
        self.altitude_to_sorted_v_idx_map = {int(angle * 100): i for i, angle in enumerate(self.sorted_vertical_angles)}

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
                time_resolution_ns=self.lidar.time_resolution_ns
            )
            self.spoofer.set_amplitude_range(spoofer_amplitude_range)
        elif self.spoofer_type == "off":
            self.spoofer = DummySpooferOff()
        else:
            raise ValueError(f"Spoofer type {self.spoofer_type} not implemented for this script yet.")

    def generate(self, num_frames: int, start_frame: int = 0, filename_prefix: str = "lidar_signal", save_to_file: bool = True):
        if self.pcd_directory:
            total_pcd_files = len(self.pcd_files)
            if start_frame >= total_pcd_files:
                print(f"Start frame {start_frame} is out of bounds. No files to process.")
                return None
            if start_frame + num_frames > total_pcd_files:
                print(f"Warning: Requested frames ({num_frames} from {start_frame}) exceeds available PCD files ({total_pcd_files}).")
                num_frames = total_pcd_files - start_frame
                print(f"Adjusting to process {num_frames} frames.")

        all_frames_data = np.zeros((num_frames, self.channels, self.horizontal_resolution, self.samples_per_scan), dtype=np.float32)
        all_labels_data = np.zeros((num_frames, self.channels, self.horizontal_resolution, self.samples_per_scan), dtype=np.uint8)
        answer_matrix = np.zeros((num_frames, self.channels, self.horizontal_resolution), dtype=np.float32)
        all_initial_azimuth_offsets = []

        # Define Spoofer's attack angle characteristics using internal angle representation
        internal_angle_deg = (self.spoofer_angle_deg) % 360
        spoofer_attack_center_az = internal_angle_deg * 100
        spoofer_attack_width_az = self.spoofer_width_deg * 100 # Convert to 0.01 deg units
        spoofer_attack_start_az = spoofer_attack_center_az - spoofer_attack_width_az / 2
        spoofer_attack_end_az = spoofer_attack_center_az + spoofer_attack_width_az / 2
        print(f"Spoofer attack cone is centered at {spoofer_attack_center_az/100} deg with width {self.spoofer_width_deg} deg (internal angle system).")
        print(f"start spoofer_attackstart_az: {spoofer_attack_start_az}, spoofer_attack_end_az: {spoofer_attack_end_az}")
        for i in range(num_frames):
            frame_idx = start_frame + i
            print(f"Generating frame {i + 1}/{num_frames} (PCD index: {frame_idx})...")
            
            if self.lidar_type in ["PCD_VLP16", "PCD_VLP32c"]:
                pcd_file_path = self.pcd_files[frame_idx]
                print(f"  - Using PCD file: {os.path.basename(pcd_file_path)}")
                current_lidar = self.lidar.new_frame(frame_num=frame_idx, base_timestamp=PreciseDuration(nanoseconds=frame_idx * 10**9))
            else:
                current_lidar = self.lidar.new_frame(base_timestamp=PreciseDuration(nanoseconds=frame_idx * 10**9))

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
            frame_labels = np.zeros((self.channels, self.horizontal_resolution, self.samples_per_scan), dtype=np.uint8)

            try:
                for scan_idx in range(current_lidar.max_index):
                    config, signal = current_lidar.scan()
                    
                    true_peak_time, _ = get_peak_time_and_amplitude(signal)
                    
                    # labeling: 0 = no return, 1 = legitimate return, 2 = HFR return
                    LEGITIMATE_PULSE = 1
                    HFR_PULSE = 2
                    current_labels = np.zeros_like(signal, dtype=np.uint8)              
                    current_labels[signal > 0.01] = LEGITIMATE_PULSE   # legitimate bin = 1

                    if self.spoofer_type != "off":
                        # Proximity-based trigger logic
                        if actual_trigger_point is not None and self.spoofer.trigger_time is None:
                            az_key_ideal = config.azimuth
                            alt_key_ideal = config.altitude
                            
                            az_key_target = actual_trigger_point[0]
                            alt_key_target = actual_trigger_point[1]

                            # Check if the current ideal scan angle is 'close' to the target trigger angle
                            # Tolerance is roughly half the step size. Azimuth step is 20 (0.2 deg).
                            azimuth_tolerance = 100
                            # Vertical steps vary, but 100 (1 deg) is a reasonable tolerance.
                            altitude_tolerance = 200

                            # Handle azimuth wraparound at 360 degrees (36000 units)
                            azimuth_diff = abs(az_key_ideal - az_key_target)
                            azimuth_diff = min(azimuth_diff, 36000 - azimuth_diff)

                            altitude_diff = abs(alt_key_ideal - alt_key_target)

                            if azimuth_diff <= azimuth_tolerance and altitude_diff <= altitude_tolerance:
                                # Trigger only once per attack.
                                self.spoofer.trigger(config, signal)
                        
                        # Default to no attack signal
                        external_signal = np.zeros_like(signal)
                        
                        # Check if spoofer is active and the current angle is within the attack cone
                        is_in_attack_angle = (spoofer_attack_start_az <= config.azimuth <= spoofer_attack_end_az)
                        if spoofer_attack_start_az < 0:
                            is_in_attack_angle = (config.azimuth >= (36000 + spoofer_attack_start_az) or config.azimuth <= spoofer_attack_end_az)
                        if self.spoofer.trigger_time is not None and is_in_attack_angle:
                            external_signal = apply_noise(self.spoofer.get_range_signal(config.start_timestamp, config.accept_duration), ratio=0.01)

                        current_labels = np.where(external_signal > signal, HFR_PULSE, current_labels)
                        signal = np.maximum(signal, external_signal)

                    signal = np.clip(signal, 0, 9)

                    # Convert config.azimuth (0-35999) to degrees (0-359.99)
                    azimuth_deg = config.azimuth / 100.0
                    
                    # Normalize azimuth by subtracting the initial offset to get the 'base' angle for this frame
                    # This correctly maps the angle to the horizontal index, counteracting the visualizer's addition of the offset.
                    normalized_azimuth = (azimuth_deg - current_lidar.initial_azimuth_offset + 360) % 360
                    
                    # Calculate horizontal_index based on the normalized angle
                    horizontal_index = int(normalized_azimuth / 360.0 * self.horizontal_resolution)

                    # # --- DEBUGGING BLOCK ---
                    # if self.spoofer_type != "off" and is_in_attack_angle:
                    #     print(f"[DEBUG] Attack Active: world_azimuth={config.azimuth/100:.2f}, h_idx={horizontal_index}, normalized_az={normalized_azimuth:.2f}")
                    # # --- END DEBUGGING BLOCK ---

                    vertical_index = self.altitude_to_sorted_v_idx_map.get(config.altitude)

                    if horizontal_index < self.horizontal_resolution and vertical_index is not None:
                        frame_data[vertical_index, horizontal_index, :] = signal
                        frame_labels[vertical_index, horizontal_index, :] = current_labels
                        answer_matrix[i, vertical_index, horizontal_index] = true_peak_time

            except StopIteration:
                pass

            all_frames_data[i, :, :, :] = frame_data
            all_labels_data[i, :, :, :] = frame_labels

        vertical_angles = self.sorted_vertical_angles
        fov = 360.0  # FOV for VLP16 is 360 degrees

        data_payload = {
            'signals': all_frames_data,
            'labels': all_labels_data,
            'answer_matrix': answer_matrix,
            'initial_azimuth_offsets': np.array(all_initial_azimuth_offsets),
            'vertical_angles': vertical_angles,
            'fov': fov,
            'time_resolution_ns': self.time_resolution_ns
        }

        if save_to_file:
            end_frame = start_frame + num_frames - 1
            output_filename_with_batch = f"{filename_prefix}_{start_frame}_to_{end_frame}.npz"
            output_filename = os.path.join(self.output_dir, output_filename_with_batch)

            np.savez_compressed(output_filename, **data_payload)
            print(f"Saved all frames to {output_filename}")
        
        return data_payload

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
    parser.add_argument("--spoofer-width-deg", type=float, default=90.0,
                        help="The angular width of the spoofer's attack cone in degrees.")
    parser.add_argument("--output-filename", type=str, default="lidar_signal",
                        help="Base name for the output .npz file.")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="Starting frame index (0-indexed) for processing PCD files.")
    parser.add_argument("--initial-point-offset", type=int, default=0,
                        help="Initial point offset to rotate the PCD point cloud.")

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
        spoofer_altitude_deg=args.spoofer_altitude,
        spoofer_width_deg=args.spoofer_width_deg,
        initial_point_offset=args.initial_point_offset
    )
    generator.generate(
        num_frames=args.num_frames,
        start_frame=args.start_frame,
        filename_prefix=args.output_filename,
        save_to_file=True
    )
