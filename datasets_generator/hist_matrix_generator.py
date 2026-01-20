import numpy as np
import os
import argparse
import glob
from typing import Tuple, Dict
from spaal2.core import (
    PreciseDuration, DummyOutdoor, apply_noise, gen_sunlight,
)
from spaal2.core.dummy_lidar.dummy_lidar_vlp16 import DummyLidarVLP16
from spaal2.core.dummy_lidar.dummy_lidar_vlp16_pcd import PcdLidarVLP16
from spaal2.core.dummy_lidar.dummy_lidar_vlp32_pcd import PcdLidarVLP32c
from spaal2.core.dummy_lidar.dummy_lidar_hdl64e import PcdLidarHDL64E
from spaal2.core.dummy_spoofer.dummy_spoofer_adaptive_hfr_with_perturbation import DummySpooferAdaptiveHFRWithPerturbation
from spaal2.core.dummy_spoofer.dummy_spoofer_off import DummySpooferOff
from tqdm import tqdm
import json
import blosc2

#@njit(fastmath=True)

#@njit(fastmath=True)
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

def reconstruct_point_cloud_from_memory(
    hist_matrix: np.ndarray,
    config_data: Dict,
    azimuth_angles: np.ndarray,
    amplitude_to_intensity_ratio: float = 255.0/10.0,
    use_answer_matrix: bool = False,
    answer_matrix: np.ndarray = None
) -> np.ndarray:
    """
    Reconstruct point cloud from in-memory data.
    """
    initial_azimuth_offset = config_data.get('initial_azimuth_offset', 0.0)
    vertical_angles = config_data.get('vertical_angles', [])
    fov = config_data.get('fov', 360.0)
    time_resolution_ns = config_data.get('time_resolution_ns', 1.0)

    is_prediction_local = len(hist_matrix.shape) == 2

    if is_prediction_local:
        channels, horizontal_resolution = hist_matrix.shape
    else:
        channels, horizontal_resolution, _ = hist_matrix.shape

    points = []

    for v_idx in range(channels):
        for h_idx in range(horizontal_resolution):
            if use_answer_matrix and answer_matrix is not None:
                highest_peak_time = answer_matrix[v_idx, h_idx]
                if highest_peak_time <= 0:
                    continue
                # Since we don't have amplitude from answer_matrix, we can't calculate real intensity.
                # We can either use a default value or try to get amplitude from signal.
                # For now, let's try to get it from the signal matrix if available.
                if not is_prediction_local:
                    signal = hist_matrix[v_idx, h_idx, :]
                    _, peak_amplitude = get_peak_time_and_amplitude(signal)
                    intensity = np.clip(peak_amplitude * amplitude_to_intensity_ratio, 0, 255)
                else:
                    intensity = 100 # Default for prediction
            elif not is_prediction_local:
                signal = hist_matrix[v_idx, h_idx, :]
                highest_peak_time, peak_amplitude = get_peak_time_and_amplitude(signal)

                if highest_peak_time == 0.0:
                    continue
                intensity = np.clip(peak_amplitude * amplitude_to_intensity_ratio, 0, 255)
            else: # is_prediction_local but not use_answer_matrix
                highest_peak_time = hist_matrix[v_idx, h_idx]
                if highest_peak_time <= 0:
                    continue
                intensity = 100  # Default intensity for predictions

            distance_m = (highest_peak_time * time_resolution_ns) * 0.15
            altitude_deg = vertical_angles[v_idx]

            if azimuth_angles is not None:
                actual_azimuth = azimuth_angles[v_idx, h_idx]
                if not np.isnan(actual_azimuth):
                    azimuth_deg = actual_azimuth
                else:
                    azimuth_deg = (h_idx / horizontal_resolution) * fov + initial_azimuth_offset
            else:
                azimuth_deg = (h_idx / horizontal_resolution) * fov + initial_azimuth_offset

            alpha = np.deg2rad(azimuth_deg)
            omega = np.deg2rad(altitude_deg)

            x = distance_m * np.cos(omega) * np.sin(alpha)
            y = distance_m * np.cos(omega) * np.cos(alpha)
            z = distance_m * np.sin(omega)

            points.append([x, y, z, intensity, v_idx])

    return np.array(points, dtype=np.float32)

class LidarSignalDatasetGenerator:
    def __init__(self,
                 lidar_type: str = "PCD_VLP32c",
                 pcd_directory: str = None,
                 json_path: str = None,
                 output_dir: str = "./datasets",
                 outdoor_distance: float = 50.0, outdoor_ratio: float = 0.8,
                 spoofer_type: str = "adaptive_hfr_perturbation",
                 spoofer_frequency: float = 10 * 1e6,
                 spoofer_duration_ms: float = 100,
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
                 sync_angle_range: tuple[float, float] = (0.2, 0.2),
                 initial_point_offset: int = 0,
                 horizontal_resolution_deg: float = 0.1,
                 output_horizontal_resolution_deg: float = None,
                 scan_mode: str = 'vertical',
                 output_channels: int = None,
                 bin_format: str = 'nuscenes'):

        self.lidar_type = lidar_type
        self.bin_format = bin_format

        # Auto-detect output_horizontal_resolution_deg if not provided
        if output_horizontal_resolution_deg is None:
            if lidar_type == "PCD_HDL64E":
                output_horizontal_resolution_deg = 0.0818  # 4400 samples
            elif lidar_type in ["PCD_VLP32c", "PCD_VLP16", "VLP16"]:
                output_horizontal_resolution_deg = 0.2  # 1800 samples
            else:
                output_horizontal_resolution_deg = 0.2  # Default

        # Auto-detect output_channels if not provided
        if output_channels is None:
            if lidar_type == "PCD_HDL64E":
                output_channels = 64  # Default: use all 64 channels
            elif lidar_type in ["PCD_VLP32c", "PCD_VLP16"]:
                output_channels = 32  # VLP32c has 32 channels
            elif lidar_type == "VLP16":
                output_channels = 16  # VLP16 has 16 channels
            else:
                output_channels = 32  # Default

        self.output_channels = output_channels

        self.pcd_directory = pcd_directory
        self.json_path = json_path
        self.time_resolution_ns = time_resolution_ns
        self.spoofer_angle_deg = spoofer_angle_deg
        self.spoofer_altitude_deg = spoofer_altitude_deg
        self.spoofer_width_deg = spoofer_width_deg
        self.sync_angle_range = sync_angle_range  # Store sync angle range (min, max)
        self.initial_point_offset = initial_point_offset
        self.horizontal_resolution_deg = horizontal_resolution_deg
        self.output_horizontal_resolution_deg = output_horizontal_resolution_deg
        self.scan_mode = scan_mode  # Store scan mode

        if self.lidar_type == "VLP16":
            self.lidar = DummyLidarVLP16(
                amplitude=lidar_amplitude_range[0], 
                pulse_width=PreciseDuration(nanoseconds=lidar_pulse_width_ns),
                time_resolution_ns=self.time_resolution_ns
            )
            self.channels = 16
            self.horizontal_resolution = 1800
        elif self.lidar_type == "PCD_VLP16" or self.lidar_type == "PCD_VLP32c" or self.lidar_type == "PCD_HDL64E":
            if self.json_path:
                if not os.path.exists(self.json_path):
                    raise FileNotFoundError(f"Input JSON file not found at {self.json_path}")
                with open(self.json_path, 'r') as f:
                    self.samples = json.load(f)
                if not self.samples:
                    raise ValueError(f"No samples found in {self.json_path}")
                self.pcd_files = [item['path'] for item in self.samples]
            elif self.pcd_directory:
                if not os.path.isdir(self.pcd_directory):
                    raise ValueError(f"PCD directory path must be a valid directory. Provided: {self.pcd_directory}")
                # Support both .pcd and .bin files
                pcd_pattern = os.path.join(self.pcd_directory, '*.pcd')
                bin_pattern = os.path.join(self.pcd_directory, '*.bin')
                self.pcd_files = sorted(glob.glob(pcd_pattern) + glob.glob(bin_pattern))
                if not self.pcd_files:
                    raise ValueError(f"No PCD or BIN files found in {self.pcd_directory}")
                self.samples = [{'path': path, 'token': os.path.splitext(os.path.basename(path))[0]} for path in self.pcd_files]
            else:
                raise ValueError("Either --json-path or --pcd-directory must be provided for PCD lidar types.")

            if self.lidar_type == "PCD_VLP16":
                lidar_class = PcdLidarVLP16
                self.channels = self.output_channels
            elif self.lidar_type == "PCD_VLP32c":
                lidar_class = PcdLidarVLP32c
                self.channels = self.output_channels
            else:  # PCD_HDL64E
                lidar_class = PcdLidarHDL64E
                self.channels = self.output_channels

            # Build initialization parameters (common to all PCD-based LiDARs)
            init_params = {
                'pcd_file_path': None,  # Initialized without a specific file
                'lidar_position': np.array([0.0, 0.0, 0.0]),
                'lidar_rotation': np.array([0.0, 0.0, 0.0]),
                'amplitude': lidar_amplitude_range[0],
                'pulse_width': PreciseDuration(nanoseconds=lidar_pulse_width_ns),
                'time_resolution_ns': self.time_resolution_ns,
                'initial_point_offset': self.initial_point_offset,
                'scan_mode': self.scan_mode,
            }

            # HDL64E-specific parameters
            if self.lidar_type == "PCD_HDL64E":
                init_params['horizontal_resolution_deg'] = self.horizontal_resolution_deg
                init_params['output_horizontal_resolution_deg'] = self.output_horizontal_resolution_deg
                init_params['output_channels'] = self.output_channels

            self.lidar = lidar_class(**init_params)
            if self.lidar_type == "PCD_VLP32c" or self.lidar_type == "PCD_HDL64E":
                if self.scan_mode == 'vertical':
                    print(f"LiDAR {self.lidar_type} (vertical mode): sync range {int(self.sync_angle_range[0])}-{int(self.sync_angle_range[1])} channels")
                else:
                    print(f"LiDAR {self.lidar_type} (horizontal mode): sync range {self.sync_angle_range[0]:.2f}°-{self.sync_angle_range[1]:.2f}°")
                #self.lidar.set_azimuth_time_perturbation([78,90,112],[20,20,20])

            if self.lidar_type == "PCD_HDL64E":
                expected_samples = int(360 / self.output_horizontal_resolution_deg)
                print(f"HDL-64E output resolution: {self.output_horizontal_resolution_deg:.4f}° ({expected_samples} samples per channel)")
            self.lidar.set_pcd_files(self.pcd_files)
            
            # Load the first frame to determine horizontal_resolution
            self.lidar.new_frame(frame_num=0)
            self.horizontal_resolution = self.lidar.max_index // self.channels
        else:
            raise ValueError(f"Unknown LiDAR model: {lidar_type}")

        # Create a spatially sorted list of vertical angles for the output matrix
        # Use the LiDAR's sorted_vertical_angles which already accounts for channel_mapping
        self.sorted_vertical_angles = self.lidar.sorted_vertical_angles

        # Print output channels info (only for LiDARs that support it)
        if hasattr(self.lidar, 'output_channels'):
            print(f"Using {len(self.sorted_vertical_angles)} vertical angles (LiDAR output channels: {self.lidar.output_channels})")
        else:
            print(f"Using {len(self.sorted_vertical_angles)} vertical angles")

        self.altitude_to_sorted_v_idx_map = {int(angle * 100): i for i, angle in enumerate(self.sorted_vertical_angles)}

        self.samples_per_scan = int(self.lidar.accept_window.in_nanoseconds / self.lidar.time_resolution_ns)
        
        # Create base output directory including the sync angle
        sync_angle_dir_name = str(self.sync_angle_range[0])
        self.output_dir = os.path.join(output_dir, sync_angle_dir_name)
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

    def _save_frame_data(self,
                         sample_token: str,
                         status: str,
                         signal_matrix: np.ndarray,
                         labels_matrix: np.ndarray,
                         answer_matrix: np.ndarray,
                         azimuth_angles: np.ndarray,
                         timestamp_matrix: np.ndarray,
                         config_data: dict):
        """Helper function to save one version (attacked/noattack) of the frame data."""
        frame_output_dir = os.path.join(self.output_dir, sample_token, status)
        bl2_dir = os.path.join(frame_output_dir, 'bl2')
        bin_dir = os.path.join(frame_output_dir, 'bin')
        os.makedirs(bl2_dir, exist_ok=True)
        os.makedirs(bin_dir, exist_ok=True)

        # Save data using blosc2
        with open(os.path.join(bl2_dir, 'signal.bl2'), 'wb') as f:
            f.write(blosc2.pack_array(signal_matrix))
        with open(os.path.join(bl2_dir, 'labels.bl2'), 'wb') as f:
            f.write(blosc2.pack_array(labels_matrix))
        with open(os.path.join(bl2_dir, 'answer_matrix.bl2'), 'wb') as f:
            f.write(blosc2.pack_array(answer_matrix))
        with open(os.path.join(bl2_dir, 'angles.bl2'), 'wb') as f:
            f.write(blosc2.pack_array(azimuth_angles))
        with open(os.path.join(bl2_dir, 'timestamps.bl2'), 'wb') as f:
            f.write(blosc2.pack_array(timestamp_matrix))

        # Save config to json
        with open(os.path.join(bl2_dir, 'config.json'), 'w') as f:
            json.dump(config_data, f, indent=2)

        # Generate and save .bin file from memory
        point_cloud = reconstruct_point_cloud_from_memory(
            hist_matrix=signal_matrix,
            config_data=config_data,
            azimuth_angles=azimuth_angles,
            use_answer_matrix=False, # Or True, depending on desired output
            answer_matrix=answer_matrix
        )

        if len(point_cloud) > 0:
            if self.bin_format == 'kitti':
                output_data = point_cloud[:, :4]
            else: # nuscenes
                output_data = point_cloud
            
            output_path = os.path.join(bin_dir, f"{sample_token}.bin")
            output_data.tofile(output_path)

    def generate(self, num_frames: int, start_frame: int = 0, filename_prefix: str = "lidar_signal", save_to_file: bool = True):
        if self.pcd_directory or self.json_path:
            total_files = len(self.pcd_files)
            if start_frame >= total_files:
                print(f"Start frame {start_frame} is out of bounds. No files to process.")
                return
            
            if num_frames == -1:
                num_frames = total_files - start_frame
            elif start_frame + num_frames > total_files:
                print(f"Warning: Requested frames ({num_frames} from {start_frame}) exceeds available files ({total_files}).")
                num_frames = total_files - start_frame
                print(f"Adjusting to process {num_frames} frames.")

        # Define Spoofer's attack angle characteristics using internal angle representation
        internal_angle_deg = (self.spoofer_angle_deg) % 360
        spoofer_attack_center_az = internal_angle_deg * 100
        spoofer_attack_width_az = self.spoofer_width_deg * 100 # Convert to 0.01 deg units
        spoofer_attack_start_az = spoofer_attack_center_az - spoofer_attack_width_az / 2
        spoofer_attack_end_az = spoofer_attack_center_az + spoofer_attack_width_az / 2
        print(f"Spoofer attack cone is centered at {spoofer_attack_center_az/100} deg with width {self.spoofer_width_deg} deg (internal angle system).")
        print(f"start spoofer_attackstart_az: {spoofer_attack_start_az}, spoofer_attack_end_az: {spoofer_attack_end_az}")

        for i in tqdm(range(num_frames), desc="Generating frames"):
            frame_idx = start_frame + i
            sample_info = self.samples[frame_idx]
            sample_token = sample_info['token']

            # Randomly select sync value from the specified range for each frame
            if self.lidar_type in ["PCD_VLP32c", "PCD_HDL64E"]:
                sync_value = np.random.uniform(self.sync_angle_range[0], self.sync_angle_range[1])

                if self.scan_mode == 'vertical':
                    # Vertical mode: interpret as channel steps (integer)
                    sync_channel = int(sync_value)
                    self.lidar.set_sync_channel_step(sync_channel)
                    #print(f"Frame {frame_idx}: Using sync {sync_channel} channels (range: {int(self.sync_angle_range[0])}-{int(self.sync_angle_range[1])} channels, vertical mode)")
                else:
                    # Horizontal mode: interpret as angle in degrees (float)
                    self.lidar.set_sync_angle_step(sync_value)
                    #print(f"Frame {frame_idx}: Using sync {sync_value:.4f}° (range: {self.sync_angle_range[0]:.2f}°-{self.sync_angle_range[1]:.2f}°, horizontal mode)")

            if self.lidar_type in ["PCD_VLP16", "PCD_VLP32c", "PCD_HDL64E"]:
                current_lidar = self.lidar.new_frame(frame_num=frame_idx, base_timestamp=PreciseDuration(nanoseconds=frame_idx * 10**9))
            else:
                current_lidar = self.lidar.new_frame(base_timestamp=PreciseDuration(nanoseconds=frame_idx * 10**9))

            initial_azimuth_offset = current_lidar.initial_azimuth_offset if hasattr(current_lidar, 'initial_azimuth_offset') else 0.0

            actual_trigger_point = None
            if self.spoofer_type != "off" and hasattr(current_lidar, 'depth_map'):
                # Convert user-facing angle (0-front, CCW) to internal angle (0-right, CCW)
                # by adding 90 degrees. The result is wrapped to [0, 360).
                internal_angle_deg = (self.spoofer_angle_deg + 90) % 360
                target_azimuth = internal_angle_deg * 100
                target_altitude = self.spoofer_altitude_deg * 100
                target_point = (target_azimuth, target_altitude)
                #print(f"Target spoofer trigger point at az={target_azimuth/100} deg, alt={target_altitude/100} deg")

                available_points = list(current_lidar.depth_map.keys())
                
                if not available_points:
                    print("Warning: Cannot determine spoofer trigger point, depth map is empty.")
                elif target_point in current_lidar.depth_map:
                    actual_trigger_point = target_point
                    #print(f"Using exact spoofer trigger point at {self.spoofer_angle_deg} deg (front=0, ccw).")
                else:
                    # Find the closest point, prioritizing points on the same altitude ring
                    points_on_same_altitude = [p for p in available_points if p[1] == target_altitude]
                    
                    if points_on_same_altitude:
                        # If points are found on the target altitude, find the one with the closest azimuth
                        azimuth_distances = [abs(az - target_azimuth) for az, alt in points_on_same_altitude]
                        closest_index_on_alt = np.argmin(azimuth_distances)
                        actual_trigger_point = points_on_same_altitude[closest_index_on_alt]
                        #print(f"Target spoofer point at az={target_azimuth/100}, alt={target_altitude/100} deg not found. Using closest point on same altitude ring: az={actual_trigger_point[0]/100}, alt={actual_trigger_point[1]/100} deg")
                    else:
                        # Fallback: find the closest point to the target azimuth on the HIGHEST altitude ring.
                        new_target_altitude = self.sorted_vertical_angles[0] * 100
                        distances = [np.sqrt((az - target_azimuth)**2 + (alt - new_target_altitude)**2) for az, alt in available_points]
                        closest_index = np.argmin(distances)
                        actual_trigger_point = available_points[closest_index]
                        #print(f"Target spoofer point at az={target_azimuth/100}, alt={target_altitude/100} deg not found. No points on target altitude ring. Using closest point to highest altitude ring: az={actual_trigger_point[0]/100}, alt={actual_trigger_point[1]/100} deg")

            # Create matrices for noattack and attacked versions
            noattack_frame_data = np.zeros((self.channels, self.horizontal_resolution, self.samples_per_scan), dtype=np.float32)
            attack_frame_data = np.zeros_like(noattack_frame_data)
            noattack_frame_labels = np.zeros((self.channels, self.horizontal_resolution, self.samples_per_scan), dtype=np.uint8)
            attack_frame_labels = np.zeros_like(noattack_frame_labels)
            
            # Shared matrices
            answer_matrix = np.zeros((self.channels, self.horizontal_resolution), dtype=np.float32)
            azimuth_angles = np.full((self.channels, self.horizontal_resolution), np.nan, dtype=np.float32)
            timestamp_matrix = np.zeros((self.channels, self.horizontal_resolution), dtype=np.int64)
            
            try:
                for scan_idx in range(current_lidar.max_index):
                    config, signal = current_lidar.scan()
                    
                    true_peak_time, _ = get_peak_time_and_amplitude(signal)
                    
                    # labeling: 0 = no return, 1 = legitimate return, 2 = HFR return
                    LEGITIMATE_PULSE = 1
                    HFR_PULSE = 2

                    # --- Base (No-Attack) Data ---
                    noattack_signal = signal.copy()
                    current_noattack_labels = np.zeros_like(signal, dtype=np.uint8)
                    current_noattack_labels[noattack_signal > 0.01] = LEGITIMATE_PULSE

                    # --- Attacked Data ---
                    attack_signal = signal # This will be modified
                    current_attack_labels = current_noattack_labels.copy()

                    if self.spoofer_type != "off":
                        # Proximity-based trigger logic
                        if actual_trigger_point is not None and self.spoofer.trigger_time is None:
                            az_key_ideal, alt_key_ideal = config.azimuth, config.altitude
                            az_key_target, alt_key_target = actual_trigger_point
                            azimuth_tolerance, altitude_tolerance = 800, 400
                            azimuth_diff = min(abs(az_key_ideal - az_key_target), 36000 - abs(az_key_ideal - az_key_target))
                            altitude_diff = abs(alt_key_ideal - alt_key_target)

                            if (current_lidar.scan_mode == 'vertical' and altitude_diff <= altitude_tolerance) or \
                               (current_lidar.scan_mode != 'vertical' and azimuth_diff <= azimuth_tolerance and altitude_diff <= altitude_tolerance):
                                self.spoofer.trigger(config, signal)
                        
                        external_signal = np.zeros_like(attack_signal)
                        is_in_attack_angle = (spoofer_attack_start_az <= config.azimuth <= spoofer_attack_end_az)
                        if spoofer_attack_start_az < 0:
                            is_in_attack_angle = (config.azimuth >= (36000 + spoofer_attack_start_az) or config.azimuth <= spoofer_attack_end_az)
                        
                        if self.spoofer.trigger_time is not None and is_in_attack_angle:
                            external_signal = apply_noise(self.spoofer.get_range_signal(config.start_timestamp, config.accept_duration), ratio=0.01)

                        current_attack_labels = np.where(external_signal > attack_signal, HFR_PULSE, current_attack_labels)
                        attack_signal = np.maximum(attack_signal, external_signal)

                    attack_signal = np.clip(attack_signal, 0, 9);

                    # Use horizontal_index if available (channel-based architecture, collision-free)
                    if hasattr(config, 'horizontal_index') and config.horizontal_index is not None:
                        horizontal_index = config.horizontal_index
                    else:
                        # Fallback to old calculation for backward compatibility
                        azimuth_deg = config.azimuth / 100.0
                        normalized_azimuth = (azimuth_deg - current_lidar.initial_azimuth_offset + 360) % 360
                        horizontal_index = int(normalized_azimuth / 360.0 * self.horizontal_resolution)

                    vertical_index = self.altitude_to_sorted_v_idx_map.get(config.altitude)

                    if horizontal_index < self.horizontal_resolution and vertical_index is not None:
                        # Populate matrices for both versions
                        noattack_frame_data[vertical_index, horizontal_index, :] = noattack_signal
                        noattack_frame_labels[vertical_index, horizontal_index, :] = current_noattack_labels
                        attack_frame_data[vertical_index, horizontal_index, :] = attack_signal
                        attack_frame_labels[vertical_index, horizontal_index, :] = current_attack_labels
                        
                        # Populate shared matrices
                        answer_matrix[vertical_index, horizontal_index] = true_peak_time
                        timestamp_matrix[vertical_index, horizontal_index] = config.start_timestamp.in_nanoseconds
                        if hasattr(config, 'azimuth_deg') and config.azimuth_deg is not None:
                            azimuth_angles[vertical_index, horizontal_index] = config.azimuth_deg

            except StopIteration:
                pass

            if save_to_file:
                config_data = {
                    'original_bin_path': sample_info['path'],
                    'initial_azimuth_offset': float(initial_azimuth_offset),
                    'vertical_angles': [float(angle) for angle in self.sorted_vertical_angles],
                    'fov': 360.0,
                    'time_resolution_ns': float(self.time_resolution_ns)
                }

                # Save "noattack" version
                self._save_frame_data(
                    sample_token=sample_token, status='noattack',
                    signal_matrix=noattack_frame_data, labels_matrix=noattack_frame_labels,
                    answer_matrix=answer_matrix, azimuth_angles=azimuth_angles,
                    timestamp_matrix=timestamp_matrix, config_data=config_data
                )

                # Save "attacked" version
                self._save_frame_data(
                    sample_token=sample_token, status='attacked',
                    signal_matrix=attack_frame_data, labels_matrix=attack_frame_labels,
                    answer_matrix=answer_matrix, azimuth_angles=azimuth_angles,
                    timestamp_matrix=timestamp_matrix, config_data=config_data
                )
        
        print(f"Finished processing {num_frames} frames. Output saved in {self.output_dir}")

if __name__ == '__main__':
    import time

    parser = argparse.ArgumentParser(description="Generate LiDAR signal datasets.")
    parser.add_argument("--lidar-type", type=str, default="PCD_VLP32c", choices=["VLP16", "PCD_VLP16", "PCD_VLP32c", "PCD_HDL64E"],
                        help="Type of LiDAR to use.")
    parser.add_argument("--pcd-directory", type=str, default=None,
                        help="Path to the directory containing PCD files, required if lidar-type starts with PCD.")
    parser.add_argument("--json-path", type=str, default=None,
                        help="Path to the JSON file containing paths to .bin files.")
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
    parser.add_argument("--spoofer-altitude", type=float, default=40.0,
                        help="The altitude for the spoofer trigger, in degrees.")
    parser.add_argument("--spoofer-width-deg", type=float, default=90.0,
                        help="The angular width of the spoofer's attack cone in degrees.")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="Starting frame index (0-indexed) for processing PCD files.")
    parser.add_argument("--initial-point-offset", type=int, default=0,
                        help="Initial point offset to rotate the PCD point cloud.")
    parser.add_argument("--sync-angle", type=float, nargs='+', default=[1.0],
                        help="Sync value for timestamp calculation. In horizontal mode: angle in degrees. In vertical mode: number of channels. Provide one value for fixed, or two values (min max) for random selection per frame. Default is 1.0.")
    parser.add_argument("--scan-mode", type=str, default='vertical', choices=['horizontal', 'vertical'],
                        help="LiDAR scan mode: 'horizontal' (angle-based sync) or 'vertical' (channel-based sync). Default is 'vertical'.")
    parser.add_argument("--horizontal-resolution-deg", type=float, default=0.1,
                        help="Internal horizontal resolution in degrees for PCD-based LiDARs. Default is 0.1 degrees.")
    parser.add_argument("--output-horizontal-resolution-deg", type=float, default=None,
                        help="Output horizontal resolution in degrees (determines samples per channel). Auto-detected if not specified: HDL-64E=0.0818° (4400 samples), VLP32c/VLP16=0.2° (1800 samples).")
    parser.add_argument("--output-channels", type=int, default=None, choices=[16, 32, 64],
                        help="Number of output channels. For HDL-64E: 32 or 64 (default 64). For VLP32c: 32. For VLP16: 16. Auto-detected if not specified.")
    parser.add_argument("--bin-format", type=str, default='nuscenes', choices=['nuscenes', 'kitti'],
                        help="Output format for .bin files: 'nuscenes' (x,y,z,intensity,ring) or 'kitti' (x,y,z,intensity). Default: nuscenes")

    args = parser.parse_args()

    if (args.lidar_type.startswith("PCD")) and not args.pcd_directory and not args.json_path:
        parser.error(f"--pcd-directory or --json-path is required when --lidar-type is {args.lidar_type}")

    # Parse sync_angle argument
    if len(args.sync_angle) == 1:
        sync_angle_range = (args.sync_angle[0], args.sync_angle[0])  # Fixed angle
    elif len(args.sync_angle) == 2:
        sync_angle_range = (args.sync_angle[0], args.sync_angle[1])  # Range
    else:
        parser.error("--sync-angle must be either one value or two values (min max)")

    generator = LidarSignalDatasetGenerator(
        lidar_type=args.lidar_type,
        pcd_directory=args.pcd_directory,
        json_path=args.json_path,
        output_dir=args.output_dir,
        time_resolution_ns=args.time_resolution_ns,
        spoofer_type=args.spoofer_type,
        spoofer_angle_deg=args.spoofer_angle,
        spoofer_altitude_deg=args.spoofer_altitude,
        spoofer_width_deg=args.spoofer_width_deg,
        initial_point_offset=args.initial_point_offset,
        sync_angle_range=sync_angle_range,
        horizontal_resolution_deg=args.horizontal_resolution_deg,
        output_horizontal_resolution_deg=args.output_horizontal_resolution_deg,
        scan_mode=args.scan_mode,
        output_channels=args.output_channels,
        bin_format=args.bin_format
    )
    
    print("\nStarting dataset generation...")
    start_time = time.perf_counter()

    generator.generate(
        num_frames=args.num_frames,
        start_frame=args.start_frame,
        save_to_file=True
    )

    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"\nDataset generation finished in {duration:.2f} seconds.")
