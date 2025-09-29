import numpy as np
import os
import argparse
import glob
import h5py
# import hdf5plugin  # Ensure hdf5plugin is imported to enable compression filters
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
                 lidar_type: str = "PCD_VLP32c",
                 pcd_directory: str = None,
                 output_dir: str = "./datasets",
                 outdoor_distance: float = 50.0, outdoor_ratio: float = 0.8,
                 spoofer_type: str = "adaptive_hfr_perturbation",
                 spoofer_frequency: float = 10 * 1e6,
                 spoofer_duration_ms: float = 5,
                 spoofer_distance_m: float = 10.0,
                 spoofer_pulse_width_ns: float = 5,
                 spoofer_perturbation_ns: float = 20.0,
                 spoofer_amplitude_range: tuple[float, float] = (3.0, 8.0),
                 lidar_amplitude_range: tuple[float, float] = (3.0, 3.0),
                 lidar_pulse_width_ns: float = 5,
                 time_resolution_ns: float = 1.0,
                 noise_ratio: float = 0.1,
                 sunlight_mean: float = 0.5,
                 spoofer_angles_deg: list[float] = [0.0], 
                 spoofer_altitude_deg: float = 8.0,
                 spoofer_width_deg: float = 90.0):

        self.lidar_type = lidar_type
        self.pcd_directory = pcd_directory
        self.time_resolution_ns = time_resolution_ns
        
        # Ensure spoofer_angles_deg is a list for frame-specific angles
        if not isinstance(spoofer_angles_deg, list):
            spoofer_angles_deg = [spoofer_angles_deg]
        self.spoofer_angles_deg = spoofer_angles_deg

        self.spoofer_altitude_deg = spoofer_altitude_deg
        self.spoofer_width_deg = spoofer_width_deg

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
            if self.lidar_type == "PCD_VLP32c":
                print("Applying azimuth time perturbation for VLP32c model.")
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

    def generate(self, num_frames: int, filename_prefix: str = "lidar_signal"):
        output_filename = os.path.join(self.output_dir, f"{filename_prefix}.h5")
        all_initial_azimuth_offsets = []

        chunk_size = 100
        signals_chunk = []
        labels_chunk = []
        answers_chunk = []

        with h5py.File(output_filename, 'w') as f:
            # Create resizable datasets on disk with compression
            signals_dset = f.create_dataset('signals', 
                                            (num_frames, self.channels, self.horizontal_resolution, self.samples_per_scan), 
                                            maxshape=(None, self.channels, self.horizontal_resolution, self.samples_per_scan),
                                            dtype='f4', compression='gzip')
            labels_dset = f.create_dataset('labels', 
                                           (num_frames, self.channels, self.horizontal_resolution, self.samples_per_scan), 
                                           maxshape=(None, self.channels, self.horizontal_resolution, self.samples_per_scan),
                                           dtype='u1', compression='gzip')
            answer_dset = f.create_dataset('answer_matrix', 
                                           (num_frames, self.channels, self.horizontal_resolution), 
                                           maxshape=(None, self.channels, self.horizontal_resolution),
                                           dtype='i4', compression='gzip')

            print(f"Generating {num_frames} frames and saving to {output_filename} in chunks of {chunk_size}...")

            for frame_num in range(num_frames):
                print(f"\nGenerating frame {frame_num + 1}/{num_frames}...")
                
                if self.lidar_type in ["PCD_VLP16", "PCD_VLP32c"]:
                    pcd_file_path = self.pcd_files[frame_num % len(self.pcd_files)] # Use modulo for safety
                    print(f"  - Using PCD file: {os.path.basename(pcd_file_path)}")
                    current_lidar = self.lidar.new_frame(frame_num=frame_num, base_timestamp=PreciseDuration(nanoseconds=frame_num * 10**9))
                else:
                    current_lidar = self.lidar.new_frame(base_timestamp=PreciseDuration(nanoseconds=frame_num * 10**9))

                if hasattr(current_lidar, 'initial_azimuth_offset'):
                    all_initial_azimuth_offsets.append(current_lidar.initial_azimuth_offset)
                else:
                    all_initial_azimuth_offsets.append(0.0)

                current_spoofer_angle = self.spoofer_angles_deg[frame_num % len(self.spoofer_angles_deg)]
                print(f"  - Using spoofer angle: {current_spoofer_angle} deg")
                internal_angle_deg = (current_spoofer_angle + 90) % 360
                spoofer_attack_center_az = internal_angle_deg * 100
                spoofer_attack_width_az = self.spoofer_width_deg * 100
                spoofer_attack_start_az = spoofer_attack_center_az - spoofer_attack_width_az / 2
                spoofer_attack_end_az = spoofer_attack_center_az + spoofer_attack_width_az / 2

                actual_trigger_point = None
                if self.spoofer_type != "off" and hasattr(current_lidar, 'depth_map'):
                    target_azimuth = internal_angle_deg * 100
                    target_altitude = self.spoofer_altitude_deg * 100
                    target_point = (target_azimuth, target_altitude)
                    available_points = list(current_lidar.depth_map.keys())
                    if not available_points:
                        print("Warning: Cannot determine spoofer trigger point, depth map is empty.")
                    elif target_point in current_lidar.depth_map:
                        actual_trigger_point = target_point
                    else:
                        distances = [np.sqrt((az - target_azimuth)**2 + (alt - target_altitude)**2) for az, alt in available_points]
                        closest_index = np.argmin(distances)
                        actual_trigger_point = available_points[closest_index]
                        print(f"Target spoofer point at {current_spoofer_angle} deg (front=0, ccw) not found. Using closest point: az={actual_trigger_point[0]/100}, alt={actual_trigger_point[1]/100} deg")
                
                frame_data = np.zeros((self.channels, self.horizontal_resolution, self.samples_per_scan), dtype=np.float32)
                frame_labels = np.zeros((self.channels, self.horizontal_resolution, self.samples_per_scan), dtype=np.uint8)
                frame_answer = np.zeros((self.channels, self.horizontal_resolution), dtype=np.int32)

                try:
                    for i in range(current_lidar.max_index):
                        config, signal = current_lidar.scan()
                        raises = np.flatnonzero((signal[:-1] < 0.01) & (signal[1:] >= 0.01)) + 1
                        true_peak_index = 0
                        if len(raises) > 0:
                            peaks = np.array([np.max(signal[r:min(len(signal), r + 50)]) for r in raises])
                            if len(peaks) > 0:
                                true_peak_index = raises[np.argmax(peaks)]

                        lidar_amp = np.random.uniform(self.lidar_amplitude_range[0], self.lidar_amplitude_range[1])
                        current_lidar.set_amplitude(lidar_amp)
                        
                        LEGITIMATE_PULSE, HFR_PULSE = 1, 2
                        current_labels = np.zeros_like(signal, dtype=np.uint8)
                        current_labels[signal > 0.01] = LEGITIMATE_PULSE

                        if self.spoofer_type != "off":
                            if actual_trigger_point is not None and self.spoofer.trigger_time is None:
                                az_key_ideal, alt_key_ideal = config.azimuth, config.altitude
                                az_key_target, alt_key_target = actual_trigger_point[0], actual_trigger_point[1]
                                azimuth_tolerance, altitude_tolerance = 10, 100
                                azimuth_diff = min(abs(az_key_ideal - az_key_target), 36000 - abs(az_key_ideal - az_key_target))
                                if azimuth_diff <= azimuth_tolerance and abs(alt_key_ideal - alt_key_target) <= altitude_tolerance:
                                    self.spoofer.trigger(config, signal)
                            
                            external_signal = np.zeros_like(signal)
                            is_in_attack_angle = (spoofer_attack_start_az <= config.azimuth <= spoofer_attack_end_az)
                            if self.spoofer.trigger_time is not None and is_in_attack_angle:
                                external_signal = apply_noise(self.spoofer.get_range_signal(config.start_timestamp, config.accept_duration), ratio=0.01)

                            current_labels = np.where(external_signal > signal, HFR_PULSE, current_labels)
                            signal = np.maximum(signal, external_signal)

                        signal = np.clip(signal, 0, 9)
                        azimuth_deg = config.azimuth / 100.0
                        normalized_azimuth = (azimuth_deg - current_lidar.initial_azimuth_offset + 360) % 360
                        horizontal_index = int(normalized_azimuth / 360.0 * self.horizontal_resolution)
                        vertical_index = self.altitude_to_sorted_v_idx_map.get(config.altitude)

                        if horizontal_index < self.horizontal_resolution and vertical_index is not None:
                            frame_data[vertical_index, horizontal_index, :] = signal
                            frame_labels[vertical_index, horizontal_index, :] = current_labels
                            frame_answer[vertical_index, horizontal_index] = true_peak_index
                except StopIteration:
                    pass

                signals_chunk.append(frame_data)
                labels_chunk.append(frame_labels)
                answers_chunk.append(frame_answer)

                is_last_frame = (frame_num + 1) == num_frames
                if (len(signals_chunk) == chunk_size) or (is_last_frame and len(signals_chunk) > 0):
                    start_idx = frame_num - (len(signals_chunk) - 1)
                    end_idx = frame_num + 1
                    print(f"  - Writing chunk of {len(signals_chunk)} frames to disk (frames {start_idx}-{frame_num})...")
                    signals_dset[start_idx:end_idx] = np.array(signals_chunk, dtype=np.float32)
                    labels_dset[start_idx:end_idx] = np.array(labels_chunk, dtype=np.uint8)
                    answer_dset[start_idx:end_idx] = np.array(answers_chunk, dtype=np.int32)
                    signals_chunk, labels_chunk, answers_chunk = [], [], []

            f.create_dataset('initial_azimuth_offsets', data=np.array(all_initial_azimuth_offsets))
            f.create_dataset('vertical_angles', data=self.sorted_vertical_angles)
            f.create_dataset('fov', data=360.0)
            f.create_dataset('time_resolution_ns', data=self.time_resolution_ns)
            
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
    parser.add_argument("--spoofer-angle", type=float, nargs='+', default=[0.0],
                        help="The angle(s) for the spoofer trigger, in degrees, counter-clockwise with 0 at the front. Can provide multiple values for frame-specific angles.")
    parser.add_argument("--spoofer-altitude", type=float, default=8.0,
                        help="The altitude for the spoofer trigger, in degrees.")
    parser.add_argument("--spoofer-width-deg", type=float, default=90.0,
                        help="The angular width of the spoofer's attack cone in degrees.")

    args = parser.parse_args()

    if (args.lidar_type.startswith("PCD")) and not args.pcd_directory:
        parser.error(f"--pcd-directory is required when --lidar-type is {args.lidar_type}")

    generator = LidarSignalDatasetGenerator(
        lidar_type=args.lidar_type,
        pcd_directory=args.pcd_directory,
        output_dir=args.output_dir,
        time_resolution_ns=args.time_resolution_ns,
        spoofer_type=args.spoofer_type,
        spoofer_angles_deg=args.spoofer_angle,
        spoofer_altitude_deg=args.spoofer_altitude,
        spoofer_width_deg=args.spoofer_width_deg
    )
    generator.generate(num_frames=args.num_frames)
