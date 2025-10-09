import numpy as np
import os
import argparse
import json
from typing import Tuple
import blosc2
import pickle
from tqdm import tqdm

from spaal2.core import PreciseDuration
from spaal2.core.dummy_lidar.dummy_lidar_vlp32_pcd import PcdLidarVLP32c

# This function is kept from the original script for signal processing
def get_peak_time_and_amplitude(signal: np.ndarray) -> Tuple[float, float]:
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

class HistMatrixGenerator:
    def __init__(self, 
                 json_path: str,
                 output_dir: str = "./nuscenes_hist_matrix_blosc",
                 lidar_amplitude_range: tuple[float, float] = (3.0, 3.0),
                 lidar_pulse_width_ns: float = 5,
                 time_resolution_ns: float = 1.0,
                 initial_point_offset: int = 0,
                 sync_angle_step_deg: float = 10):

        self.json_path = json_path
        self.output_dir = output_dir
        self.time_resolution_ns = time_resolution_ns
        self.initial_point_offset = initial_point_offset
        self.sync_angle_step_deg = sync_angle_step_deg

        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"Input JSON file not found at {self.json_path}")
            
        with open(self.json_path, 'r') as f:
            self.lidar_samples = json.load(f)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")

        # Initialize LiDAR model
        self.lidar = PcdLidarVLP32c(
            pcd_file_path=None, # We will load points directly
            lidar_position=np.array([0.0, 0.0, 0.0]),
            lidar_rotation=np.array([0.0, 0.0, 0.0]),
            amplitude=lidar_amplitude_range[0],
            pulse_width=PreciseDuration(nanoseconds=lidar_pulse_width_ns),
            time_resolution_ns=self.time_resolution_ns,
            initial_point_offset=self.initial_point_offset
        )
        self.lidar.set_sync_angle_step(self.sync_angle_step_deg)
        
        self.channels = 32
        # Manually set horizontal resolution for VLP-32c
        self.horizontal_resolution = 1800 
        self.samples_per_scan = int(self.lidar.accept_window.in_nanoseconds / self.lidar.time_resolution_ns)
        
        self.sorted_vertical_angles = sorted(self.lidar.vertical_angles, reverse=True)
        self.altitude_to_sorted_v_idx_map = {int(angle * 100): i for i, angle in enumerate(self.sorted_vertical_angles)}

    def process_and_save_frame(self, sample_info: dict, frame_idx: int):
        token = sample_info['token']
        bin_path = sample_info['path']
        
        output_filename = os.path.join(self.output_dir, f"{token}.bl2")
        if os.path.exists(output_filename):
            # print(f"File {output_filename} already exists. Skipping.")
            return

        # Load and reshape the point cloud data
        # The format is [x, y, z, intensity, ring]
        raw_data = np.fromfile(bin_path, dtype=np.float32)
        # Ensure the data is a multiple of 5
        if raw_data.size % 5 != 0:
            print(f"Warning: Skipping {bin_path} due to unexpected data size.")
            return
        
        point_cloud = raw_data.reshape(-1, 5)
        points = point_cloud[:, :3]
        intensities = point_cloud[:, 3]
        
        # Use the new method to load points into the lidar model
        current_lidar = self.lidar.new_frame_from_points(
            points=points,
            intensities=intensities,
            base_timestamp=PreciseDuration(nanoseconds=frame_idx * 10**9)
        )

        initial_azimuth_offset = current_lidar.initial_azimuth_offset

        frame_data = np.zeros((self.channels, self.horizontal_resolution, self.samples_per_scan), dtype=np.float32)
        answer_matrix = np.zeros((self.channels, self.horizontal_resolution), dtype=np.float32)

        try:
            for scan_idx in range(current_lidar.max_index):
                config, signal = current_lidar.scan() 
                
                true_peak_time, _ = get_peak_time_and_amplitude(signal)
                
                signal = np.clip(signal, 0, 9)

                azimuth_deg = config.azimuth / 100.0
                normalized_azimuth = (azimuth_deg - initial_azimuth_offset + 360) % 360
                horizontal_index = int(normalized_azimuth / 360.0 * self.horizontal_resolution)
                vertical_index = self.altitude_to_sorted_v_idx_map.get(config.altitude)

                if horizontal_index < self.horizontal_resolution and vertical_index is not None:
                    frame_data[vertical_index, horizontal_index, :] = signal
                    answer_matrix[vertical_index, horizontal_index] = true_peak_time

        except StopIteration:
            pass

        # Prepare data payload for blosc2
        data_payload = {
            'signals': frame_data,
            'answer_matrix': answer_matrix,
            'initial_azimuth_offset': np.array([initial_azimuth_offset]),
            'vertical_angles': np.array(self.sorted_vertical_angles),
            'fov': np.array([360.0]),
            'time_resolution_ns': np.array([self.time_resolution_ns])
        }
        
        # Serialize the dictionary with pickle and compress with blosc2
        pickled_data = pickle.dumps(data_payload)
        compressed_data = blosc2.pack(pickled_data)
        
        with open(output_filename, 'wb') as f:
            f.write(compressed_data)

    def generate(self, num_frames: int = -1, start_frame: int = 0):
        total_samples = len(self.lidar_samples)
        if start_frame >= total_samples:
            print(f"Start frame {start_frame} is out of bounds. No files to process.")
            return

        end_frame = total_samples
        if num_frames != -1:
            end_frame = min(start_frame + num_frames, total_samples)
        
        process_range = range(start_frame, end_frame)
        
        print(f"Processing {len(process_range)} frames from index {start_frame} to {end_frame - 1}...")

        for i in tqdm(process_range, desc="Generating Hist-Matrix Files"):
            sample_info = self.lidar_samples[i]
            self.process_and_save_frame(sample_info, frame_idx=i)
            
        print(f"\nFinished processing. Hist-matrix files are saved in {self.output_dir}")


if __name__ == '__main__':
    import time

    parser = argparse.ArgumentParser(description="Generate LiDAR hist-matrix datasets from nuScenes .bin files.")
    parser.add_argument("--json-path", type=str, required=True,
                        help="Path to the JSON file containing nuScenes sample info (e.g., nuscenes_info.json).")
    parser.add_argument("--num-frames", type=int, default=-1,
                        help="Number of frames to generate. -1 for all frames in the JSON.")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="Starting frame index (0-indexed) from the JSON file.")
    parser.add_argument("--output-dir", type=str, default="./nuscenes_hist_matrix_blosc",
                        help="Directory to save the generated .bl2 files.")
    parser.add_argument("--time-resolution-ns", type=float, default=1.0,
                        help="Time resolution in nanoseconds for the simulation.")
    parser.add_argument("--initial-point-offset", type=int, default=0,
                        help="Initial point offset to rotate the point cloud.")
    
    args = parser.parse_args()

    generator = HistMatrixGenerator(
        json_path=args.json_path,
        output_dir=args.output_dir,
        time_resolution_ns=args.time_resolution_ns,
        initial_point_offset=args.initial_point_offset
    )
    
    print("\nStarting hist-matrix generation...")
    start_time = time.perf_counter()

    generator.generate(
        num_frames=args.num_frames,
        start_frame=args.start_frame
    )

    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"\nGeneration finished in {duration:.2f} seconds.")