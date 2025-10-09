
import numpy as np
import open3d as o3d
import argparse
import os
import glob
from typing import Tuple
import json
import blosc2

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

class HistMatrixVisualizer:
    def __init__(self, dataset_root_path: str, frame_index: int = 0, pcd_directory_path: str = None, data: dict = None, amplitude_to_intensity_ratio: float = 255.0/10.0, use_answer_matrix: bool = False):
        self.dataset_root_path = dataset_root_path
        self.frame_index = frame_index
        self.pcd_directory_path = pcd_directory_path
        self.is_prediction = use_answer_matrix
        self.amplitude_to_intensity_ratio = amplitude_to_intensity_ratio
        print(f"amplitude_to_intensity_ratio: {self.amplitude_to_intensity_ratio}")

        if data is None:
            sample_dirs = sorted([d for d in os.listdir(dataset_root_path) if os.path.isdir(os.path.join(dataset_root_path, d))])
            if not sample_dirs:
                raise FileNotFoundError(f"No sample directories found in {dataset_root_path}")
            if frame_index >= len(sample_dirs):
                raise ValueError(f"Frame index {frame_index} is out of bounds for {len(sample_dirs)} sample directories.")
            
            sample_dir = os.path.join(dataset_root_path, sample_dirs[frame_index])
            print(f"Loading data from {sample_dir}")

            with open(os.path.join(sample_dir, 'config.json'), 'r') as f:
                config_data = json.load(f)
            
            if use_answer_matrix:
                bl2_file = os.path.join(sample_dir, 'answer_matrix.bl2')
                self.is_prediction = True
            else:
                bl2_file = os.path.join(sample_dir, 'signal.bl2')
                self.is_prediction = False

            with open(bl2_file, 'rb') as f:
                packed_data = f.read()
            
            hist_matrix_single_frame = blosc2.unpack_array(packed_data)
            self.hist_matrix = np.expand_dims(hist_matrix_single_frame, axis=0)
            
            data = config_data

        print(f"shape of hist_matrix: {self.hist_matrix.shape}")

        self.initial_azimuth_offsets = [data.get('initial_azimuth_offset', 0.0)]
        
        v_angles_default = sorted([-30.67, -9.33, -29.33, -8.0, -28.0, -6.66, -26.66, -5.33, -25.33, -4.0, -24.0, -2.67, -22.67, -1.33, -21.33, 0.0, -20.0, 1.33, -18.67, 2.67, -17.33, 4.0, -16.0, 5.33, -14.67, 6.67, -13.33, 8.0, -12.0, 9.33, -10.67, 10.67], reverse=True)
        self.vertical_angles = data.get('vertical_angles', v_angles_default)
        self.fov = data.get('fov', 360.0)
        self.time_resolution_ns = data.get('time_resolution_ns', 1.0)
        self.original_bin_path = data.get('original_bin_path')

        self.pcd_files = []
        if self.pcd_directory_path:
            if not os.path.isdir(self.pcd_directory_path):
                raise ValueError(f"PCD directory path is not a valid directory: {self.pcd_directory_path}")
            self.pcd_files = sorted(glob.glob(os.path.join(self.pcd_directory_path, '*.pcd')))
            if not self.pcd_files:
                print(f"Warning: No PCD files found in {self.pcd_directory_path}")

    def _reconstruct_point_cloud(self, frame_index: int = 0):
        points = []
        intensities = []
        if frame_index >= len(self.hist_matrix):
            raise ValueError(f"Frame index {frame_index} is out of bounds for hist_matrix with {len(self.hist_matrix)} frames.")
        if frame_index < len(self.initial_azimuth_offsets):
            current_azimuth_offset = self.initial_azimuth_offsets[frame_index]
        else:
            current_azimuth_offset = self.initial_azimuth_offsets[-1] if self.initial_azimuth_offsets else 0.0
            print(f"Warning: Frame index {frame_index} is out of bounds for azimuth offsets. Using last available offset.")

        frame_data = self.hist_matrix[frame_index]
        
        is_prediction_local = len(frame_data.shape) == 2
        
        if is_prediction_local:
            channels, horizontal_resolution = frame_data.shape
        else:
            channels, horizontal_resolution, _ = frame_data.shape

        for v_idx in range(channels):
            for h_idx in range(horizontal_resolution):
                if not is_prediction_local:
                    signal = frame_data[v_idx, h_idx, :]
                    highest_peak_time, peak_amplitude = get_peak_time_and_amplitude(signal)

                    if highest_peak_time == 0.0:
                        continue
                    
                    intensity = np.clip(peak_amplitude * self.amplitude_to_intensity_ratio, 0, 255)
                else:
                    highest_peak_time = frame_data[v_idx, h_idx]
                    if highest_peak_time <= 0:
                        continue
                    intensity = 100 # Default intensity for predictions

                distance_m = (highest_peak_time * self.time_resolution_ns) * 0.15
                
                altitude_deg = self.vertical_angles[v_idx]
                azimuth_deg = (h_idx / horizontal_resolution) * self.fov + current_azimuth_offset

                alpha = np.deg2rad(azimuth_deg)
                omega = np.deg2rad(altitude_deg)
                
                x = distance_m * np.cos(omega) * np.sin(alpha)
                y = distance_m * np.cos(omega) * np.cos(alpha)
                z = distance_m * np.sin(omega)
                
                points.append([x, y, z])
                intensities.append(intensity)

        pcd = o3d.geometry.PointCloud()
        if points:
            pcd.points = o3d.utility.Vector3dVector(np.array(points))
            if intensities:
                #color_values = np.clip(np.array(intensities) / 255.0, 0, 1)
                colors = [[val, val, val] for val in intensities]
                pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
        return pcd

    def visualize(self, frame_index: int = 0):
        reconstructed_pcd = self._reconstruct_point_cloud(frame_index)
        reconstructed_pcd.paint_uniform_color([1, 0, 0])  # Red for reconstructed

        geometries = [reconstructed_pcd]

        if self.original_bin_path and os.path.exists(self.original_bin_path):
            print(f"Loading original BIN for comparison: {self.original_bin_path}")
            raw_data = np.fromfile(self.original_bin_path, dtype=np.float32)
            if raw_data.size % 5 != 0:
                raw_data = raw_data[:-(raw_data.size % 5)]
            
            points = raw_data.reshape(-1, 5)[:, :3]
            
            original_pcd = o3d.geometry.PointCloud()
            original_pcd.points = o3d.utility.Vector3dVector(points)
            original_pcd.paint_uniform_color([0, 0, 1])  # Blue for original
            geometries.append(original_pcd)
        elif self.pcd_files and frame_index < len(self.pcd_files):
            pcd_file_to_load = self.pcd_files[frame_index]
            print(f"Loading original PCD for comparison: {pcd_file_to_load}")
            original_pcd = o3d.io.read_point_cloud(pcd_file_to_load)
            original_pcd.paint_uniform_color([0, 0, 1])  # Blue for original
            geometries.append(original_pcd)
        elif self.pcd_directory_path:
             print(f"Warning: Frame index {frame_index} is out of bounds for the number of PCD files found ({len(self.pcd_files)}). Original PCD will not be displayed.")

        o3d.visualization.draw_geometries(geometries, window_name=f"Frame {frame_index}")

    def save_reconstructed_pcds(self, output_dir: str):
        if not self.pcd_directory_path:
            print("Error: --pcd-directory must be provided to name the output files.")
            return

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        num_frames = len(self.hist_matrix)
        for i in range(num_frames):
            print(f"Reconstructing frame {i + 1}/{num_frames}...")
            reconstructed_pcd = self._reconstruct_point_cloud(i)

            if i < len(self.pcd_files):
                base_name = os.path.basename(self.pcd_files[i])
                output_path = os.path.join(output_dir, base_name)
            else:
                print(f"Warning: Not enough original PCD files for naming. Using default name for frame {i}.")
                output_path = os.path.join(output_dir, f"reconstructed_frame_{i}.pcd")

            o3d.io.write_point_cloud(output_path, reconstructed_pcd)
            print(f"Saved reconstructed PCD to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize LiDAR data from the new dataset format.")
    parser.add_argument("--dataset-root-path", required=True, type=str, help="Path to the root directory of the dataset.")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to visualize.")
    parser.add_argument("--pcd-directory", type=str, default=None, help="Path to the directory with original .pcd files for comparison.")
    parser.add_argument("--output-pcd-dir", type=str, default=None, help="Path to the directory to save reconstructed .pcd files. If provided, visualization is skipped.")
    parser.add_argument("--amplitude-to-intensity-ratio", type=float, default=1.0, help="Ratio to convert signal amplitude to intensity for reconstructed PCD.")
    parser.add_argument("--use-answer-matrix", action='store_true', help="Use answer_matrix.bl2 instead of signal.bl2 for reconstruction.")

    args = parser.parse_args()

    visualizer = HistMatrixVisualizer(
        dataset_root_path=args.dataset_root_path,
        frame_index=args.frame,
        pcd_directory_path=args.pcd_directory,
        amplitude_to_intensity_ratio=args.amplitude_to_intensity_ratio,
        use_answer_matrix=args.use_answer_matrix
    )

    if args.output_pcd_dir:
        # The save_reconstructed_pcds method loops through all frames, which is not what we want here
        # as we only loaded a single frame. We can modify it or just save the single frame.
        # For now, let's just save the single reconstructed frame.
        pcd = visualizer._reconstruct_point_cloud(frame_index=0) # We always use frame_index 0 of the loaded data
        os.makedirs(args.output_pcd_dir, exist_ok=True)
        output_path = os.path.join(args.output_pcd_dir, f"reconstructed_frame_{args.frame}.pcd")
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"Saved reconstructed PCD to {output_path}")
    else:
        visualizer.visualize(0) # We always use frame_index 0 of the loaded data
