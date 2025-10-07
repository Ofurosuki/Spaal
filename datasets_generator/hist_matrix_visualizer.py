
import numpy as np
import open3d as o3d
import argparse
import os
import glob
from typing import Tuple

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
    def __init__(self, npz_file_path: str = None, pcd_directory_path: str = None, data: dict = None, amplitude_to_intensity_ratio: float = 255.0/10.0):
        self.npz_file_path = npz_file_path
        self.pcd_directory_path = pcd_directory_path
        self.is_prediction = False
        self.amplitude_to_intensity_ratio = amplitude_to_intensity_ratio
        print(f"amplitude_to_intensity_ratio: {self.amplitude_to_intensity_ratio}")

        if data is None and npz_file_path:
            print(f"Loading data from {npz_file_path}")
            with np.load(npz_file_path) as loaded_data:
                data = {key: loaded_data[key] for key in loaded_data}
        elif data is None:
            raise ValueError("Either 'npz_file_path' or 'data' dictionary must be provided.")

        print(f"shape of signals: {data['signals'].shape if 'signals' in data else 'N/A'}")
        if 'signals' not in data and 'prediction' in data:
            self.is_prediction = True
            self.hist_matrix = data['prediction']
        else:
            self.hist_matrix = data['signals']
        print(f"Loaded hist_matrix with shape: {self.hist_matrix.shape}")

        if 'initial_azimuth_offsets' in data:
            self.initial_azimuth_offsets = data['initial_azimuth_offsets']
        else:
            print("Warning: 'initial_azimuth_offsets' not found. Defaulting to 0.0 for all frames.")
            self.initial_azimuth_offsets = [data.get('initial_azimuth_offset', 0.0)] * len(self.hist_matrix)
        
        v_angles_default = sorted([-30.67, -9.33, -29.33, -8.0, -28.0, -6.66, -26.66, -5.33, -25.33, -4.0, -24.0, -2.67, -22.67, -1.33, -21.33, 0.0, -20.0, 1.33, -18.67, 2.67, -17.33, 4.0, -16.0, 5.33, -14.67, 6.67, -13.33, 8.0, -12.0, 9.33, -10.67, 10.67], reverse=True)
        self.vertical_angles = data.get('vertical_angles', v_angles_default)
        self.fov = data.get('fov', 360.0)
        self.time_resolution_ns = data.get('time_resolution_ns', 1.0)

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

        if self.pcd_files and frame_index < len(self.pcd_files):
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
    parser = argparse.ArgumentParser(description="Visualize or save LiDAR histogram matrix from .npz file.")
    parser.add_argument("--npz-file", required=True, type=str, help="Path to the .npz histogram matrix file.")
    parser.add_argument("--pcd-directory", type=str, default=None, help="Path to the directory with original .pcd files for comparison or for output naming.")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to visualize.")
    parser.add_argument("--output-pcd-dir", type=str, default=None, help="Path to the directory to save reconstructed .pcd files. If provided, visualization is skipped and all frames are processed.")
    parser.add_argument("--amplitude-to-intensity-ratio", type=float, default=1.0, help="Ratio to convert signal amplitude to intensity for reconstructed PCD.")

    args = parser.parse_args()

    visualizer = HistMatrixVisualizer(npz_file_path=args.npz_file, pcd_directory_path=args.pcd_directory, amplitude_to_intensity_ratio=args.amplitude_to_intensity_ratio)
    if args.output_pcd_dir:
        visualizer.save_reconstructed_pcds(args.output_pcd_dir)
    else:
        visualizer.visualize(args.frame)
