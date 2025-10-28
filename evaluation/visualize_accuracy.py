import numpy as np
import os
import blosc2
import argparse
import json
from typing import Tuple
import open3d as o3d

def get_peak_time_and_amplitude(signal: np.ndarray) -> Tuple[float, float]:
    """
    Finds the interpolated time and amplitude of the highest peak in a signal.
    This function is copied from datasets_generator/hist_matrix_visualizer.py.
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

class AccuracyVisualizer:
    def __init__(self, directory: str, threshold: float, min_gt_distance: float = 0.0):
        self.directory = directory
        self.threshold = threshold
        self.min_gt_distance = min_gt_distance
        
        config = self._load_config()
        self.time_resolution_ns = config.get('time_resolution_ns', 1.0)
        self.vertical_angles = config.get('vertical_angles')
        self.fov = config.get('fov', 360.0)

        if not self.vertical_angles:
            raise ValueError("Could not load 'vertical_angles' from config.json")

        gt_path_option1 = os.path.join(directory, 'gt.bl2')
        gt_path_option2 = os.path.join(directory, 'answer_matrix.bl2')
        
        if os.path.exists(gt_path_option1):
            self.gt_path = gt_path_option1
        elif os.path.exists(gt_path_option2):
            self.gt_path = gt_path_option2
        else:
            raise FileNotFoundError(f"Could not find 'gt.bl2' or 'answer_matrix.bl2' in {directory}")

        self.signal_path = os.path.join(directory, 'signal.bl2')

        print(f"Loading GT from: {self.gt_path}")
        print(f"Loading signal from: {self.signal_path}")

        self.gt_matrix = self._load_bl2(self.gt_path)
        self.signal_matrix = self._load_bl2(self.signal_path)

    def _load_config(self) -> dict:
        config_path = os.path.join(self.directory, 'config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found in {self.directory}")
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            print(f"Loaded config from {config_path}")
            return config_data

    def _load_bl2(self, file_path: str) -> np.ndarray:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found.")
        with open(file_path, 'rb') as f:
            data = blosc2.unpack_array(f.read())
            if data.ndim in [2, 3]:
                return data
            else:
                raise ValueError(f"Unsupported matrix dimension: {data.ndim}")

    def _calculate_distances(self, hist_matrix: np.ndarray) -> np.ndarray:
        if hist_matrix.ndim == 2:
            return hist_matrix * self.time_resolution_ns * 0.15

        num_channels, num_horizontal_steps, _ = hist_matrix.shape
        distances = np.zeros((num_channels, num_horizontal_steps))
        
        for c in range(num_channels):
            for h in range(num_horizontal_steps):
                signal = hist_matrix[c, h, :]
                interpolated_time, _ = get_peak_time_and_amplitude(signal)
                distances[c, h] = interpolated_time * self.time_resolution_ns * 0.15
        return distances

    def visualize(self):
        print("Calculating distances...")
        gt_distances = self._calculate_distances(self.gt_matrix)
        signal_distances = self._calculate_distances(self.signal_matrix)

        if gt_distances.shape != signal_distances.shape:
            raise ValueError(f"Shape mismatch between GT distances {gt_distances.shape} and signal distances {signal_distances.shape}")

        abs_error = np.abs(gt_distances - signal_distances)

        correct_points = []
        incorrect_points = []
        incorrect_gt_points = []

        num_channels, num_horizontal_steps = gt_distances.shape

        print("Reconstructing and classifying points...")
        for v_idx in range(num_channels):
            for h_idx in range(num_horizontal_steps):
                gt_dist = gt_distances[v_idx, h_idx]

                if gt_dist <= self.min_gt_distance:
                    continue

                signal_dist = signal_distances[v_idx, h_idx]
                if signal_dist <= 0:
                    continue

                altitude_deg = self.vertical_angles[v_idx]
                azimuth_deg = (h_idx / num_horizontal_steps) * self.fov

                alpha = np.deg2rad(azimuth_deg)
                omega = np.deg2rad(altitude_deg)

                # Calculate signal point
                x_signal = signal_dist * np.cos(omega) * np.sin(alpha)
                y_signal = signal_dist * np.cos(omega) * np.cos(alpha)
                z_signal = signal_dist * np.sin(omega)
                signal_point = [x_signal, y_signal, z_signal]

                if abs_error[v_idx, h_idx] <= self.threshold:
                    correct_points.append(signal_point)
                else:
                    incorrect_points.append(signal_point)

                    # Also add GT point for incorrect cases (yellow)
                    x_gt = gt_dist * np.cos(omega) * np.sin(alpha)
                    y_gt = gt_dist * np.cos(omega) * np.cos(alpha)
                    z_gt = gt_dist * np.sin(omega)
                    gt_point = [x_gt, y_gt, z_gt]
                    incorrect_gt_points.append(gt_point)

        print(f"Found {len(correct_points)} correct points (red), {len(incorrect_points)} incorrect points (blue), and {len(incorrect_gt_points)} GT points for incorrect (yellow).")

        pcds_to_draw = []
        if correct_points:
            correct_pcd = o3d.geometry.PointCloud()
            correct_pcd.points = o3d.utility.Vector3dVector(np.array(correct_points))
            correct_pcd.paint_uniform_color([1, 0, 0])  # Red
            pcds_to_draw.append(correct_pcd)

        if incorrect_points:
            incorrect_pcd = o3d.geometry.PointCloud()
            incorrect_pcd.points = o3d.utility.Vector3dVector(np.array(incorrect_points))
            incorrect_pcd.paint_uniform_color([0, 0, 1])  # Blue
            pcds_to_draw.append(incorrect_pcd)

        if incorrect_gt_points:
            incorrect_gt_pcd = o3d.geometry.PointCloud()
            incorrect_gt_pcd.points = o3d.utility.Vector3dVector(np.array(incorrect_gt_points))
            incorrect_gt_pcd.paint_uniform_color([1, 1, 0])  # Yellow
            pcds_to_draw.append(incorrect_gt_pcd)

        if pcds_to_draw:
            dir_name = os.path.basename(os.path.normpath(self.directory))
            o3d.visualization.draw_geometries(pcds_to_draw, window_name=f"Accuracy Visualization: {dir_name}")
        else:
            print("No valid points found to visualize.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize reconstruction accuracy by coloring points.")
    parser.add_argument("--directory", required=True, help="Directory containing the .bl2 files and config.json.")
    parser.add_argument("--threshold", type=float, required=True, help="Error threshold to classify points as correct (red) or incorrect (blue).")
    parser.add_argument("--min-gt-distance", type=float, default=0.0, help="Minimum ground truth distance to include in evaluation (in meters). Defaults to 0.0.")
    args = parser.parse_args()

    try:
        visualizer = AccuracyVisualizer(args.directory, args.threshold, args.min_gt_distance)
        visualizer.visualize()
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
