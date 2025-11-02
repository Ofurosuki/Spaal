import numpy as np
import os
import blosc2
import argparse
import json
from typing import Tuple
import matplotlib.pyplot as plt

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

class Bl2Evaluator:
    def __init__(self, directory: str, threshold: float, min_gt_distance: float = 0.0, plot: bool = False):
        self.directory = directory
        self.threshold = threshold
        self.min_gt_distance = min_gt_distance
        self.plot = plot
        self.time_resolution_ns = self._load_config()

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

    def _load_config(self) -> float:
        config_path = os.path.join(self.directory, 'config.json')
        if not os.path.exists(config_path):
            print("Warning: config.json not found. Using default time_resolution_ns=1.0")
            return 1.0
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            time_res = config_data.get('time_resolution_ns', 1.0)
            print(f"Loaded time_resolution_ns: {time_res}")
            return time_res

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
            print("Input matrix is 2D, assuming it already contains peak times.")
            return hist_matrix * self.time_resolution_ns * 0.15

        num_channels, num_horizontal_steps, _ = hist_matrix.shape
        distances = np.zeros((num_channels, num_horizontal_steps))
        
        for c in range(num_channels):
            for h in range(num_horizontal_steps):
                signal = hist_matrix[c, h, :]
                interpolated_time, _ = get_peak_time_and_amplitude(signal)
                distances[c, h] = interpolated_time * self.time_resolution_ns * 0.15
        return distances

    def _plot_error_distribution(self, errors: np.ndarray):
        """Generates and saves a histogram of the absolute errors."""
        plt.figure(figsize=(10, 6))
        
        # p99 = np.percentile(errors, 99.5)
        # plt.hist(errors, bins=50, range=(0, p99))
        plt.hist(errors, bins=50, range=(0, 2.5))
        
        dir_name = os.path.basename(os.path.normpath(self.directory))
        plt.title(f"Absolute Error Distribution\nDirectory: {dir_name}")
        plt.xlabel("Absolute Error (m)")
        plt.ylabel("Number of Points")
        plt.grid(True, linestyle='--', alpha=0.6)
        
        output_path = os.path.join(self.directory, "error_distribution.png")
        try:
            plt.savefig(output_path)
            print(f"Saved error distribution plot to {output_path}")
        except Exception as e:
            print(f"Error saving plot: {e}")
        finally:
            plt.close()

    def evaluate(self):
        print("Calculating distances for ground truth matrix...")
        gt_distances = self._calculate_distances(self.gt_matrix)
        print("Calculating distances for signal matrix...")
        signal_distances = self._calculate_distances(self.signal_matrix)

        if gt_distances.shape != signal_distances.shape:
            raise ValueError(f"Shape mismatch between GT distances {gt_distances.shape} and signal distances {signal_distances.shape}")

        valid_pixels_mask = gt_distances >= self.min_gt_distance
        total_pixels = gt_distances.size
        total_valid_pixels = np.sum(valid_pixels_mask)

        if total_valid_pixels == 0:
            print(f"Warning: No valid ground truth points found (all distances are <= {self.min_gt_distance}m). Evaluation cannot proceed.")
            return 0.0, 0.0

        print(f"Evaluating on {total_valid_pixels} valid pixels (out of {total_pixels} total) where GT distance > {self.min_gt_distance}m.")

        abs_error = np.abs(gt_distances - signal_distances)
        valid_errors = abs_error[valid_pixels_mask]
        
        mae = np.mean(valid_errors)

        correct_pixels = np.sum(valid_errors <= self.threshold)
        accuracy = (correct_pixels / total_valid_pixels) * 100

        if self.plot:
            self._plot_error_distribution(valid_errors)

        return mae, accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare gt.bl2/answer_matrix.bl2 and signal.bl2 for reconstruction evaluation.")
    parser.add_argument("--directory", required=True, help="Directory containing the .bl2 files and config.json.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for accuracy calculation (in meters).")
    parser.add_argument("--min-gt-distance", type=float, default=0.0, help="Minimum ground truth distance to include in evaluation (in meters). Defaults to 0.0.")
    parser.add_argument("--plot", action="store_true", help="Generate and save a plot of the absolute error distribution.")
    args = parser.parse_args()

    try:
        evaluator = Bl2Evaluator(args.directory, args.threshold, args.min_gt_distance, args.plot)
        mae, accuracy = evaluator.evaluate()

        print("\n--- Evaluation Results ---")
        print(f"Mean Absolute Error: {mae:.4f} meters")
        print(f"Accuracy (error <= {args.threshold}m): {accuracy:.2f}%")
        print(f"(Evaluation based on GT distances > {args.min_gt_distance}m)")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")