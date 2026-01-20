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
    def __init__(self, gt_directory: str, signal_directory: str, threshold: float, min_gt_distance: float = 0.0, plot: bool = False, eval_angle: float = None, eval_width: float = None):
        self.gt_directory = gt_directory
        self.signal_directory = signal_directory
        self.threshold = threshold
        self.min_gt_distance = min_gt_distance
        self.plot = plot
        self.eval_angle = eval_angle  # Center angle for evaluation (user-facing, 0=front, CCW)
        self.eval_width = eval_width  # Width of evaluation cone
        self.time_resolution_ns = self._load_config()

        if self.eval_angle is not None and self.eval_width is not None:
            print(f"Evaluation will be restricted to azimuth angle: {self.eval_angle}° ± {self.eval_width/2}° (width: {self.eval_width}°)")

        # Load GT signal.bl2 from gt_directory
        self.gt_path = os.path.join(gt_directory, 'signal.bl2')
        if not os.path.exists(self.gt_path):
            raise FileNotFoundError(f"Could not find 'signal.bl2' in GT directory: {gt_directory}")

        # Load signal.bl2 from signal_directory
        self.signal_path = os.path.join(signal_directory, 'signal.bl2')
        if not os.path.exists(self.signal_path):
            raise FileNotFoundError(f"Could not find 'signal.bl2' in signal directory: {signal_directory}")

        print(f"Loading GT signal from: {self.gt_path}")
        print(f"Loading comparison signal from: {self.signal_path}")

        self.gt_matrix = self._load_bl2(self.gt_path)
        self.signal_matrix = self._load_bl2(self.signal_path)

    def _load_config(self) -> float:
        # Try to load config from GT directory first
        config_path = os.path.join(self.gt_directory, 'config.json')
        if not os.path.exists(config_path):
            # Fallback to signal directory
            config_path = os.path.join(self.signal_directory, 'config.json')
            if not os.path.exists(config_path):
                print("Warning: config.json not found in either directory. Using default time_resolution_ns=1.0")
                return 1.0
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            time_res = config_data.get('time_resolution_ns', 1.0)
            print(f"Loaded time_resolution_ns: {time_res} from {config_path}")
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

    def _create_angle_mask(self, shape: Tuple[int, int], gt_directory: str) -> np.ndarray:
        """
        Create a boolean mask for pixels within the specified angle range.

        Parameters:
        -----------
        shape : tuple
            Shape of the matrix (num_channels, num_horizontal_steps)
        gt_directory : str
            Directory containing config.json and angles.bl2

        Returns:
        --------
        np.ndarray
            Boolean mask (True for pixels within angle range)
        """
        if self.eval_angle is None or self.eval_width is None:
            # No angle restriction - return all True
            return np.ones(shape, dtype=bool)

        num_channels, num_horizontal_steps = shape

        # Load config for azimuth calculation
        config_path = os.path.join(gt_directory, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            initial_azimuth_offset = config_data.get('initial_azimuth_offset', 0.0)
            fov = config_data.get('fov', 360.0)
        else:
            initial_azimuth_offset = 0.0
            fov = 360.0

        # Try to load angles.bl2 for accurate angle masking
        angles_matrix = None
        angles_file = os.path.join(gt_directory, 'angles.bl2')
        if os.path.exists(angles_file):
            with open(angles_file, 'rb') as f:
                import blosc2
                angles_matrix = blosc2.unpack_array(f.read())

        # Convert user-facing angle to internal angle (add 90 degrees)
        internal_angle_deg = (self.eval_angle + 90) % 360
        eval_center_az = internal_angle_deg * 100  # Convert to 0.01 deg units
        eval_width_az = self.eval_width * 100
        eval_start_az = eval_center_az - eval_width_az / 2
        eval_end_az = eval_center_az + eval_width_az / 2

        angle_mask = np.zeros(shape, dtype=bool)

        for v_idx in range(num_channels):
            for h_idx in range(num_horizontal_steps):
                # Get azimuth angle for this pixel
                if angles_matrix is not None and not np.isnan(angles_matrix[v_idx, h_idx]):
                    azimuth_deg = angles_matrix[v_idx, h_idx]
                else:
                    # Use synthetic calculation as fallback
                    azimuth_deg = (h_idx / num_horizontal_steps) * fov + initial_azimuth_offset

                # Convert to internal angle system (add 90 degrees)
                internal_azimuth_deg = (azimuth_deg + 90) % 360
                azimuth_key = internal_azimuth_deg * 100

                # Check if within angle range
                is_in_range = (eval_start_az <= azimuth_key <= eval_end_az)

                # Handle wrapping around 360 degrees
                if eval_start_az < 0:
                    is_in_range = (azimuth_key >= (36000 + eval_start_az) or azimuth_key <= eval_end_az)
                elif eval_end_az > 36000:
                    is_in_range = (azimuth_key >= eval_start_az or azimuth_key <= (eval_end_az - 36000))

                angle_mask[v_idx, h_idx] = is_in_range

        return angle_mask

    def _plot_error_distribution(self, errors: np.ndarray):
        """Generates and saves a histogram of the absolute errors."""
        plt.figure(figsize=(10, 6))
        
        # p99 = np.percentile(errors, 99.5)
        # plt.hist(errors, bins=50, range=(0, p99))
        plt.hist(errors, bins=50, range=(0, 2.5))
        
        gt_name = os.path.basename(os.path.normpath(self.gt_directory))
        signal_name = os.path.basename(os.path.normpath(self.signal_directory))
        plt.title(f"Absolute Error Distribution\nGT: {gt_name} vs Signal: {signal_name}")
        plt.xlabel("Absolute Error (m)")
        plt.ylabel("Number of Points")
        plt.grid(True, linestyle='--', alpha=0.6)

        output_path = os.path.join(self.signal_directory, "error_distribution.png")
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

<<<<<<< HEAD
        # Create angle mask if angle restriction is specified
        angle_mask = self._create_angle_mask(gt_distances.shape, self.gt_directory)

        # Calculate metrics with both distance and angle masks
        valid_pixels_mask = (gt_distances > self.min_gt_distance) & angle_mask
=======
        valid_pixels_mask = gt_distances >= self.min_gt_distance
>>>>>>> 1bbb0ae6e35556cfcca48d13dbacc006f6112dd7
        total_pixels = gt_distances.size
        total_valid_pixels = np.sum(valid_pixels_mask)
        total_angle_pixels = np.sum(angle_mask)

        if total_valid_pixels == 0:
            print(f"Warning: No valid ground truth points found (all distances are <= {self.min_gt_distance}m or outside angle range). Evaluation cannot proceed.")
            return 0.0, 0.0

        print(f"Evaluating on {total_valid_pixels} valid pixels (out of {total_pixels} total)")
        print(f"  - Pixels in angle range: {total_angle_pixels}")
        print(f"  - Pixels with GT distance > {self.min_gt_distance}m and in angle range: {total_valid_pixels}")

        abs_error = np.abs(gt_distances - signal_distances)
        valid_errors = abs_error[valid_pixels_mask]

        mae = np.mean(valid_errors)

        correct_pixels = np.sum(valid_errors <= self.threshold)
        accuracy = (correct_pixels / total_valid_pixels) * 100

        if self.plot:
            self._plot_error_distribution(valid_errors)

        return mae, accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare signal.bl2 from two directories for reconstruction evaluation.")
    parser.add_argument("--gt-directory", required=True, help="Directory containing ground truth signal.bl2.")
    parser.add_argument("--signal-directory", required=True, help="Directory containing the signal.bl2 to compare (e.g., denoised data).")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for accuracy calculation (in meters).")
    parser.add_argument("--min-gt-distance", type=float, default=0.0, help="Minimum ground truth distance to include in evaluation (in meters). Defaults to 0.0.")
    parser.add_argument("--plot", action="store_true", help="Generate and save a plot of the absolute error distribution.")
    parser.add_argument("--eval-angle", type=float, default=90,
                        help="Center angle for evaluation (user-facing: 0=front, CCW). Corresponds to spoofer-angle.")
    parser.add_argument("--eval-width", type=float, default=90,
                        help="Width of evaluation angle cone in degrees. Corresponds to spoofer-width-deg.")
    args = parser.parse_args()

    try:
        evaluator = Bl2Evaluator(args.gt_directory, args.signal_directory, args.threshold, args.min_gt_distance, args.plot, args.eval_angle, args.eval_width)
        mae, accuracy = evaluator.evaluate()

        print("\n--- Evaluation Results ---")
        print(f"Mean Absolute Error: {mae:.4f} meters")
        print(f"Accuracy (error <= {args.threshold}m): {accuracy:.2f}%")
        print(f"(Evaluation based on GT distances > {args.min_gt_distance}m)")
        if args.eval_angle is not None and args.eval_width is not None:
            print(f"(Angle restriction: {args.eval_angle}° ± {args.eval_width/2}°)")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")