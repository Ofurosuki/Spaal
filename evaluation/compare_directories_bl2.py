import numpy as np
import os
import blosc2
import argparse
import json
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
from tqdm import tqdm

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

class DirectoryBl2Evaluator:
    def __init__(self, gt_directory: str, denoised_directory: str, threshold: float,
                 min_gt_distance: float = 0.0, plot: bool = False, output_dir: str = None,
                 eval_angle: float = None, eval_width: float = None):
        self.gt_directory = gt_directory
        self.denoised_directory = denoised_directory
        self.threshold = threshold
        self.min_gt_distance = min_gt_distance
        self.plot = plot
        self.output_dir = output_dir if output_dir else denoised_directory
        self.eval_angle = eval_angle  # Center angle for evaluation (user-facing, 0=front, CCW)
        self.eval_width = eval_width  # Width of evaluation cone

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Find all subdirectories in GT directory
        self.subdirs = self._find_matching_subdirs()

        if not self.subdirs:
            raise ValueError(f"No matching subdirectories found between {gt_directory} and {denoised_directory}")

        print(f"Found {len(self.subdirs)} matching subdirectories to evaluate")

        if self.eval_angle is not None and self.eval_width is not None:
            print(f"Evaluation will be restricted to azimuth angle: {self.eval_angle}° ± {self.eval_width/2}° (width: {self.eval_width}°)")

    def _find_matching_subdirs(self) -> List[str]:
        """Find subdirectories that exist in both GT and denoised directories."""
        gt_subdirs = set([d for d in os.listdir(self.gt_directory)
                         if os.path.isdir(os.path.join(self.gt_directory, d))])
        denoised_subdirs = set([d for d in os.listdir(self.denoised_directory)
                               if os.path.isdir(os.path.join(self.denoised_directory, d))])

        matching = sorted(list(gt_subdirs & denoised_subdirs))
        return matching

    def _load_config(self, directory: str) -> float:
        """Load time resolution from config.json."""
        config_path = os.path.join(directory, 'config.json')
        if not os.path.exists(config_path):
            return 1.0
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            return config_data.get('time_resolution_ns', 1.0)

    def _load_bl2(self, file_path: str) -> np.ndarray:
        """Load a single bl2 file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found.")
        with open(file_path, 'rb') as f:
            data = blosc2.unpack_array(f.read())
            if data.ndim in [2, 3]:
                return data
            else:
                raise ValueError(f"Unsupported matrix dimension: {data.ndim}")

    def _calculate_distances(self, hist_matrix: np.ndarray, time_resolution_ns: float) -> np.ndarray:
        """Calculate distances from histogram matrix."""
        if hist_matrix.ndim == 2:
            return hist_matrix * time_resolution_ns * 0.15

        num_channels, num_horizontal_steps, _ = hist_matrix.shape
        distances = np.zeros((num_channels, num_horizontal_steps))

        for c in range(num_channels):
            for h in range(num_horizontal_steps):
                signal = hist_matrix[c, h, :]
                interpolated_time, _ = get_peak_time_and_amplitude(signal)
                distances[c, h] = interpolated_time * time_resolution_ns * 0.15
        return distances

    def _create_angle_mask(self, shape: Tuple[int, int], angles_matrix: np.ndarray = None,
                          initial_azimuth_offset: float = 0.0, fov: float = 360.0) -> np.ndarray:
        """
        Create a boolean mask for pixels within the specified angle range.

        Parameters:
        -----------
        shape : tuple
            Shape of the matrix (num_channels, num_horizontal_steps)
        angles_matrix : np.ndarray, optional
            Matrix of actual azimuth angles (if angles.bl2 exists)
        initial_azimuth_offset : float
            Initial azimuth offset for synthetic calculation
        fov : float
            Field of view

        Returns:
        --------
        np.ndarray
            Boolean mask (True for pixels within angle range)
        """
        if self.eval_angle is None or self.eval_width is None:
            # No angle restriction - return all True
            return np.ones(shape, dtype=bool)

        num_channels, num_horizontal_steps = shape

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
                if angles_matrix is not None:
                    azimuth_deg = angles_matrix[v_idx, h_idx]
                    if np.isnan(azimuth_deg):
                        # Use synthetic calculation as fallback
                        azimuth_deg = (h_idx / num_horizontal_steps) * fov + initial_azimuth_offset
                else:
                    # Synthetic calculation
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

    def _evaluate_single_pair(self, subdir_name: str) -> Dict:
        """Evaluate a single pair of GT and denoised subdirectories."""
        gt_subdir = os.path.join(self.gt_directory, subdir_name)
        denoised_subdir = os.path.join(self.denoised_directory, subdir_name)

        # Load config (use GT's config)
        config_path = os.path.join(gt_subdir, 'config.json')
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        time_resolution_ns = config_data.get('time_resolution_ns', 1.0)
        initial_azimuth_offset = config_data.get('initial_azimuth_offset', 0.0)
        fov = config_data.get('fov', 360.0)

        # Load GT signal
        gt_signal_path = os.path.join(gt_subdir, 'signal.bl2')
        gt_matrix = self._load_bl2(gt_signal_path)

        # Load denoised signal
        denoised_signal_path = os.path.join(denoised_subdir, 'signal.bl2')
        denoised_matrix = self._load_bl2(denoised_signal_path)

        # Calculate distances
        gt_distances = self._calculate_distances(gt_matrix, time_resolution_ns)
        denoised_distances = self._calculate_distances(denoised_matrix, time_resolution_ns)

        if gt_distances.shape != denoised_distances.shape:
            raise ValueError(f"Shape mismatch in {subdir_name}: GT {gt_distances.shape} vs Denoised {denoised_distances.shape}")

        # Try to load angles.bl2 for accurate angle masking
        angles_matrix = None
        angles_file = os.path.join(gt_subdir, 'angles.bl2')
        if os.path.exists(angles_file):
            with open(angles_file, 'rb') as f:
                angles_matrix = blosc2.unpack_array(f.read())

        # Create angle mask if angle restriction is specified
        angle_mask = self._create_angle_mask(
            gt_distances.shape,
            angles_matrix=angles_matrix,
            initial_azimuth_offset=initial_azimuth_offset,
            fov=fov
        )

        # Calculate metrics with both distance and angle masks
        valid_pixels_mask = (gt_distances > self.min_gt_distance) & angle_mask
        total_valid_pixels = np.sum(valid_pixels_mask)
        total_angle_pixels = np.sum(angle_mask)

        if total_valid_pixels == 0:
            return {
                'subdir': subdir_name,
                'mae': np.nan,
                'accuracy': np.nan,
                'valid_pixels': 0,
                'total_pixels': gt_distances.size,
                'angle_pixels': total_angle_pixels
            }

        abs_error = np.abs(gt_distances - denoised_distances)
        valid_errors = abs_error[valid_pixels_mask]

        mae = np.mean(valid_errors)
        correct_pixels = np.sum(valid_errors <= self.threshold)
        accuracy = (correct_pixels / total_valid_pixels) * 100

        return {
            'subdir': subdir_name,
            'mae': mae,
            'accuracy': accuracy,
            'valid_pixels': total_valid_pixels,
            'total_pixels': gt_distances.size,
            'angle_pixels': total_angle_pixels,
            'errors': valid_errors
        }

    def _plot_overall_error_distribution(self, all_errors: np.ndarray):
        """Generate and save a histogram of all errors combined."""
        plt.figure(figsize=(12, 6))

        plt.hist(all_errors, bins=100, range=(0, 2.5), alpha=0.7, edgecolor='black')

        mean_error = np.mean(all_errors)
        median_error = np.median(all_errors)

        plt.axvline(mean_error, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.3f}m')
        plt.axvline(median_error, color='g', linestyle='--', linewidth=2, label=f'Median: {median_error:.3f}m')

        plt.title(f"Overall Absolute Error Distribution\n{len(self.subdirs)} frames, {len(all_errors)} valid points")
        plt.xlabel("Absolute Error (m)")
        plt.ylabel("Number of Points")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        output_path = os.path.join(self.output_dir, "overall_error_distribution.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved overall error distribution plot to {output_path}")
        plt.close()

    def _plot_per_frame_metrics(self, results: List[Dict]):
        """Generate and save per-frame MAE and accuracy plots."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        subdirs = [r['subdir'] for r in results]
        maes = [r['mae'] for r in results]
        accuracies = [r['accuracy'] for r in results]

        # MAE plot
        ax1.plot(range(len(subdirs)), maes, marker='o', linestyle='-', markersize=3)
        ax1.axhline(np.nanmean(maes), color='r', linestyle='--', label=f'Mean: {np.nanmean(maes):.4f}m')
        ax1.set_xlabel('Frame Index')
        ax1.set_ylabel('MAE (m)')
        ax1.set_title('Per-Frame Mean Absolute Error')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)

        # Accuracy plot
        ax2.plot(range(len(subdirs)), accuracies, marker='o', linestyle='-', markersize=3, color='green')
        ax2.axhline(np.nanmean(accuracies), color='r', linestyle='--', label=f'Mean: {np.nanmean(accuracies):.2f}%')
        ax2.set_xlabel('Frame Index')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title(f'Per-Frame Accuracy (error <= {self.threshold}m)')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "per_frame_metrics.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved per-frame metrics plot to {output_path}")
        plt.close()

    def evaluate(self) -> Tuple[float, float, List[Dict]]:
        """Evaluate all subdirectory pairs and return overall metrics."""
        results = []
        all_errors = []

        print("\nEvaluating subdirectories...")
        for subdir in tqdm(self.subdirs, desc="Processing frames"):
            try:
                result = self._evaluate_single_pair(subdir)
                results.append(result)

                # Collect errors for overall distribution
                if 'errors' in result and len(result['errors']) > 0:
                    all_errors.extend(result['errors'])

            except Exception as e:
                print(f"\nError processing {subdir}: {e}")
                results.append({
                    'subdir': subdir,
                    'mae': np.nan,
                    'accuracy': np.nan,
                    'valid_pixels': 0,
                    'total_pixels': 0,
                    'error': str(e)
                })

        # Calculate overall metrics
        valid_results = [r for r in results if not np.isnan(r['mae'])]

        if not valid_results:
            print("Error: No valid results obtained!")
            return np.nan, np.nan, results

        overall_mae = np.mean([r['mae'] for r in valid_results])
        overall_accuracy = np.mean([r['accuracy'] for r in valid_results])

        # Generate plots if requested
        if self.plot and all_errors:
            all_errors_array = np.array(all_errors)
            self._plot_overall_error_distribution(all_errors_array)
            self._plot_per_frame_metrics(results)

        return overall_mae, overall_accuracy, results

    def save_results_csv(self, results: List[Dict], output_path: str = None):
        """Save detailed results to CSV file."""
        if output_path is None:
            output_path = os.path.join(self.output_dir, "evaluation_results.csv")

        with open(output_path, 'w') as f:
            # Write header
            f.write("subdir,mae,accuracy,valid_pixels,total_pixels\n")

            # Write data
            for r in results:
                f.write(f"{r['subdir']},{r['mae']:.6f},{r['accuracy']:.2f},{r['valid_pixels']},{r['total_pixels']}\n")

        print(f"\nSaved detailed results to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare signal.bl2 files between GT and denoised directories.")
    parser.add_argument("--gt-dir", required=True, help="Directory containing ground truth subdirectories.")
    parser.add_argument("--denoised-dir", required=True, help="Directory containing denoised subdirectories.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for accuracy calculation (in meters).")
    parser.add_argument("--min-gt-distance", type=float, default=0.0, help="Minimum ground truth distance to include in evaluation (in meters).")
    parser.add_argument("--plot", action="store_true", help="Generate and save plots of error distributions and per-frame metrics.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for plots and CSV. Defaults to denoised-dir.")
    parser.add_argument("--save-csv", action="store_true", help="Save detailed per-frame results to CSV.")
    parser.add_argument("--eval-angle", type=float, default=None,
                        help="Center angle for evaluation (user-facing: 0=front, CCW). Corresponds to spoofer-angle.")
    parser.add_argument("--eval-width", type=float, default=None,
                        help="Width of evaluation angle cone in degrees. Corresponds to spoofer-width-deg.")

    args = parser.parse_args()

    try:
        evaluator = DirectoryBl2Evaluator(
            gt_directory=args.gt_dir,
            denoised_directory=args.denoised_dir,
            threshold=args.threshold,
            min_gt_distance=args.min_gt_distance,
            plot=args.plot,
            output_dir=args.output_dir,
            eval_angle=args.eval_angle,
            eval_width=args.eval_width
        )

        overall_mae, overall_accuracy, results = evaluator.evaluate()

        print("\n" + "="*70)
        print("OVERALL EVALUATION RESULTS")
        print("="*70)
        print(f"Number of frames evaluated: {len(results)}")
        print(f"Overall Mean Absolute Error: {overall_mae:.4f} meters")
        print(f"Overall Accuracy (error <= {args.threshold}m): {overall_accuracy:.2f}%")
        print(f"(Evaluation based on GT distances > {args.min_gt_distance}m)")
        print("="*70)

        # Show statistics
        valid_maes = [r['mae'] for r in results if not np.isnan(r['mae'])]
        if valid_maes:
            print(f"\nMAE Statistics:")
            print(f"  Min:    {np.min(valid_maes):.4f}m")
            print(f"  Max:    {np.max(valid_maes):.4f}m")
            print(f"  Median: {np.median(valid_maes):.4f}m")
            print(f"  Std:    {np.std(valid_maes):.4f}m")

        valid_accs = [r['accuracy'] for r in results if not np.isnan(r['accuracy'])]
        if valid_accs:
            print(f"\nAccuracy Statistics:")
            print(f"  Min:    {np.min(valid_accs):.2f}%")
            print(f"  Max:    {np.max(valid_accs):.2f}%")
            print(f"  Median: {np.median(valid_accs):.2f}%")
            print(f"  Std:    {np.std(valid_accs):.2f}%")

        # Save CSV if requested
        if args.save_csv:
            evaluator.save_results_csv(results)

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
