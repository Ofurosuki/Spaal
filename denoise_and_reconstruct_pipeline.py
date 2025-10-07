import argparse
import os
import numpy as np
import torch
import open3d as o3d
from tqdm import tqdm

# Add project root to sys.path to allow for module imports
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets_generator.hist_matrix_generator import LidarSignalDatasetGenerator
from HFR_Denoise.pipeline.denoise_pipeline import DenoisePipeline, run_denoising
from datasets_generator.hist_matrix_visualizer import HistMatrixVisualizer

def get_peak_time_from_signal(signal: np.ndarray) -> float:
    """
    Finds the interpolated time of the highest peak in a signal.
    Returns 0.0 if no peak is found.
    """
    raises = np.flatnonzero((signal[:-1] < 0.01) & (signal[1:] >= 0.01)) + 1
    if len(raises) == 0:
        return 0.0

    peak_values = np.array([np.max(signal[r:min(len(signal), r + 50)]) for r in raises])
    if len(peak_values) == 0:
        return 0.0

    highest_pulse_start_index = raises[np.argmax(peak_values)]
    pulse_region = signal[highest_pulse_start_index:min(len(signal), highest_pulse_start_index + 50)]

    if len(pulse_region) == 0:
        return 0.0

    peak_idx_in_region = np.argmax(pulse_region)
    peak_idx_global = highest_pulse_start_index + peak_idx_in_region

    # Gaussian interpolation for sub-sample accuracy
    if 0 < peak_idx_global < len(signal) - 1:
        y0, y1, y2 = signal[peak_idx_global - 1 : peak_idx_global + 2]
        if y0 > 0 and y1 > 0 and y2 > 0:
            ln_y0, ln_y1, ln_y2 = np.log(y0), np.log(y1), np.log(y2)
            denominator = (ln_y0 - 2 * ln_y1 + ln_y2)
            if abs(denominator) > 1e-9:
                offset = (ln_y0 - ln_y2) / (2 * denominator)
                return peak_idx_global + offset
    return float(peak_idx_global)

def get_peaks_matrix_from_signals(signal_frame: np.ndarray) -> np.ndarray:
    """
    Converts a 3D signal matrix (channels, h_res, samples) to a 2D matrix of peak times.
    """
    channels, horizontal_resolution, _ = signal_frame.shape
    peaks_matrix = np.zeros((channels, horizontal_resolution), dtype=np.float32)
    for v_idx in range(channels):
        for h_idx in range(horizontal_resolution):
            peaks_matrix[v_idx, h_idx] = get_peak_time_from_signal(signal_frame[v_idx, h_idx, :])
    return peaks_matrix

def main():
    parser = argparse.ArgumentParser(description="Full pipeline: Generate, Denoise, Reconstruct, and Evaluate LiDAR data.")
    
    # Generator args
    parser.add_argument("--lidar-type", type=str, default="PCD_VLP32c", choices=["VLP16", "PCD_VLP16", "PCD_VLP32c"], help="Type of LiDAR to use.")
    parser.add_argument("--pcd-directory", type=str, required=True, help="Path to the directory containing source PCD files.")
    parser.add_argument("--num-frames", type=int, default=81, help="Number of frames to process.")
    parser.add_argument("--start-frame", type=int, default=0, help="Starting frame index for processing PCD files.")
    parser.add_argument("--spoofer-type", type=str, default="adaptive_hfr_perturbation", choices=["adaptive_hfr_perturbation", "off"], help="Type of spoofer to use.")
    parser.add_argument("--spoofer-angle", type=float, default=90.0, help="The angle for the spoofer trigger, in degrees, counter-clockwise with 0 at the front.")
    parser.add_argument("--spoofer-altitude", type=float, default=50.0, help="The altitude for the spoofer trigger, in degrees.")
    parser.add_argument("--spoofer-width-deg", type=float, default=90.0, help="The angular width of the spoofer's attack cone in degrees.")
    parser.add_argument("--sync-angle-step-deg", type=float, default=0.2, help="Sync angle step in degrees for VLP32c LiDAR.")

    # Denoiser args
    parser.add_argument("--ckpt-path", type=str, default=None, help="Optional: Path to the denoiser model checkpoint file. If not provided, denoising is skipped.")

    # Reconstructor args
    parser.add_argument("--output-dir", type=str, required=True, help="Path to the base directory to save output files.")

    # Evaluation args
    parser.add_argument("--tolerance", type=int, default=2, help="Tolerance in bins for accuracy evaluation.")

    args = parser.parse_args()

    # --- 0. Setup Output Directories ---
    pcd_out_dir = os.path.join(args.output_dir, "pcd")
    pcd_bin_out_dir = os.path.join(args.output_dir, "pcd_bin")
    os.makedirs(pcd_out_dir, exist_ok=True)
    os.makedirs(pcd_bin_out_dir, exist_ok=True)
    print(f"Output .pcd files will be saved to: {pcd_out_dir}")
    print(f"Output .pcd.bin files will be saved to: {pcd_bin_out_dir}")

    # --- 1. Data Generation ---
    print("\n--- Step 1: Generating LiDAR Data ---")
    generator = LidarSignalDatasetGenerator(
        lidar_type=args.lidar_type,
        pcd_directory=args.pcd_directory,
        spoofer_type=args.spoofer_type,
        spoofer_angle_deg=args.spoofer_angle,
        spoofer_altitude_deg=args.spoofer_altitude,
        spoofer_width_deg=args.spoofer_width_deg,
        sync_angle_step_deg=args.sync_angle_step_deg
    )
    generated_data = generator.generate(
        num_frames=args.num_frames,
        start_frame=args.start_frame,
        save_to_file=False
    )
    
    if not generated_data or 'signals' not in generated_data:
        print("Data generation failed or produced no signals. Exiting.")
        return

    # --- 2. Denoising (Optional) ---
    processed_signals = None

    if args.ckpt_path and os.path.exists(args.ckpt_path):
        print("\n--- Step 2: Denoising Signals ---")
        denoiser = DenoisePipeline(ckpt_path=args.ckpt_path)
        processed_signals = run_denoising(denoiser, generated_data['signals'])
    else:
        print("\n--- Step 2: Denoising Skipped ---")
        if not args.ckpt_path:
            print("Reason: --ckpt-path not provided.")
        else:
            print(f"Reason: Checkpoint file not found at {args.ckpt_path}")
        processed_signals = generated_data['signals']

    # --- 3. Reconstruction and Saving ---
    print("\n--- Step 3: Reconstructing and Saving Point Clouds ---")
    
    reconstruction_data = {
        'signals': processed_signals,
        'initial_azimuth_offsets': generated_data['initial_azimuth_offsets'],
        'vertical_angles': generated_data['vertical_angles'],
        'fov': generated_data['fov'],
        'time_resolution_ns': generated_data['time_resolution_ns']
    }

    visualizer = HistMatrixVisualizer(
        data=reconstruction_data,
        pcd_directory_path=args.pcd_directory
    )
    
    num_frames_to_save = len(processed_signals)
    for i in tqdm(range(num_frames_to_save), desc="Reconstructing and Saving"):
        reconstructed_pcd = visualizer._reconstruct_point_cloud(i)

        if i < len(visualizer.pcd_files):
            base_name = os.path.basename(visualizer.pcd_files[i])
            base_name_no_ext = os.path.splitext(base_name)[0]
        else:
            base_name_no_ext = f"reconstructed_frame_{i}"

        pcd_path = os.path.join(pcd_out_dir, f"{base_name_no_ext}.pcd")
        o3d.io.write_point_cloud(pcd_path, reconstructed_pcd)

        points = np.asarray(reconstructed_pcd.points)
        if points.shape[0] > 0:
            if reconstructed_pcd.has_colors():
                colors = np.asarray(reconstructed_pcd.colors)
                intensity = np.round(colors[:, 0]).astype(np.float32).reshape(-1, 1)
            else:
                intensity = np.full((points.shape[0], 1), 10.0, dtype=np.float32)

            ring_index = np.zeros((points.shape[0], 1), dtype=np.float32)
            nuscenes_points = np.hstack((points, intensity, ring_index)).astype(np.float32)
            
            bin_path = os.path.join(pcd_bin_out_dir, f"{base_name_no_ext}.pcd.bin")
            nuscenes_points.tofile(bin_path)

    # --- 4. Evaluating Accuracy ---
    print("\n--- Step 4: Evaluating Peak Prediction Accuracy ---")
    
    answer_matrix = generated_data['answer_matrix']
    
    if len(processed_signals.shape) == 3:
        processed_signals = np.expand_dims(processed_signals, axis=0)

    num_eval_frames = processed_signals.shape[0]
    all_true_peaks = []
    all_pred_peaks = []

    for i in tqdm(range(num_eval_frames), desc="Evaluating Accuracy"):
        prediction_peaks_matrix = get_peaks_matrix_from_signals(processed_signals[i])
        true_peaks_matrix = answer_matrix[i]

        valid_points_mask = true_peaks_matrix > 0
        
        if np.any(valid_points_mask):
            all_true_peaks.append(true_peaks_matrix[valid_points_mask])
            all_pred_peaks.append(prediction_peaks_matrix[valid_points_mask])

    if not all_true_peaks:
        print("No valid points found for evaluation.")
    else:
        true_peaks = np.concatenate(all_true_peaks)
        pred_peaks = np.concatenate(all_pred_peaks)

        absolute_errors = np.abs(true_peaks - pred_peaks)
        mae = np.mean(absolute_errors)

        correct_predictions = np.sum(absolute_errors <= args.tolerance)
        accuracy = correct_predictions / len(true_peaks)

        print(f"\nEvaluation based on {len(true_peaks)} valid return points across {num_eval_frames} frames:")
        print(f"Accuracy (within +/- {args.tolerance} bins): {accuracy:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f} bins")

    print("\nPipeline finished successfully!")

if __name__ == '__main__':
    main()
