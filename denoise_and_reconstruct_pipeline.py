import argparse
import os
import numpy as np
import torch
import open3d as o3d

# Add project root to sys.path to allow for module imports
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets_generator.hist_matrix_generator import LidarSignalDatasetGenerator
from HFR_Denoise.pipeline.denoise_pipeline import DenoisePipeline, run_denoising
from datasets_generator.hist_matrix_visualizer import HistMatrixVisualizer

def main():
    parser = argparse.ArgumentParser(description="Full pipeline: Generate, Denoise, and Reconstruct LiDAR data to .pcd and .pcd.bin.")
    
    # Generator args
    parser.add_argument("--lidar-type", type=str, default="PCD_VLP32c", choices=["VLP16", "PCD_VLP16", "PCD_VLP32c"], help="Type of LiDAR to use.")
    parser.add_argument("--pcd-directory", type=str, required=True, help="Path to the directory containing source PCD files.")
    parser.add_argument("--num-frames", type=int, default=81, help="Number of frames to process.")
    parser.add_argument("--start-frame", type=int, default=0, help="Starting frame index for processing PCD files.")
    parser.add_argument("--spoofer-type", type=str, default="adaptive_hfr_perturbation", choices=["adaptive_hfr_perturbation", "off"], help="Type of spoofer to use.")
    parser.add_argument("--spoofer-angle", type=float, default=90.0, help="The angle for the spoofer trigger, in degrees, counter-clockwise with 0 at the front.")
    parser.add_argument("--spoofer-altitude", type=float, default=50.0, help="The altitude for the spoofer trigger, in degrees.")
    parser.add_argument("--spoofer-width-deg", type=float, default=90.0, help="The angular width of the spoofer's attack cone in degrees.")

    # Denoiser args
    parser.add_argument("--ckpt-path", type=str, default=None, help="Optional: Path to the denoiser model checkpoint file. If not provided, denoising is skipped.")

    # Reconstructor args
    parser.add_argument("--output-dir", type=str, required=True, help="Path to the base directory to save output files.")

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
        spoofer_width_deg=args.spoofer_width_deg
    )
    generated_data = generator.generate(
        num_frames=args.num_frames,
        start_frame=args.start_frame,
        save_to_file=False # Important: Do not save to .npz
    )
    
    if not generated_data or 'signals' not in generated_data:
        print("Data generation failed or produced no signals. Exiting.")
        return

    # --- 2. Denoising (Optional) ---
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
        pcd_directory_path=args.pcd_directory # Needed for output file naming
    )
    
    num_frames_to_save = len(processed_signals)
    for i in range(num_frames_to_save):
        print(f"  Processing frame {i+1}/{num_frames_to_save}...")
        reconstructed_pcd = visualizer._reconstruct_point_cloud(i)

        if i < len(visualizer.pcd_files):
            base_name = os.path.basename(visualizer.pcd_files[i])
            base_name_no_ext = os.path.splitext(base_name)[0]
        else:
            base_name_no_ext = f"reconstructed_frame_{i}"

        # Save .pcd file
        pcd_path = os.path.join(pcd_out_dir, f"{base_name_no_ext}.pcd")
        o3d.io.write_point_cloud(pcd_path, reconstructed_pcd)
        print(f"    -> Saved {pcd_path}")

        # Save .pcd.bin file
        points = np.asarray(reconstructed_pcd.points)
        if points.shape[0] > 0:
            if reconstructed_pcd.has_colors():
                # Intensity is stored as grayscale color (r=g=b) normalized to [0, 1].
                # Extract from the red channel and scale to [0, 255] for the .bin format.
                colors = np.asarray(reconstructed_pcd.colors)
                intensity = (colors[:, 0] * 255.0).astype(np.float32).reshape(-1, 1)
            else:
                # Fallback to a dummy intensity if no color info is present
                intensity = np.full((points.shape[0], 1), 10.0, dtype=np.float32)

            ring_index = np.zeros((points.shape[0], 1), dtype=np.float32)
            nuscenes_points = np.hstack((points, intensity, ring_index)).astype(np.float32)
            
            bin_path = os.path.join(pcd_bin_out_dir, f"{base_name_no_ext}.pcd.bin")
            nuscenes_points.tofile(bin_path)
            print(f"    -> Saved {bin_path}")
        else:
            print(f"    -> Skipping empty point cloud for frame {i}.")


    print("\nPipeline finished successfully!")

if __name__ == '__main__':
    main()
