import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets_generator.hist_matrix_generator import LidarSignalDatasetGenerator
from reconstruction.peak_interval_reconstructor import PeakIntervalReconstructor
from datasets_generator.distance_visualizer import DistanceVisualizer

def run_evaluation(pcd_file: str, output_dir: str = "./evaluation_output"):
    """
    Runs the full evaluation pipeline:
    1. Generate clean, attacked datasets.
    2. Reconstruct the attacked dataset.
    3. Calculate distance matrices for all three.
    4. Generate and save difference heatmaps.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define file paths
    clean_npz = os.path.join(output_dir, "clean.npz")
    attacked_npz = os.path.join(output_dir, "attacked.npz")
    reconstructed_npz = os.path.join(output_dir, "reconstructed.npz")

    # --- Step 1: Generate Datasets ---
    print("--- Generating Clean Dataset ---")
    clean_generator = LidarSignalDatasetGenerator(
        lidar_type="PCD_VLP16", pcd_file=pcd_file, output_dir=output_dir, spoofer_type="off"
    )
    clean_generator.generate(num_frames=1, filename_prefix="clean")

    print("--- Generating Attacked Dataset ---")
    attack_generator = LidarSignalDatasetGenerator(
        lidar_type="PCD_VLP16", pcd_file=pcd_file, output_dir=output_dir, spoofer_type="adaptive_hfr_perturbation"
    )
    attack_generator.generate(num_frames=1, filename_prefix="attacked")

    # --- Step 2: Reconstruct Dataset ---
    print("--- Reconstructing Attacked Dataset ---")
    reconstructor = PeakIntervalReconstructor(attacked_npz)
    # You might need to adjust the HFR frequency based on the generator's settings
    reconstructor.reconstruct(hfr_freq_mhz=10.0) 
    reconstructor.save(reconstructed_npz)

    # --- Step 3: Calculate Distance Matrices ---
    print("--- Calculating Distance Matrices ---")
    vis_clean = DistanceVisualizer(clean_npz)
    vis_attacked = DistanceVisualizer(attacked_npz)
    vis_reconstructed = DistanceVisualizer(reconstructed_npz)

    dist_clean = vis_clean.calculate_distance_matrix()
    dist_attacked = vis_attacked.calculate_distance_matrix()
    dist_reconstructed = vis_reconstructed.calculate_distance_matrix()

    # --- Step 4: Generate and Save Difference Heatmaps ---
    print("--- Generating Difference Heatmaps ---")
    
    # Calculate differences
    diff_attack = dist_attacked - dist_clean
    diff_reconstructed = dist_reconstructed - dist_clean

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Attacked vs Clean
    im1 = axes[0].imshow(diff_attack, cmap='coolwarm', aspect='auto')
    axes[0].set_title('Distance Difference (Attacked - Clean)')
    axes[0].set_xlabel("Horizontal Index")
    axes[0].set_ylabel("Vertical Channel")
    fig.colorbar(im1, ax=axes[0], label='Distance Difference (m)')

    # Reconstructed vs Clean
    im2 = axes[1].imshow(diff_reconstructed, cmap='coolwarm', aspect='auto')
    axes[1].set_title('Distance Difference (Reconstructed - Clean)')
    axes[1].set_xlabel("Horizontal Index")
    axes[1].set_ylabel("Vertical Channel")
    fig.colorbar(im2, ax=axes[1], label='Distance Difference (m)')

    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, "difference_heatmaps.png")
    plt.savefig(heatmap_path)
    print(f"Saved difference heatmaps to {heatmap_path}")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run LiDAR reconstruction evaluation and generate heatmaps.")
    parser.add_argument("--pcd-file", type=str, required=True, help="Path to the PCD file for generating the datasets.")
    parser.add_argument("--output-dir", type=str, default="./evaluation_output", help="Directory to save all generated files and heatmaps.")
    args = parser.parse_args()

    run_evaluation(pcd_file=args.pcd_file, output_dir=args.output_dir)