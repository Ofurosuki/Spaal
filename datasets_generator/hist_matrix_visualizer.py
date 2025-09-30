
import numpy as np
import open3d as o3d
import argparse
import os
import glob

class HistMatrixVisualizer:
    def __init__(self, npz_file_path: str = None, pcd_directory_path: str = None, data: dict = None):
        self.npz_file_path = npz_file_path
        self.pcd_directory_path = pcd_directory_path
        self.is_prediction = False

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
        if frame_index >= len(self.hist_matrix):
            raise ValueError(f"Frame index {frame_index} is out of bounds for hist_matrix with {len(self.hist_matrix)} frames.")
        if frame_index < len(self.initial_azimuth_offsets):
            current_azimuth_offset = self.initial_azimuth_offsets[frame_index]
        else:
            current_azimuth_offset = self.initial_azimuth_offsets[-1] if self.initial_azimuth_offsets else 0.0
            print(f"Warning: Frame index {frame_index} is out of bounds for azimuth offsets. Using last available offset.")

        frame_data = self.hist_matrix[frame_index]
        channels, horizontal_resolution, samples_per_scan = frame_data.shape

        for v_idx in range(channels):
            for h_idx in range(horizontal_resolution):
                if not self.is_prediction:
                    signal = frame_data[v_idx, h_idx, :]
                    
                    raises = np.flatnonzero((signal[:-1] < 0.01) & (signal[1:] >= 0.01)) + 1
                    if len(raises) == 0:
                        continue

                    # Find the pulse with the highest peak
                    peak_values = np.array([np.max(signal[r:min(len(signal), r + 50)]) for r in raises])
                    if len(peak_values) == 0:
                        continue
                    
                    # Determine the region of the highest pulse
                    highest_pulse_start_index = raises[np.argmax(peak_values)]
                    pulse_region = signal[highest_pulse_start_index:min(len(signal), highest_pulse_start_index + 50)]
                    
                    # Find the integer index of the peak within that pulse region
                    if len(pulse_region) == 0:
                        continue
                    peak_idx_in_region = np.argmax(pulse_region)
                    peak_idx_global = highest_pulse_start_index + peak_idx_in_region

                    # Perform parabolic interpolation for sub-sample precision
                    if 0 < peak_idx_global < len(signal) - 1:
                        y0 = signal[peak_idx_global - 1]
                        y1 = signal[peak_idx_global]
                        y2 = signal[peak_idx_global + 1]
                        
                        denominator = (y0 - 2 * y1 + y2)
                        if abs(denominator) > 1e-6: # Avoid division by zero for flat peaks
                            offset = (y0 - y2) / (2 * denominator)
                            highest_peak_time = peak_idx_global + offset
                        else:
                            highest_peak_time = float(peak_idx_global) # Fallback for flat peak
                    else:
                        highest_peak_time = float(peak_idx_global) # Fallback for peaks at signal boundary
                else:
                    highest_peak_time = frame_data[v_idx, h_idx, 0]


                distance_m = (highest_peak_time * self.time_resolution_ns) * 0.15
                
                altitude_deg = self.vertical_angles[v_idx]
                azimuth_deg = (h_idx / horizontal_resolution) * self.fov + current_azimuth_offset

                alpha = np.deg2rad(azimuth_deg)
                omega = np.deg2rad(altitude_deg)
                
                # Spherical to cartesian conversion (Y-forward, X-right, Z-up)
                x = distance_m * np.cos(omega) * np.sin(alpha)
                y = distance_m * np.cos(omega) * np.cos(alpha)
                z = distance_m * np.sin(omega)
                
                points.append([x, y, z])

        pcd = o3d.geometry.PointCloud()
        if points:
            pcd.points = o3d.utility.Vector3dVector(np.array(points))
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

    args = parser.parse_args()

    visualizer = HistMatrixVisualizer(npz_file_path=args.npz_file, pcd_directory_path=args.pcd_directory)
    if args.output_pcd_dir:
        visualizer.save_reconstructed_pcds(args.output_pcd_dir)
    else:
        visualizer.visualize(args.frame)
