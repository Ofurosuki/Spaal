
import numpy as np
import open3d as o3d
import argparse
import os
import glob
import h5py

class HistMatrixVisualizer:
    def __init__(self, h5_file_path: str, pcd_directory_path: str = None):
        self.h5_file_path = h5_file_path
        self.pcd_directory_path = pcd_directory_path
        self.is_prediction = False 
        with h5py.File(h5_file_path, 'r') as data:
            print(f"Loading data from {h5_file_path}")
            if 'signals' not in data and 'prediction' in data:
                self.is_prediction = True
                self.hist_matrix = data['prediction'][:]
            else:
                self.hist_matrix = data['signals'][:]
            print(f"Loaded hist_matrix with shape: {self.hist_matrix.shape}")

            if 'initial_azimuth_offsets' in data:
                self.initial_azimuth_offsets = data['initial_azimuth_offsets'][:]
            else:
                print("Warning: 'initial_azimuth_offsets' not found in .h5 file. Defaulting to 0.0 for all frames.")
                self.initial_azimuth_offsets = [0.0]
            self.vertical_angles = data['vertical_angles'][:]
            self.fov = data['fov'][()]
            self.time_resolution_ns = data['time_resolution_ns'][()]

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
        print(f"type: {type(self.initial_azimuth_offsets)},")
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

                    # Perform Gaussian interpolation (parabolic on log values) for better accuracy
                    if 0 < peak_idx_global < len(signal) - 1:
                        y0 = signal[peak_idx_global - 1]
                        y1 = signal[peak_idx_global]
                        y2 = signal[peak_idx_global + 1]

                        # Ensure values are positive for log
                        if y0 > 0 and y1 > 0 and y2 > 0:
                            ln_y0 = np.log(y0)
                            ln_y1 = np.log(y1)
                            ln_y2 = np.log(y2)
                            
                            denominator = (ln_y0 - 2 * ln_y1 + ln_y2)
                            if abs(denominator) > 1e-9:
                                offset = (ln_y0 - ln_y2) / (2 * denominator)
                                highest_peak_time = peak_idx_global + offset
                            else:
                                highest_peak_time = float(peak_idx_global) # Fallback for flat log-parabola
                        else:
                            highest_peak_time = float(peak_idx_global) # Fallback if values are not suitable for log
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize LiDAR histogram matrix from .h5 file.")
    parser.add_argument("--h5-file", required=True, type=str, help="Path to the .h5 histogram matrix file.")
    parser.add_argument("--pcd-directory", type=str, default=None, help="Path to the directory with original .pcd files for comparison.")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to visualize.")
    
    args = parser.parse_args()

    visualizer = HistMatrixVisualizer(args.h5_file, args.pcd_directory)
    visualizer.visualize(args.frame)
