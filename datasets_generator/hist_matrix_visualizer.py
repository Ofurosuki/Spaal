
import numpy as np
import open3d as o3d
import argparse
import os
import glob
from typing import List

class HistMatrixVisualizer:
    def __init__(self, npz_file_path: str, pcd_directory_path: str = None):
        self.npz_file_path = npz_file_path
        self.pcd_directory_path = pcd_directory_path
        
        with np.load(npz_file_path) as data:
            self.hist_matrix = data['signals']
            if 'initial_azimuth_offsets' in data:
                self.initial_azimuth_offsets = data['initial_azimuth_offsets']
            else:
                self.initial_azimuth_offsets = [data.get('initial_azimuth_offset', 0.0)]
            self.vertical_angles = data['vertical_angles']
            self.fov = data['fov']
            self.time_resolution_ns = data['time_resolution_ns']

        # --- Parameters to match PcdLidarVLP16AmplitudeAuth ---
        self.auth_amplitude_ratios: List[float] = [1.0, 0.6, 0.3]
        self.auth_pulse_interval_ns: float = 20.0
        # -----------------------------------------------------

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

        frame_data = self.hist_matrix[frame_index]
        channels, horizontal_resolution, samples_per_scan = frame_data.shape

        interval_indices = int(self.auth_pulse_interval_ns / self.time_resolution_ns)
        tolerance_indices = int(interval_indices * 0.2) # 20% tolerance for timing

        for v_idx in range(channels):
            for h_idx in range(horizontal_resolution):
                signal = frame_data[v_idx, h_idx, :]
                
                # --- Start of Amplitude Ratio Verification Logic ---
                raises = np.flatnonzero((signal[:-1] < 0.01) & (signal[1:] >= 0.01)) + 1
                if len(raises) < len(self.auth_amplitude_ratios):
                    continue

                peaks = {r: np.max(signal[r:min(len(signal), r + 50)]) for r in raises}
                peak_times = sorted(peaks.keys())
                
                used_peak_times = set()

                for i in range(len(peak_times)):
                    if peak_times[i] in used_peak_times:
                        continue

                    sequence_times = [peak_times[i]]
                    for j in range(1, len(self.auth_amplitude_ratios)):
                        expected_next_time = sequence_times[-1] + interval_indices
                        found_next = False
                        for k in range(i + 1, len(peak_times)):
                            if peak_times[k] in used_peak_times:
                                continue
                            if abs(peak_times[k] - expected_next_time) <= tolerance_indices:
                                sequence_times.append(peak_times[k])
                                found_next = True
                                break
                        if not found_next:
                            break
                    
                    if len(sequence_times) == len(self.auth_amplitude_ratios):
                        base_amplitude = peaks[sequence_times[0]]
                        if base_amplitude < 0.1: continue

                        ratios_match = True
                        for k in range(1, len(self.auth_amplitude_ratios)):
                            expected_ratio = self.auth_amplitude_ratios[k] / self.auth_amplitude_ratios[0]
                            actual_ratio = peaks[sequence_times[k]] / base_amplitude
                            if abs(actual_ratio - expected_ratio) > 0.15:
                                ratios_match = False
                                break
                        
                        if ratios_match:
                            for t in sequence_times:
                                used_peak_times.add(t)

                            # This is a valid point, calculate its coordinates
                            valid_peak_time = sequence_times[0]
                            distance_m = (valid_peak_time * self.time_resolution_ns) * 0.15
                            
                            altitude_deg = self.vertical_angles[v_idx]
                            current_azimuth_offset = self.initial_azimuth_offsets[frame_index]
                            azimuth_deg = (h_idx / horizontal_resolution) * self.fov + current_azimuth_offset

                            alpha = np.deg2rad(azimuth_deg)
                            omega = np.deg2rad(altitude_deg)
                            
                            x = distance_m * np.cos(omega) * np.cos(alpha)
                            y = distance_m * np.cos(omega) * np.sin(alpha)
                            z = distance_m * np.sin(omega)
                            
                            points.append([x, y, z])
                # --- End of Amplitude Ratio Verification Logic ---

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
    parser = argparse.ArgumentParser(description="Visualize LiDAR histogram matrix from .npz file.")
    parser.add_argument("--npz-file", required=True, type=str, help="Path to the .npz histogram matrix file.")
    parser.add_argument("--pcd-directory", type=str, default=None, help="Path to the directory with original .pcd files for comparison.")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to visualize.")
    
    args = parser.parse_args()

    visualizer = HistMatrixVisualizer(args.npz_file, args.pcd_directory)
    visualizer.visualize(args.frame)
