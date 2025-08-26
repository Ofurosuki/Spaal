import numpy as np
import open3d as o3d
import argparse
import os
import glob

# Assuming the script is run from the project root or the package is installed
from spaal2.core.noise_utils import detect_echo
from spaal2.core.dummy_lidar.echo import Echo
from spaal2.core.dummy_lidar.echogroup import EchoGroup

class HistMatrixVisualizer:
    def __init__(self, npz_file_path: str, pcd_directory_path: str = None):
        self.npz_file_path = npz_file_path
        self.pcd_directory_path = pcd_directory_path
        
        with np.load(npz_file_path) as data:
            self.hist_matrix = data['signals']
            self.initial_azimuth_offsets = data.get('initial_azimuth_offsets', [0.0])
            self.vertical_angles = data['vertical_angles']
            self.fov = data['fov']
            self.time_resolution_ns = data['time_resolution_ns']

            # Load authentication parameters
            self.pulse_num = int(data.get('pulse_num', 1))
            if self.pulse_num > 1:
                self.consider_amp = bool(data['consider_amp'])
                self.gt_intervals_ns = data['gt_intervals_ns']
                self.gt_amps_ratio = data.get('gt_amps_ratio', np.ones(self.pulse_num))
                self.max_torelance_error_ns = float(data['max_torelance_error_ns'])
                self.amplitude = float(data['amplitude'])
                self.pulse_half_width_ns = int(data['pulse_half_width_ns'])
                self.thd_factor = float(data['thd_factor'])
                self.use_height_estimation = bool(data['use_height_estimation'])
                self.max_amp_torelance_error = float(data['max_amp_torelance_error'])

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
        channels, horizontal_resolution, _ = frame_data.shape

        for v_idx in range(channels):
            for h_idx in range(horizontal_resolution):
                signal = frame_data[v_idx, h_idx, :]
                
                max_height = self.amplitude
                effective_echoes = detect_echo(signal, max_height, self.thd_factor, self.use_height_estimation, self.pulse_half_width_ns, time_resolution_ns=self.time_resolution_ns)

                if not effective_echoes: continue

                if self.pulse_num <= 1:
                    strongest_echo = max(effective_echoes, key=lambda echo: echo.peak_height)
                    point = self._calculate_point(strongest_echo.peak_position, v_idx, h_idx, frame_index, horizontal_resolution)
                    points.append(point)
                    continue

                # --- Full `receive` logic implementation ---
                certified_echoes: list[EchoGroup] = []
                error = self.max_torelance_error_ns
                peaks = np.array([x.peak_position for x in effective_echoes])
                
                for i in range(len(effective_echoes) - self.pulse_num + 1):
                    if peaks[i] + self.gt_intervals_ns[-1] - error > len(signal):
                        break
                    
                    certified = True
                    fingerprint_echoes: list[Echo] = [effective_echoes[i]]
                    first_echo_amp = effective_echoes[i].peak_height
                    if first_echo_amp == 0: continue

                    for pulse_index in range(self.pulse_num - 1):
                        base_position = self.gt_intervals_ns[pulse_index]
                        applicable_echoes_indices = np.flatnonzero(np.abs((peaks - peaks[i]) - base_position) <= error)
                        applicable_echoes_indices = applicable_echoes_indices[applicable_echoes_indices != i]

                        if len(applicable_echoes_indices) == 0:
                            certified = False
                            break
                        
                        selected_echo_idx = -1
                        if self.consider_amp:
                            for a_echo_idx in applicable_echoes_indices:
                                actual_ratio = effective_echoes[a_echo_idx].peak_height / first_echo_amp
                                ideal_ratio = self.gt_amps_ratio[pulse_index + 1] / self.gt_amps_ratio[0]
                                if abs(actual_ratio - ideal_ratio) <= self.max_amp_torelance_error:
                                    selected_echo_idx = a_echo_idx
                                    break
                            if selected_echo_idx == -1:
                                certified = False
                                break
                        else:
                            selected_echo_idx = applicable_echoes_indices[0]

                        fingerprint_echoes.append(effective_echoes[selected_echo_idx])

                    if certified:
                        certified_echoes.append(EchoGroup(fingerprint_echoes))
                
                if certified_echoes:
                    certified_echoes.sort(key=lambda x: (-x[0].peak_height, x[0].peak_position))
                    strongest_group = certified_echoes[0]
                    point = self._calculate_point(strongest_group[0].peak_position, v_idx, h_idx, frame_index, horizontal_resolution)
                    points.append(point)

        pcd = o3d.geometry.PointCloud()
        if points:
            pcd.points = o3d.utility.Vector3dVector(np.array(points))
        return pcd

    def _calculate_point(self, peak_time, v_idx, h_idx, frame_index, horizontal_resolution):
        distance_m = (peak_time * self.time_resolution_ns) * 0.15
        altitude_deg = self.vertical_angles[v_idx]
        current_azimuth_offset = self.initial_azimuth_offsets[frame_index]
        azimuth_deg = (h_idx / horizontal_resolution) * self.fov + current_azimuth_offset

        alpha = np.deg2rad(azimuth_deg)
        omega = np.deg2rad(altitude_deg)
        
        x = distance_m * np.sin(alpha) * np.cos(omega)
        y = distance_m * np.cos(alpha) * np.cos(omega)
        z = distance_m * np.sin(omega)
        
        return [x, y, z]

    def visualize(self, frame_index: int = 0):
        print("Reconstructing point cloud from histogram matrix...")
        reconstructed_pcd = self._reconstruct_point_cloud(frame_index)
        if not reconstructed_pcd.has_points():
            print("Warning: No points were reconstructed from the .npz file.")
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