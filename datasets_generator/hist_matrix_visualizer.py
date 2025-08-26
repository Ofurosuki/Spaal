
import numpy as np
import open3d as o3d
import argparse
import os
import glob

from spaal2.core.noise_utils import detect_echo
from spaal2.core.dummy_lidar.echo import Echo
from spaal2.core.dummy_lidar.echogroup import EchoGroup

class HistMatrixVisualizer:
    def __init__(self, npz_file_path: str, pcd_directory_path: str = None, debug: bool = False):
        self.npz_file_path = npz_file_path
        self.pcd_directory_path = pcd_directory_path
        self.debug = debug
        
        with np.load(npz_file_path) as data:
            self.hist_matrix = data['signals']
            self.initial_azimuth_offsets = data.get('initial_azimuth_offsets', [0.0])
            self.vertical_angles = data['vertical_angles']
            self.fov = data['fov']
            self.time_resolution_ns = data['time_resolution_ns']

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
        debug_prints = 0
        max_debug_prints = 5 # Limit debug output

        if frame_index >= len(self.hist_matrix):
            raise ValueError(f"Frame index {frame_index} is out of bounds for hist_matrix with {len(self.hist_matrix)} frames.")

        frame_data = self.hist_matrix[frame_index]
        channels, horizontal_resolution, _ = frame_data.shape

        for v_idx in range(channels):
            for h_idx in range(horizontal_resolution):
                signal = frame_data[v_idx, h_idx, :]
                if np.max(signal) == 0: continue

                max_height = self.amplitude
                effective_echoes = detect_echo(signal, max_height, self.thd_factor, self.use_height_estimation, self.pulse_half_width_ns, time_resolution_ns=self.time_resolution_ns)

                if not effective_echoes: continue
                
                # --- Start Debug Block ---
                if self.debug and debug_prints < max_debug_prints:
                    print(f"\n--- DEBUG: (v_idx={v_idx}, h_idx={h_idx}) ---")
                    print(f"Time Resolution (ns): {self.time_resolution_ns}")
                    print(f"Detected {len(effective_echoes)} echoes.")
                    peaks_for_debug = np.array([x.peak_position for x in effective_echoes])
                    print(f"Peak positions (samples): {peaks_for_debug}")
                # --- End Debug Block ---

                if self.pulse_num <= 1:
                    strongest_echo = max(effective_echoes, key=lambda echo: echo.peak_height)
                    point = self._calculate_point(strongest_echo.peak_position, v_idx, h_idx, frame_index, horizontal_resolution)
                    points.append(point)
                    continue

                error_samples = self.max_torelance_error_ns / self.time_resolution_ns
                intervals_samples = np.array(self.gt_intervals_ns) 
                peaks = np.array([x.peak_position for x in effective_echoes])

                if self.debug and debug_prints < max_debug_prints:
                    print(f"Intervals (samples): {intervals_samples}")
                    print(f"Tolerance (samples): {error_samples}")

                certified_groups_for_this_scan = []
                for i in range(len(effective_echoes) - self.pulse_num + 1):
                    certified = True
                    first_echo_amp = effective_echoes[i].peak_height
                    if first_echo_amp == 0: continue

                    if self.debug and debug_prints < max_debug_prints: print(f"  - Checking sequence starting at peak {i} (pos: {peaks[i]}) ...")

                    for pulse_index in range(self.pulse_num - 1):
                        base_position_samples = intervals_samples[pulse_index]
                        diffs = np.abs((peaks - peaks[i]) - base_position_samples)
                        applicable_echoes_indices = np.flatnonzero(diffs <= error_samples)
                        applicable_echoes_indices = applicable_echoes_indices[applicable_echoes_indices != i]

                        if len(applicable_echoes_indices) == 0:
                            certified = False
                            if self.debug and debug_prints < max_debug_prints: print(f"    - Pulse {pulse_index+2}: FAILED (No peak at expected interval)")
                            break
                        
                        selected_echo_idx = -1
                        if self.consider_amp:
                            # ... amplitude check ...
                            pass # For now, focus on timing
                        else:
                            selected_echo_idx = applicable_echoes_indices[0]
                        
                        if self.debug and debug_prints < max_debug_prints: print(f"    - Pulse {pulse_index+2}: PASSED")

                    if certified:
                        if self.debug and debug_prints < max_debug_prints: print(f"  -> CERTIFIED sequence starting at peak {i}")
                        # This is a simplified placeholder for creating the EchoGroup
                        certified_groups_for_this_scan.append(effective_echoes[i]) 

                if certified_groups_for_this_scan:
                    strongest_certified_echo = max(certified_groups_for_this_scan, key=lambda echo: echo.peak_height)
                    point = self._calculate_point(strongest_certified_echo.peak_position, v_idx, h_idx, frame_index, horizontal_resolution)
                    points.append(point)
                
                if self.debug and debug_prints < max_debug_prints: debug_prints += 1

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
        reconstructed_pcd.paint_uniform_color([1, 0, 0])
        geometries = [reconstructed_pcd]
        if self.pcd_files and frame_index < len(self.pcd_files):
            pcd_file_to_load = self.pcd_files[frame_index]
            print(f"Loading original PCD for comparison: {pcd_file_to_load}")
            original_pcd = o3d.io.read_point_cloud(pcd_file_to_load)
            original_pcd.paint_uniform_color([0, 0, 1])
            geometries.append(original_pcd)
        elif self.pcd_directory_path:
             print(f"Warning: Frame index {frame_index} is out of bounds for the number of PCD files found ({len(self.pcd_files)}). Original PCD will not be displayed.")
        o3d.visualization.draw_geometries(geometries, window_name=f"Frame {frame_index}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize LiDAR histogram matrix from .npz file.")
    parser.add_argument("--npz-file", required=True, type=str, help="Path to the .npz histogram matrix file.")
    parser.add_argument("--pcd-directory", type=str, default=None, help="Path to the directory with original .pcd files for comparison.")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to visualize.")
    parser.add_argument("--debug", action='store_true', help="Enable debug print statements.")
    
    args = parser.parse_args()

    visualizer = HistMatrixVisualizer(args.npz_file, args.pcd_directory, debug=args.debug)
    visualizer.visualize(args.frame)
