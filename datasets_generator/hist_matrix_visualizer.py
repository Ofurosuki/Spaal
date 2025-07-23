
import numpy as np
import open3d as o3d
import argparse

class HistMatrixVisualizer:
    def __init__(self, npz_file_path: str, pcd_file_path: str = None, time_resolution_ns: float = 1.0):
        self.npz_file_path = npz_file_path
        self.pcd_file_path = pcd_file_path
        self.time_resolution_ns = time_resolution_ns
        
        with np.load(npz_file_path) as data:
            self.hist_matrix = data['signals']
            self.initial_azimuth_offset = data['initial_azimuth_offset']
            self.vertical_angles = data['vertical_angles']
            self.fov = data['fov']


    def _reconstruct_point_cloud(self, frame_index: int = 0):
        points = []
        frame_data = self.hist_matrix[frame_index]
        channels, horizontal_resolution, _ = frame_data.shape

        for v_idx in range(channels):
            for h_idx in range(horizontal_resolution):
                signal = frame_data[v_idx, h_idx, :]
                raises = np.flatnonzero((signal[:-1] < 0.1) & (signal[1:] >= 0.1)) + 1
                if not raises.any():
                    continue
                
                # A simple approach: take the first detected peak
                highest_peak_time = raises[0]
                distance_m = (highest_peak_time * self.time_resolution_ns) * 0.15
                
                altitude_deg = self.vertical_angles[v_idx]
                azimuth_deg = (h_idx / horizontal_resolution) * 360.0 + self.initial_azimuth_offset

                alpha = np.deg2rad(azimuth_deg)
                omega = np.deg2rad(altitude_deg)
                
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
        reconstructed_pcd.paint_uniform_color([1, 0, 0])  # Red

        geometries = [reconstructed_pcd]

        if self.pcd_file_path:
            original_pcd = o3d.io.read_point_cloud(self.pcd_file_path)
            original_pcd.paint_uniform_color([0, 0, 1])  # Blue
            geometries.append(original_pcd)

        o3d.visualization.draw_geometries(geometries)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize LiDAR histogram matrix from .npz file.")
    parser.add_argument("--npz-file", type=str, help="Path to the .npz histogram matrix file.")
    parser.add_argument("--pcd-file", type=str, default=None, help="Path to the .pcd file for comparison.")
    parser.add_argument("--time-resolution-ns", type=float, default=0.2, help="Time resolution in nanoseconds.")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to visualize.")
    
    args = parser.parse_args()

    visualizer = HistMatrixVisualizer(args.npz_file, args.pcd_file, args.time_resolution_ns)
    visualizer.visualize(args.frame)
