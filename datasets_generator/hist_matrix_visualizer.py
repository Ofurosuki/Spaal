import numpy as np
import open3d as o3d
import argparse

class HistMatrixVisualizer:
    vertical_angles: list[int] = [-15, 1, -13, 3, -11, 5, -9, 7, -7, 9, -5, 11, -3, 13, -1, 15]

    def __init__(self, npy_file_path: str, pcd_file_path: str = None, time_resolution_ns: float = 1.0, initial_azimuth_offset: float = 0.0):
        self.npy_file_path = npy_file_path
        self.pcd_file_path = pcd_file_path
        self.time_resolution_ns = time_resolution_ns
        self.initial_azimuth_offset = initial_azimuth_offset
        self.hist_matrix = np.load(npy_file_path)

    def _reconstruct_point_cloud(self, frame_index: int = 0):
        points = []
        frame_data = self.hist_matrix[frame_index]
        channels, horizontal_resolution, samples_per_scan = frame_data.shape

        for v_idx in range(channels):
            for h_idx in range(horizontal_resolution):
                signal = frame_data[v_idx, h_idx, :]
                
                raises = np.flatnonzero((signal[:-1] < 0.01) & (signal[1:] >= 0.01)) + 1
                if len(raises) == 0:
                    continue

                peaks = np.array([np.max(signal[r:min(len(signal), r + 50)]) for r in raises])
                
                if len(peaks) == 0:
                    continue

                highest_peak_index = np.argmax(peaks)
                highest_peak_time = raises[highest_peak_index]

                distance_m = (highest_peak_time * self.time_resolution_ns) * 0.15
                
                altitude_deg = self.vertical_angles[v_idx]
                azimuth_deg = (h_idx / horizontal_resolution) * 360.0 + self.initial_azimuth_offset

                alpha = np.deg2rad(azimuth_deg)
                omega = np.deg2rad(altitude_deg)
                
                # Standard spherical to cartesian conversion (X-forward)
                y = distance_m * np.cos(omega) * np.cos(alpha)
                x = distance_m * np.cos(omega) * np.sin(alpha)
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
    parser = argparse.ArgumentParser(description="Visualize LiDAR histogram matrix.")
    parser.add_argument("--npy-file", type=str, help="Path to the .npy histogram matrix file.")
    parser.add_argument("--pcd-file", type=str, default=None, help="Path to the .pcd file for comparison.")
    parser.add_argument("--time-resolution-ns", type=float, default=0.2, help="Time resolution in nanoseconds.")
    parser.add_argument("--initial-azimuth-offset", type=float, default=0.0, help="Initial azimuth offset in degrees.")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to visualize.")
    
    args = parser.parse_args()

    visualizer = HistMatrixVisualizer(args.npy_file, args.pcd_file, args.time_resolution_ns, args.initial_azimuth_offset)
    visualizer.visualize(args.frame)