import numpy as np
import matplotlib.pyplot as plt
from spaal2.core.dummy_lidar.dummy_lidar_vlp16 import DummyLidarVLP16
from spaal2.core.precise_duration import PreciseDuration

class DistanceVisualizer:
    def __init__(self, hist_matrix_path: str):
        with np.load(hist_matrix_path) as data:
            self.signals = data['signals']
            self.time_resolution_ns = data['time_resolution_ns']
            # Reconstruct other necessary metadata if needed
            self.vertical_angles = data['vertical_angles']
            self.horizontal_fov = data['fov']

        # Initialize a dummy LiDAR to use its receive method
        self.lidar = DummyLidarVLP16(
            amplitude=1.0, # Amplitude doesn't affect peak detection logic
            pulse_width=PreciseDuration(nanoseconds=5), # This also might not be critical for receive
            time_resolution_ns=self.time_resolution_ns
        )
        self.num_frames, self.num_channels, self.num_horizontal, _ = self.signals.shape

    def calculate_distance_matrix(self, frame_index: int = 0) -> np.ndarray:
        """
        Calculates the distance for each scan in a given frame and returns it as a 2D matrix.
        """
        if frame_index >= self.num_frames:
            raise ValueError(f"Frame index {frame_index} is out of bounds.")

        distance_matrix = np.zeros((self.num_channels, self.num_horizontal))

        for v_idx in range(self.num_channels):
            for h_idx in range(self.num_horizontal):
                signal = self.signals[frame_index, v_idx, h_idx, :]
                
                # The receive method returns a list of tuples (peak_index, distance)
                # We'll take the first detected peak for simplicity.
                peaks = self.lidar.receive(signal)
                
                if peaks:
                    # Store the distance of the first peak
                    distance_matrix[v_idx, h_idx] = peaks[0][1]
                else:
                    # No peak detected, you might want to represent this with a specific value, e.g., 0 or NaN
                    distance_matrix[v_idx, h_idx] = 0 
        
        return distance_matrix

    def plot_heatmap(self, distance_matrix: np.ndarray, title: str = "Distance Heatmap"):
        """
        Plots a 2D heatmap of the distance matrix.
        """
        plt.figure(figsize=(12, 6))
        plt.imshow(distance_matrix, aspect='auto', cmap='viridis')
        plt.colorbar(label='Distance (m)')
        plt.title(title)
        plt.xlabel("Horizontal Index")
        plt.ylabel("Vertical Channel")
        plt.show()
