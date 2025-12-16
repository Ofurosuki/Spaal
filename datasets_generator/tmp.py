import sys
import os
import argparse
import numpy as np
import blosc2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lifting.reconstruct_3d import get_peak_time_and_amplitude
from lifting.pixel_to_angle import get_angle_from_pixel

# --- Configuration ---
LIDAR_A_WIDTH = 192
TARGET_WIDTH = 1800
TARGET_HEIGHT = 32
Y_CROP_TOP = 40
X_PADDING_LEFT = (TARGET_WIDTH - LIDAR_A_WIDTH) // 2  # 804

PEAK_THRSHOLD = 1.1  # Minimum amplitude to consider a peak valid

# --- Helper Functions ---

def load_bl2_file(file_path: str) -> np.ndarray:
    """Loads and unpacks a Blosc2 compressed NumPy array."""
    print(f"Loading data from {file_path}...")
    try:
        with open(file_path, 'rb') as f:
            packed_array = f.read()
            unpacked_array = blosc2.unpack_array(packed_array)
        print(f"Loaded array with shape {unpacked_array.shape}")
        return unpacked_array
    except FileNotFoundError:
        print(f"Error: Input file not found at {file_path}")
        sys.exit(1)

def spherical_to_cartesian(depth, azimuth, altitude):
    """Converts spherical coordinates to Cartesian coordinates."""
    r = depth * 0.15  # Convert depth from bin number to meters
    az_rad = np.deg2rad(azimuth)
    alt_rad = np.deg2rad(altitude)
    x = r * np.cos(alt_rad) * np.sin(az_rad)
    y = r * np.cos(alt_rad) * np.cos(az_rad)
    z = -r * np.sin(alt_rad)
    return x, y, z

def extract_all_points(bl2_array):
    """Extract all valid points from bl2 array with their indices."""
    print("Extracting all points from bl2 array...")
    points_data = []  # List of (x, y, z, y_b, x_b, intensity)

    x_start = X_PADDING_LEFT
    x_end = X_PADDING_LEFT + LIDAR_A_WIDTH

    for y_b in tqdm(range(TARGET_HEIGHT), desc="Processing rows"):
        for x_b in range(x_start, x_end):
            hist = bl2_array[y_b, x_b, :]
            if np.any(hist > 0):
                depth, amp = get_peak_time_and_amplitude(hist, PEAK_THRSHOLD)
                if amp > 0 and depth > 0:
                    x_a = x_b - X_PADDING_LEFT
                    y_a = y_b + Y_CROP_TOP
                    azimuth, altitude = get_angle_from_pixel(x_a, y_a)
                    x, y, z = spherical_to_cartesian(depth, azimuth, altitude)
                    points_data.append((x, y, z, y_b, x_b, amp))

    print(f"Extracted {len(points_data)} valid points.")
    return points_data

class PointCloudViewer:
    def __init__(self, points_data, bl2_array, y_max_distance=15.0):
        """
        Initialize the viewer.
        points_data: List of (x, y, z, y_b, x_b, intensity)
        bl2_array: The original bl2 array for histogram access
        y_max_distance: Maximum Y distance in meters to display (default: 40.0)
        """
        self.points_data = np.array(points_data)
        self.bl2_array = bl2_array

        # Filter points by Y distance
        y_coords = self.points_data[:, 1]  # Y coordinate
        mask = y_coords <= y_max_distance
        self.points_data_filtered = self.points_data[mask]

        print(f"Filtered {np.sum(~mask)} points beyond {y_max_distance}m")
        print(f"Displaying {len(self.points_data_filtered)} points")

        # Extract coordinates and indices from filtered data
        self.xyz = self.points_data_filtered[:, :3]  # x, y, z
        self.y_indices = self.points_data_filtered[:, 3].astype(int)  # y_b
        self.x_indices = self.points_data_filtered[:, 4].astype(int)  # x_b
        self.intensities = self.points_data_filtered[:, 5]  # amplitude

        # Determine valid ranges for sliders
        self.y_min = int(self.y_indices.min())
        self.y_max = int(self.y_indices.max())
        self.x_min = int(self.x_indices.min())
        self.x_max = int(self.x_indices.max())

        print(f"Vertical range: {self.y_min} - {self.y_max}")
        print(f"Horizontal range: {self.x_min} - {self.x_max}")

        # Create figure with two subplots
        self.fig = plt.figure(figsize=(18, 8))

        # Left: 3D point cloud
        self.ax_3d = self.fig.add_subplot(121, projection='3d')

        # Right: Histogram signal
        self.ax_hist = self.fig.add_subplot(122)

        # Initial plot - 3D scatter
        self.scatter = self.ax_3d.scatter(
            self.xyz[:, 0],
            self.xyz[:, 1],
            self.xyz[:, 2],
            c='blue',
            s=1,
            alpha=0.5
        )

        # Highlighted point scatter (initially empty)
        self.highlight = self.ax_3d.scatter([], [], [], c='red', s=50, marker='o')

        # Set labels for 3D plot
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')
        self.ax_3d.set_title('Point Cloud Viewer (Y ≤ 40m)')

        # Initialize histogram plot
        self.hist_line, = self.ax_hist.plot([], [], 'b-', linewidth=1)
        self.hist_peak_line = self.ax_hist.axvline(x=0, color='r', linestyle='--', label='Peak')
        self.ax_hist.set_xlabel('Bin Number')
        self.ax_hist.set_ylabel('Amplitude')
        self.ax_hist.set_title('Histogram Signal')
        self.ax_hist.grid(True, alpha=0.3)
        self.ax_hist.legend()

        # Create sliders
        plt.subplots_adjust(bottom=0.15, left=0.05, right=0.95)

        ax_vertical = plt.axes([0.2, 0.05, 0.6, 0.03])
        ax_horizontal = plt.axes([0.2, 0.01, 0.6, 0.03])

        self.slider_vertical = Slider(
            ax_vertical,
            'Vertical (y)',
            self.y_min,
            self.y_max,
            valinit=self.y_min,
            valstep=1
        )

        self.slider_horizontal = Slider(
            ax_horizontal,
            'Horizontal (x)',
            self.x_min,
            self.x_max,
            valinit=self.x_min,
            valstep=1
        )

        # Connect slider events
        self.slider_vertical.on_changed(self.update)
        self.slider_horizontal.on_changed(self.update)

        # Initial update
        self.update(None)

    def update(self, val):
        """Update the highlighted point and histogram based on slider values."""
        y_val = int(self.slider_vertical.val)
        x_val = int(self.slider_horizontal.val)

        # Find points matching the selected indices
        mask = (self.y_indices == y_val) & (self.x_indices == x_val)
        highlighted_points = self.xyz[mask]

        # Update highlight scatter
        if len(highlighted_points) > 0:
            self.highlight._offsets3d = (
                highlighted_points[:, 0],
                highlighted_points[:, 1],
                highlighted_points[:, 2]
            )
            self.ax_3d.set_title(
                f'Point Cloud - Highlighting (y={y_val}, x={x_val}) - {len(highlighted_points)} points'
            )
        else:
            # No points at this location
            self.highlight._offsets3d = ([], [], [])
            self.ax_3d.set_title(
                f'Point Cloud - No points at (y={y_val}, x={x_val})'
            )

        # Get and display histogram
        histogram = self.bl2_array[y_val, x_val, :]
        bins = np.arange(len(histogram))

        # Update histogram plot
        self.hist_line.set_data(bins, histogram)
        self.ax_hist.relim()
        self.ax_hist.autoscale_view()

        # Find and mark peak if histogram has data
        if np.any(histogram > 0):
            peak_bin, peak_amp = get_peak_time_and_amplitude(histogram, PEAK_THRSHOLD)
            self.hist_peak_line.set_xdata([peak_bin, peak_bin])
            self.ax_hist.set_title(
                f'Histogram Signal (y={y_val}, x={x_val})\nPeak: bin={peak_bin:.1f}, amp={peak_amp:.1f}'
            )
        else:
            self.hist_peak_line.set_xdata([0, 0])
            self.ax_hist.set_title(f'Histogram Signal (y={y_val}, x={x_val}) - No data')

        self.fig.canvas.draw_idle()

    def show(self):
        """Display the viewer."""
        plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Visualize .bl2 point cloud with interactive highlighting by horizon/vertical index."
    )
    parser.add_argument("input_bl2", help="Path to the .bl2 file to visualize.")

    args = parser.parse_args()

    # Load bl2 file
    bl2_array = load_bl2_file(args.input_bl2)

    # Extract all points
    points_data = extract_all_points(bl2_array)

    if len(points_data) == 0:
        print("No valid points found in the bl2 file. Exiting.")
        return

    # Create and show viewer
    viewer = PointCloudViewer(points_data, bl2_array)
    viewer.show()

if __name__ == "__main__":
    main()
