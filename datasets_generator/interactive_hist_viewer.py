import sys
import os
import argparse
import numpy as np
import blosc2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import json

def get_peak_time_and_amplitude(signal: np.ndarray):
    """
    Finds the interpolated time and amplitude of the highest peak in a signal.
    Returns (0.0, 0.0) if no peak is found.
    """
    raises = np.flatnonzero((signal[:-1] < 0.01) & (signal[1:] >= 0.01)) + 1
    if len(raises) == 0:
        return 0.0, 0.0

    peak_values = np.array([np.max(signal[r:min(len(signal), r + 50)]) for r in raises])
    if len(peak_values) == 0:
        return 0.0, 0.0

    peak_amplitude = np.max(peak_values)
    if peak_amplitude < 0.01:
        return 0.0, 0.0

    highest_pulse_start_index = raises[np.argmax(peak_values)]
    pulse_region = signal[highest_pulse_start_index:min(len(signal), highest_pulse_start_index + 50)]

    if len(pulse_region) == 0:
        return 0.0, 0.0

    peak_idx_in_region = np.argmax(pulse_region)
    peak_idx_global = highest_pulse_start_index + peak_idx_in_region

    interpolated_time = float(peak_idx_global)
    if 0 < peak_idx_global < len(signal) - 1:
        y0, y1, y2 = signal[peak_idx_global - 1:peak_idx_global + 2]
        if y0 > 0 and y1 > 0 and y2 > 0:
            ln_y0, ln_y1, ln_y2 = np.log(y0), np.log(y1), np.log(y2)
            denominator = (ln_y0 - 2 * ln_y1 + ln_y2)
            if abs(denominator) > 1e-9:
                offset = (ln_y0 - ln_y2) / (2 * denominator)
                interpolated_time = peak_idx_global + offset

    return interpolated_time, peak_amplitude


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


def load_dataset(dataset_root_path: str, frame_index: int, use_answer_matrix: bool = False, use_gt_mask: bool = False):
    """Load dataset from directory structure."""
    sample_dirs = sorted([d for d in os.listdir(dataset_root_path) if os.path.isdir(os.path.join(dataset_root_path, d))])
    if not sample_dirs:
        raise FileNotFoundError(f"No sample directories found in {dataset_root_path}")
    if frame_index >= len(sample_dirs):
        raise ValueError(f"Frame index {frame_index} is out of bounds for {len(sample_dirs)} sample directories.")

    sample_dir = os.path.join(dataset_root_path, sample_dirs[frame_index])
    print(f"Loading data from {sample_dir}")

    # Load config
    with open(os.path.join(sample_dir, 'config.json'), 'r') as f:
        config_data = json.load(f)

    # Load signal or answer_matrix
    if use_answer_matrix:
        bl2_file = os.path.join(sample_dir, 'answer_matrix.bl2')
    else:
        bl2_file = os.path.join(sample_dir, 'signal.bl2')

    hist_matrix = load_bl2_file(bl2_file)

    # Load ground truth mask if requested
    gt_mask = None
    if use_gt_mask:
        answer_file = os.path.join(sample_dir, 'answer_matrix.bl2')
        if os.path.exists(answer_file):
            answer_matrix = load_bl2_file(answer_file)
            gt_mask = answer_matrix > 0
            print(f"Loaded ground truth mask with {np.sum(gt_mask)} valid points")
        else:
            print(f"Warning: --use-gt-mask specified but answer_matrix.bl2 not found")

    # Load angles if available
    angles = None
    angles_file = os.path.join(sample_dir, 'angles.bl2')
    if os.path.exists(angles_file):
        angles = load_bl2_file(angles_file)
        print(f"Loaded angles.bl2 with shape: {angles.shape}")

    return hist_matrix, config_data, gt_mask, angles


def spherical_to_cartesian(depth, azimuth, altitude, time_resolution_ns=1.0):
    """Converts spherical coordinates to Cartesian coordinates."""
    r = depth * time_resolution_ns * 0.15  # Convert depth to meters
    az_rad = np.deg2rad(azimuth)
    alt_rad = np.deg2rad(altitude)
    x = r * np.cos(alt_rad) * np.sin(az_rad)
    y = r * np.cos(alt_rad) * np.cos(az_rad)
    z = r * np.sin(alt_rad)
    return x, y, z


def extract_all_points(hist_matrix, config_data, gt_mask=None, azimuth_angles=None, y_max_distance=100.0):
    """Extract all valid points from hist_matrix with their indices."""
    print("Extracting all points from hist_matrix...")
    points_data = []  # List of (x, y, z, v_idx, h_idx, depth, intensity)

    is_prediction = len(hist_matrix.shape) == 2

    if is_prediction:
        channels, horizontal_resolution = hist_matrix.shape
    else:
        channels, horizontal_resolution, _ = hist_matrix.shape

    vertical_angles = config_data.get('vertical_angles', [])
    fov = config_data.get('fov', 360.0)
    time_resolution_ns = config_data.get('time_resolution_ns', 1.0)
    initial_azimuth_offset = config_data.get('initial_azimuth_offset', 0.0)

    for v_idx in tqdm(range(channels), desc="Processing channels"):
        altitude_deg = vertical_angles[v_idx] if v_idx < len(vertical_angles) else 0.0

        for h_idx in range(horizontal_resolution):
            # Apply ground truth mask if available
            if gt_mask is not None:
                if not gt_mask[v_idx, h_idx]:
                    continue

            if not is_prediction:
                signal = hist_matrix[v_idx, h_idx, :]

                # Check if signal dimension is 1 (already contains interpolated_time)
                if len(signal) == 1:
                    highest_peak_time = signal[0]
                    if highest_peak_time <= 0:
                        continue
                    intensity = 100  # Default intensity
                else:
                    # Normal case: calculate peak time and amplitude from histogram
                    highest_peak_time, peak_amplitude = get_peak_time_and_amplitude(signal)
                    if highest_peak_time == 0.0:
                        continue
                    intensity = peak_amplitude
            else:
                highest_peak_time = hist_matrix[v_idx, h_idx]
                if highest_peak_time <= 0:
                    continue
                intensity = 100  # Default intensity

            # Calculate azimuth
            if azimuth_angles is not None:
                actual_azimuth = azimuth_angles[v_idx, h_idx]
                if not np.isnan(actual_azimuth):
                    azimuth_deg = actual_azimuth
                else:
                    azimuth_deg = (h_idx / horizontal_resolution) * fov + initial_azimuth_offset
            else:
                azimuth_deg = (h_idx / horizontal_resolution) * fov + initial_azimuth_offset

            # Convert to Cartesian
            x, y, z = spherical_to_cartesian(highest_peak_time, azimuth_deg, altitude_deg, time_resolution_ns)

            # Filter by Y distance
            if y <= y_max_distance:
                points_data.append((x, y, z, v_idx, h_idx, highest_peak_time, intensity))

    print(f"Extracted {len(points_data)} valid points.")
    return points_data


class InteractiveHistViewer:
    def __init__(self, points_data, hist_matrix, config_data, y_max_distance=100.0, gt_hist_matrix=None):
        """
        Initialize the interactive histogram viewer.

        Parameters:
        -----------
        points_data : list
            List of (x, y, z, v_idx, h_idx, depth, intensity)
        hist_matrix : np.ndarray
            The original histogram matrix
        config_data : dict
            Configuration data from dataset
        y_max_distance : float
            Maximum Y distance in meters to display
        gt_hist_matrix : np.ndarray, optional
            Ground truth histogram matrix for comparison
        """
        self.points_data = np.array(points_data)
        self.hist_matrix = hist_matrix
        self.config_data = config_data
        self.is_prediction = len(hist_matrix.shape) == 2
        self.gt_hist_matrix = gt_hist_matrix
        self.has_gt = gt_hist_matrix is not None

        print(f"Displaying {len(self.points_data)} points (Y ≤ {y_max_distance}m)")

        # Extract coordinates and indices
        self.xyz = self.points_data[:, :3]  # x, y, z
        self.v_indices = self.points_data[:, 3].astype(int)  # vertical index
        self.h_indices = self.points_data[:, 4].astype(int)  # horizontal index
        self.depths = self.points_data[:, 5]  # depth values
        self.intensities = self.points_data[:, 6]  # intensity values

        # Determine valid ranges for sliders
        self.v_min = int(self.v_indices.min())
        self.v_max = int(self.v_indices.max())
        self.h_min = int(self.h_indices.min())
        self.h_max = int(self.h_indices.max())

        print(f"Vertical range: {self.v_min} - {self.v_max}")
        print(f"Horizontal range: {self.h_min} - {self.h_max}")

        # Create figure with subplots
        if self.has_gt:
            # 3 subplots: point cloud, prediction histogram, GT histogram
            self.fig = plt.figure(figsize=(24, 8))
            self.ax_3d = self.fig.add_subplot(131, projection='3d')
            self.ax_hist = self.fig.add_subplot(132)
            self.ax_hist_gt = self.fig.add_subplot(133)
        else:
            # 2 subplots: point cloud, histogram
            self.fig = plt.figure(figsize=(18, 8))
            self.ax_3d = self.fig.add_subplot(121, projection='3d')
            self.ax_hist = self.fig.add_subplot(122)
            self.ax_hist_gt = None

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
        self.ax_3d.set_title(f'Point Cloud Viewer (Y ≤ {y_max_distance}m)')

        # Initialize histogram plot for prediction
        self.hist_line, = self.ax_hist.plot([], [], 'b-', linewidth=1)
        self.hist_peak_line = self.ax_hist.axvline(x=0, color='r', linestyle='--', label='Peak')
        self.ax_hist.set_xlabel('Bin Number / Interpolated Time')
        self.ax_hist.set_ylabel('Amplitude')
        self.ax_hist.set_title('Prediction Histogram')
        self.ax_hist.grid(True, alpha=0.3)
        self.ax_hist.legend()

        # Initialize GT histogram plot if available
        if self.has_gt:
            self.hist_line_gt, = self.ax_hist_gt.plot([], [], 'g-', linewidth=1)
            self.hist_peak_line_gt = self.ax_hist_gt.axvline(x=0, color='r', linestyle='--', label='Peak')
            self.ax_hist_gt.set_xlabel('Bin Number / Interpolated Time')
            self.ax_hist_gt.set_ylabel('Amplitude')
            self.ax_hist_gt.set_title('Ground Truth Histogram')
            self.ax_hist_gt.grid(True, alpha=0.3)
            self.ax_hist_gt.legend()

        # Create sliders and buttons
        plt.subplots_adjust(bottom=0.2, left=0.05, right=0.95)

        # Vertical slider and buttons
        ax_vertical = plt.axes([0.2, 0.10, 0.6, 0.03])
        self.slider_vertical = Slider(
            ax_vertical,
            'Vertical (channel)',
            self.v_min,
            self.v_max,
            valinit=self.v_min,
            valstep=1
        )

        # Vertical decrement button
        ax_v_dec = plt.axes([0.12, 0.10, 0.04, 0.03])
        self.btn_v_dec = Button(ax_v_dec, '-1')
        self.btn_v_dec.on_clicked(lambda event: self.change_vertical(-1))

        # Vertical increment button
        ax_v_inc = plt.axes([0.84, 0.10, 0.04, 0.03])
        self.btn_v_inc = Button(ax_v_inc, '+1')
        self.btn_v_inc.on_clicked(lambda event: self.change_vertical(1))

        # Horizontal slider and buttons
        ax_horizontal = plt.axes([0.2, 0.05, 0.6, 0.03])
        self.slider_horizontal = Slider(
            ax_horizontal,
            'Horizontal (azimuth)',
            self.h_min,
            self.h_max,
            valinit=self.h_min,
            valstep=1
        )

        # Horizontal decrement button
        ax_h_dec = plt.axes([0.12, 0.05, 0.04, 0.03])
        self.btn_h_dec = Button(ax_h_dec, '-1')
        self.btn_h_dec.on_clicked(lambda event: self.change_horizontal(-1))

        # Horizontal increment button
        ax_h_inc = plt.axes([0.84, 0.05, 0.04, 0.03])
        self.btn_h_inc = Button(ax_h_inc, '+1')
        self.btn_h_inc.on_clicked(lambda event: self.change_horizontal(1))

        # Large step buttons for horizontal (±10)
        ax_h_dec10 = plt.axes([0.06, 0.05, 0.04, 0.03])
        self.btn_h_dec10 = Button(ax_h_dec10, '-10')
        self.btn_h_dec10.on_clicked(lambda event: self.change_horizontal(-10))

        ax_h_inc10 = plt.axes([0.90, 0.05, 0.04, 0.03])
        self.btn_h_inc10 = Button(ax_h_inc10, '+10')
        self.btn_h_inc10.on_clicked(lambda event: self.change_horizontal(10))

        # Connect slider events
        self.slider_vertical.on_changed(self.update)
        self.slider_horizontal.on_changed(self.update)

        # Initial update
        self.update(None)

    def change_vertical(self, delta):
        """Change vertical slider value by delta."""
        new_val = int(self.slider_vertical.val) + delta
        new_val = max(self.v_min, min(self.v_max, new_val))
        self.slider_vertical.set_val(new_val)

    def change_horizontal(self, delta):
        """Change horizontal slider value by delta."""
        new_val = int(self.slider_horizontal.val) + delta
        new_val = max(self.h_min, min(self.h_max, new_val))
        self.slider_horizontal.set_val(new_val)

    def update(self, val):
        """Update the highlighted point and histogram based on slider values."""
        v_val = int(self.slider_vertical.val)
        h_val = int(self.slider_horizontal.val)

        # Find points matching the selected indices
        mask = (self.v_indices == v_val) & (self.h_indices == h_val)
        highlighted_points = self.xyz[mask]

        # Update highlight scatter
        if len(highlighted_points) > 0:
            self.highlight._offsets3d = (
                highlighted_points[:, 0],
                highlighted_points[:, 1],
                highlighted_points[:, 2]
            )
            depth_val = self.depths[mask][0]
            self.ax_3d.set_title(
                f'Point Cloud - Highlighting (ch={v_val}, az={h_val}) - {len(highlighted_points)} points\nDepth={depth_val:.2f}'
            )
        else:
            # No points at this location
            self.highlight._offsets3d = ([], [], [])
            self.ax_3d.set_title(
                f'Point Cloud - No points at (ch={v_val}, az={h_val})'
            )

        # Get and display histogram
        if self.is_prediction:
            # For prediction (2D array), show the single value
            value = self.hist_matrix[v_val, h_val]
            self.hist_line.set_data([value], [1.0])
            self.hist_peak_line.set_xdata([value, value])
            self.ax_hist.set_xlim(max(0, value - 10), value + 10)
            self.ax_hist.set_ylim(0, 1.5)
            self.ax_hist.set_title(
                f'Interpolated Time (ch={v_val}, az={h_val})\nValue: {value:.2f}'
            )
        else:
            # For full histogram (3D array)
            histogram = self.hist_matrix[v_val, h_val, :]

            # Check if it's a single-value case
            if histogram.shape[0] == 1:
                value = histogram[0]
                self.hist_line.set_data([value], [1.0])
                self.hist_peak_line.set_xdata([value, value])
                self.ax_hist.set_xlim(max(0, value - 10), value + 10)
                self.ax_hist.set_ylim(0, 1.5)
                self.ax_hist.set_title(
                    f'Interpolated Time (ch={v_val}, az={h_val})\nValue: {value:.2f}'
                )
            else:
                bins = np.arange(len(histogram))

                # Update histogram plot
                self.hist_line.set_data(bins, histogram)
                self.ax_hist.relim()
                self.ax_hist.autoscale_view()

                # Find and mark peak if histogram has data
                if np.any(histogram > 0):
                    peak_bin, peak_amp = get_peak_time_and_amplitude(histogram)
                    self.hist_peak_line.set_xdata([peak_bin, peak_bin])
                    self.ax_hist.set_title(
                        f'Histogram Signal (ch={v_val}, az={h_val})\nPeak: bin={peak_bin:.1f}, amp={peak_amp:.2f}'
                    )
                else:
                    self.hist_peak_line.set_xdata([0, 0])
                    self.ax_hist.set_title(f'Histogram Signal (ch={v_val}, az={h_val}) - No data')

        # Display GT histogram if available
        if self.has_gt:
            is_gt_prediction = len(self.gt_hist_matrix.shape) == 2

            if is_gt_prediction:
                # GT is 2D array (answer_matrix style)
                gt_value = self.gt_hist_matrix[v_val, h_val]
                self.hist_line_gt.set_data([gt_value], [1.0])
                self.hist_peak_line_gt.set_xdata([gt_value, gt_value])
                self.ax_hist_gt.set_xlim(max(0, gt_value - 10), gt_value + 10)
                self.ax_hist_gt.set_ylim(0, 1.5)
                self.ax_hist_gt.set_title(
                    f'GT Interpolated Time (ch={v_val}, az={h_val})\nValue: {gt_value:.2f}'
                )
            else:
                # GT is 3D array (full histogram)
                gt_histogram = self.gt_hist_matrix[v_val, h_val, :]

                if gt_histogram.shape[0] == 1:
                    gt_value = gt_histogram[0]
                    self.hist_line_gt.set_data([gt_value], [1.0])
                    self.hist_peak_line_gt.set_xdata([gt_value, gt_value])
                    self.ax_hist_gt.set_xlim(max(0, gt_value - 10), gt_value + 10)
                    self.ax_hist_gt.set_ylim(0, 1.5)
                    self.ax_hist_gt.set_title(
                        f'GT Interpolated Time (ch={v_val}, az={h_val})\nValue: {gt_value:.2f}'
                    )
                else:
                    gt_bins = np.arange(len(gt_histogram))
                    self.hist_line_gt.set_data(gt_bins, gt_histogram)
                    self.ax_hist_gt.relim()
                    self.ax_hist_gt.autoscale_view()

                    if np.any(gt_histogram > 0):
                        gt_peak_bin, gt_peak_amp = get_peak_time_and_amplitude(gt_histogram)
                        self.hist_peak_line_gt.set_xdata([gt_peak_bin, gt_peak_bin])
                        self.ax_hist_gt.set_title(
                            f'GT Histogram (ch={v_val}, az={h_val})\nPeak: bin={gt_peak_bin:.1f}, amp={gt_peak_amp:.2f}'
                        )
                    else:
                        self.hist_peak_line_gt.set_xdata([0, 0])
                        self.ax_hist_gt.set_title(f'GT Histogram (ch={v_val}, az={h_val}) - No data')

        self.fig.canvas.draw_idle()

    def show(self):
        """Display the viewer."""
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive histogram viewer for LiDAR dataset with point cloud visualization."
    )
    parser.add_argument("--dataset-root-path", required=True, type=str,
                        help="Path to the root directory of the dataset.")
    parser.add_argument("--frame", type=int, default=0,
                        help="Frame index to visualize.")
    parser.add_argument("--use-answer-matrix", action='store_true',
                        help="Use answer_matrix.bl2 instead of signal.bl2.")
    parser.add_argument("--use-gt-mask", action='store_true',
                        help="Only visualize points where ground truth exists.")
    parser.add_argument("--gt-dataset-root-path", type=str, default=None,
                        help="Path to ground truth dataset for comparison. If provided, GT histogram will be displayed alongside prediction.")
    parser.add_argument("--y-max-distance", type=float, default=100.0,
                        help="Maximum Y distance in meters to display (default: 100.0).")

    args = parser.parse_args()

    # Load prediction dataset
    hist_matrix, config_data, gt_mask, azimuth_angles = load_dataset(
        args.dataset_root_path,
        args.frame,
        args.use_answer_matrix,
        args.use_gt_mask
    )

    # Load GT dataset if provided
    gt_hist_matrix = None
    if args.gt_dataset_root_path:
        print("\nLoading ground truth dataset for comparison...")
        gt_hist_matrix, _, _, _ = load_dataset(
            args.gt_dataset_root_path,
            args.frame,
            use_answer_matrix=False,  # Load full histogram from GT
            use_gt_mask=False
        )
        print(f"GT histogram shape: {gt_hist_matrix.shape}")

    # Extract all points
    points_data = extract_all_points(
        hist_matrix,
        config_data,
        gt_mask,
        azimuth_angles,
        args.y_max_distance
    )

    if len(points_data) == 0:
        print("No valid points found in the dataset. Exiting.")
        return

    # Create and show viewer
    viewer = InteractiveHistViewer(points_data, hist_matrix, config_data, args.y_max_distance, gt_hist_matrix)
    viewer.show()


if __name__ == "__main__":
    main()
