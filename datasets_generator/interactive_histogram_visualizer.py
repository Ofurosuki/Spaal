import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import argparse
import os
import json
import blosc2

def visualize_interactive(dataset_root_path: str, frame_index: int):
    """
    Loads a frame from the new dataset format and provides an interactive plot
    with sliders to select and view individual histograms.
    """
    # 1. Load data from the dataset directory structure
    try:
        sample_dirs = sorted([d for d in os.listdir(dataset_root_path) if os.path.isdir(os.path.join(dataset_root_path, d))])
        if not sample_dirs:
            raise FileNotFoundError(f"No sample directories found in {dataset_root_path}")
        if frame_index >= len(sample_dirs):
            raise ValueError(f"Frame index {frame_index} is out of bounds for {len(sample_dirs)} sample directories.")
        
        sample_dir = os.path.join(dataset_root_path, sample_dirs[frame_index])
        print(f"Loading data from {sample_dir}")

        with open(os.path.join(sample_dir, 'config.json'), 'r') as f:
            config_data = json.load(f)
        
        vertical_angles = config_data['vertical_angles']

        with open(os.path.join(sample_dir, 'signal.bl2'), 'rb') as f:
            hist_matrix = blosc2.unpack_array(f.read())

        labels_path = os.path.join(sample_dir, 'labels.bl2')
        if os.path.exists(labels_path):
            with open(labels_path, 'rb') as f:
                label_matrix = blosc2.unpack_array(f.read())
        else:
            label_matrix = None

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Get dimensions from the loaded data
    num_channels, num_horizontal_steps, num_samples = hist_matrix.shape

    # 3. Setup the initial Matplotlib plot
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.3)  # Make room for sliders

    # Initial indices for the first plot
    initial_altitude_idx = 0
    initial_azimuth_idx = 0

    # Create x-axis for the histogram plot (representing time samples)
    time_axis = np.arange(num_samples)

    # Create x-axis for the histogram plot (representing time samples)
    time_axis = np.arange(num_samples)

    def draw_histogram(alt_idx, azi_idx):
        """Clears the axes and redraws the histogram with colored labels."""
        ax.clear()
        
        signal = hist_matrix[alt_idx, azi_idx, :]
        
        # Plot the signal outline for clarity
        ax.plot(time_axis, signal, color='black', linewidth=0.75, label="Signal")

        # Check if label_matrix exists and has the correct shape
        if label_matrix is not None and label_matrix.shape == hist_matrix.shape:
            labels = label_matrix[alt_idx, azi_idx, :]
            bar_height = 10  # A small height for the color bar on the x-axis

            # Draw a colored bar on the x-axis for genuine signals
            ax.fill_between(time_axis, 0, bar_height, where=labels == 1, 
                            color='skyblue', alpha=0.8, label='Genuine (1)')
            # Draw a colored bar on the x-axis for HFR signals
            ax.fill_between(time_axis, 0, bar_height, where=labels == 2, 
                            color='pink', alpha=0.8, label='HFR (2)')
        
        ax.set_xlabel("Time Sample Index")
        ax.set_ylabel("Intensity")
        ax.set_ylim(0, bar_height)
        ax.grid(True)
        ax.legend(loc='upper right')
        
        # Update the title to show current indices and angle
        altitude_deg = vertical_angles[alt_idx]
        fig.suptitle(f'Altitude Idx: {alt_idx} (~{altitude_deg:.2f} deg), Azimuth Idx: {azi_idx}')

    # 4. Create axes for the sliders
    ax_altitude = plt.axes([0.25, 0.15, 0.65, 0.03])
    ax_azimuth = plt.axes([0.25, 0.1, 0.65, 0.03])

    # 5. Create the sliders for altitude and azimuth
    slider_altitude = Slider(
        ax=ax_altitude,
        label='Altitude Index',
        valmin=0,
        valmax=num_channels - 1,
        valinit=initial_altitude_idx,
        valstep=1
    )

    slider_azimuth = Slider(
        ax=ax_azimuth,
        label='Azimuth Index',
        valmin=0,
        valmax=num_horizontal_steps - 1,
        valinit=initial_azimuth_idx,
        valstep=1
    )

    # 6. Define the update function to be called by sliders
    def update(val):
        alt_idx = int(slider_altitude.val)
        azi_idx = int(slider_azimuth.val)
        draw_histogram(alt_idx, azi_idx)
        fig.canvas.draw_idle()

    # Initial draw
    draw_histogram(initial_altitude_idx, initial_azimuth_idx)

    # 7. Register the update function with the sliders
    slider_altitude.on_changed(update)
    slider_azimuth.on_changed(update)

    # 8. Display the plot
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Interactively visualize histograms from the new dataset format.")
    parser.add_argument("--dataset-root-path", required=True, help="Path to the root directory of the dataset.")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to visualize.")
    args = parser.parse_args()

    visualize_interactive(args.dataset_root_path, args.frame)
