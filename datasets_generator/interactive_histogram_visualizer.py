import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import argparse
import os
import json
import blosc2

def visualize_interactive(dataset_root_path: str, frame_specifier: str):
    """
    Loads frames from the new dataset format and provides an interactive plot
    with sliders to select and view individual histograms.
    Can visualize a single frame or an overlay of a range of frames.
    """
    # 1. Parse frame_specifier to get frame indices
    if '-' in frame_specifier:
        try:
            start, end = map(int, frame_specifier.split('-'))
            if start > end:
                print(f"Invalid range: start index {start} is greater than end index {end}.")
                return
            frame_indices = list(range(start, end + 1))
        except ValueError:
            print(f"Invalid frame range format: {frame_specifier}. Expected 'start-end'.")
            return
    else:
        try:
            frame_indices = [int(frame_specifier)]
        except ValueError:
            print(f"Invalid frame index: {frame_specifier}.")
            return

    # 2. Load data from the dataset directory structure
    hist_matrices = []
    label_matrices = []
    vertical_angles = None

    try:
        sample_dirs = sorted([d for d in os.listdir(dataset_root_path) if os.path.isdir(os.path.join(dataset_root_path, d))])
        if not sample_dirs:
            raise FileNotFoundError(f"No sample directories found in {dataset_root_path}")

        for frame_index in frame_indices:
            if frame_index >= len(sample_dirs):
                raise ValueError(f"Frame index {frame_index} is out of bounds for {len(sample_dirs)} sample directories.")
            
            sample_dir = os.path.join(dataset_root_path, sample_dirs[frame_index])
            print(f"Loading data from {sample_dir}")

            if vertical_angles is None: # Load config only for the first frame
                with open(os.path.join(sample_dir, 'config.json'), 'r') as f:
                    config_data = json.load(f)
                vertical_angles = config_data['vertical_angles']

            with open(os.path.join(sample_dir, 'signal.bl2'), 'rb') as f:
                hist_matrices.append(blosc2.unpack_array(f.read()))

            labels_path = os.path.join(sample_dir, 'labels.bl2')
            if os.path.exists(labels_path):
                with open(labels_path, 'rb') as f:
                    label_matrices.append(blosc2.unpack_array(f.read()))
            else:
                label_matrices.append(None)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 3. Get dimensions from the loaded data (using the first frame)
    num_channels, num_horizontal_steps, num_samples = hist_matrices[0].shape

    # 4. Setup the initial Matplotlib plot
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.3)  # Make room for sliders

    # Initial indices for the first plot
    initial_altitude_idx = 0
    initial_azimuth_idx = 0

    # Create x-axis for the histogram plot (representing time samples)
    time_axis = np.arange(num_samples)

    def draw_histogram(alt_idx, azi_idx):
        """Clears the axes and redraws the histogram(s) with colored labels if applicable."""
        ax.clear()
        
        # Plot signals from all selected frames
        for i, hist_matrix in enumerate(hist_matrices):
            signal = hist_matrix[alt_idx, azi_idx, :]
            ax.plot(time_axis, signal, linewidth=0.75, label=f"Frame {frame_indices[i]}")

        # If only one frame is visualized, show labels
        if len(hist_matrices) == 1 and label_matrices[0] is not None:
            label_matrix = label_matrices[0]
            # Check if label_matrix has the correct shape
            if label_matrix.shape == hist_matrices[0].shape:
                labels = label_matrix[alt_idx, azi_idx, :]
                bar_height = 10  # A small height for the color bar on the x-axis

                # Draw a colored bar on the x-axis for genuine signals
                ax.fill_between(time_axis, 0, bar_height, where=labels == 1, 
                                color='skyblue', alpha=0.8, label='Genuine (1)')
                # Draw a colored bar on the x-axis for HFR signals
                ax.fill_between(time_axis, 0, bar_height, where=labels == 2, 
                                color='pink', alpha=0.8, label='HFR (2)')
        
        bar_height = 10
        ax.set_xlabel("Time Sample Index")
        ax.set_ylabel("Intensity")
        ax.set_ylim(0, bar_height)
        ax.grid(True)
        ax.legend(loc='upper right')
        
        # Update the title to show current indices and angle
        altitude_deg = vertical_angles[alt_idx]
        fig.suptitle(f'Altitude Idx: {alt_idx} (~{altitude_deg:.2f} deg), Azimuth Idx: {azi_idx}')

    # 5. Create axes for the sliders
    ax_altitude = plt.axes([0.25, 0.15, 0.65, 0.03])
    ax_azimuth = plt.axes([0.25, 0.1, 0.65, 0.03])

    # 6. Create the sliders for altitude and azimuth
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

    # 7. Define the update function to be called by sliders
    def update(val):
        alt_idx = int(slider_altitude.val)
        azi_idx = int(slider_azimuth.val)
        draw_histogram(alt_idx, azi_idx)
        fig.canvas.draw_idle()

    # Initial draw
    draw_histogram(initial_altitude_idx, initial_azimuth_idx)

    # 8. Register the update function with the sliders
    slider_altitude.on_changed(update)
    slider_azimuth.on_changed(update)

    # 9. Display the plot
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Interactively visualize histograms from the new dataset format.")
    parser.add_argument("--dataset-root-path", required=True, help="Path to the root directory of the dataset.")
    parser.add_argument("--frame", type=str, default='0', help="Frame index or range (e.g., '0' or '0-4') to visualize.")
    args = parser.parse_args()

    visualize_interactive(args.dataset_root_path, args.frame)