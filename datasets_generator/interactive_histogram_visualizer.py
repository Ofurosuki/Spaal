import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
import argparse
import os
import json
import blosc2

def visualize_interactive(dataset_root_path: str, frame_specifier: str):
    """
    Loads frames from the new dataset format and provides an interactive plot
    with sliders and checkboxes to select and view individual histograms.
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
    timestamp_matrices = []
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

            timestamps_path = os.path.join(sample_dir, 'timestamps.bl2')
            if os.path.exists(timestamps_path):
                with open(timestamps_path, 'rb') as f:
                    timestamp_matrices.append(blosc2.unpack_array(f.read()))
            else:
                timestamp_matrices.append(None)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 3. Get dimensions from the loaded data (using the first frame)
    num_channels, num_horizontal_steps, num_samples = hist_matrices[0].shape

    # 4. Setup the initial Matplotlib plot
    fig = plt.figure(figsize=(14, 8))
    ax = plt.subplot2grid((1, 3), (0, 0), colspan=2)
    ax_text = plt.subplot2grid((1, 3), (0, 2))
    ax_text.axis('off')
    plt.subplots_adjust(left=0.05, bottom=0.25, right=0.98, top=0.92)

    # Initial indices for the first plot
    initial_altitude_idx = 0
    initial_azimuth_idx = 0

    # Create x-axis for the histogram plot (representing time samples)
    time_axis = np.arange(num_samples)

    # 5. Create widget axes
    ax_altitude = plt.axes([0.15, 0.15, 0.5, 0.03])
    ax_azimuth = plt.axes([0.15, 0.1, 0.5, 0.03])

    check_buttons = None
    if len(frame_indices) > 1:
        ax_check = plt.axes([0.02, 0.4, 0.08, 0.5])
        check_labels = [f'Frame {i}' for i in frame_indices]
        check_actives = [True] * len(frame_indices)
        check_buttons = CheckButtons(ax_check, check_labels, check_actives)

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

    # 7. Define the main drawing function
    def draw_histogram(alt_idx, azi_idx):
        """Clears the axes and redraws the histogram(s) based on widget states."""
        ax.clear()
        ax_text.clear()
        ax_text.axis('off')

        visibles = [True] * len(hist_matrices)
        if check_buttons:
            visibles = check_buttons.get_status()

        # Plot signals for visible frames
        for i, hist_matrix in enumerate(hist_matrices):
            if visibles[i]:
                signal = hist_matrix[alt_idx, azi_idx, :]
                ax.plot(time_axis, signal, linewidth=0.75, label=f"Frame {frame_indices[i]}")

        # If exactly one frame is visible, show its labels
        if sum(visibles) == 1:
            visible_idx = visibles.index(True)
            label_matrix = label_matrices[visible_idx]
            if label_matrix is not None and label_matrix.shape == hist_matrices[visible_idx].shape:
                labels = label_matrix[alt_idx, azi_idx, :]
                bar_height = 10
                ax.fill_between(time_axis, 0, bar_height, where=labels == 1,
                                color='skyblue', alpha=0.8, label='Genuine (1)')
                ax.fill_between(time_axis, 0, bar_height, where=labels == 2,
                                color='pink', alpha=0.8, label='HFR (2)')

        bar_height = 10
        ax.set_xlabel("Time Sample Index")
        ax.set_ylabel("Intensity")
        ax.set_ylim(0, bar_height)
        ax.grid(True)
        ax.legend(loc='upper right')

        altitude_deg = vertical_angles[alt_idx]
        title = f'Altitude Idx: {alt_idx} (~{altitude_deg:.2f} deg), Azimuth Idx: {azi_idx}'
        fig.suptitle(title)

        # Display detailed timestamp info in text area
        if any(tm is not None for tm in timestamp_matrices):
            text_lines = ["Timestamp Info\n" + "="*30 + "\n"]

            # Show timestamps for current position
            text_lines.append(f"Current (Alt {alt_idx}, Azi {azi_idx}):\n")
            for i, timestamp_matrix in enumerate(timestamp_matrices):
                if timestamp_matrix is not None and visibles[i]:
                    ts = timestamp_matrix[alt_idx, azi_idx]
                    text_lines.append(f"  Frame {frame_indices[i]}: {ts:,} ns\n")

            # Show timestamps for nearby altitudes (same azimuth)
            text_lines.append(f"\nNearby Altitudes (Azi {azi_idx}):\n")
            alt_range = range(max(0, alt_idx - 3), min(num_channels, alt_idx + 4))
            for nearby_alt in alt_range:
                text_lines.append(f"  Alt {nearby_alt}:")
                for i, timestamp_matrix in enumerate(timestamp_matrices):
                    if timestamp_matrix is not None and visibles[i]:
                        ts = timestamp_matrix[nearby_alt, azi_idx]
                        if nearby_alt == alt_idx:
                            text_lines.append(f" [{ts:,}]")
                        else:
                            text_lines.append(f" {ts:,}")
                text_lines.append("\n")

            ax_text.text(0.05, 0.95, ''.join(text_lines),
                        verticalalignment='top',
                        fontfamily='monospace',
                        fontsize=8,
                        transform=ax_text.transAxes)

    # 8. Define the single update function for all widgets
    def update(val):
        alt_idx = int(slider_altitude.val)
        azi_idx = int(slider_azimuth.val)
        draw_histogram(alt_idx, azi_idx)
        fig.canvas.draw_idle()

    # 9. Register the update function with the widgets
    slider_altitude.on_changed(update)
    slider_azimuth.on_changed(update)
    if check_buttons:
        check_buttons.on_clicked(update)

    # 10. Initial draw and display
    draw_histogram(initial_altitude_idx, initial_azimuth_idx)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Interactively visualize histograms from the new dataset format.")
    parser.add_argument("--dataset-root-path", required=True, help="Path to the root directory of the dataset.")
    parser.add_argument("--frame", type=str, default='0', help="Frame index or range (e.g., '0' or '0-4') to visualize.")
    args = parser.parse_args()

    visualize_interactive(args.dataset_root_path, args.frame)
