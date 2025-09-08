import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import argparse

def visualize_interactive(npz_file_path: str):
    """
    Loads a hist-matrix .npz file and provides an interactive plot
    with sliders to select and view individual histograms.
    """
    # 1. Load data from the .npz file
    try:
        with np.load(npz_file_path) as data:
            # Use the first frame [0] for visualization
            hist_matrix = data['signals'][1]
            vertical_angles = data['vertical_angles']
            # Try to load labels, but don't fail if they don't exist
            label_matrix = data.get('labels', [None])[0]
    except FileNotFoundError:
        print(f"Error: File not found at {npz_file_path}")
        return
    except Exception as e:
        print(f"Error loading .npz file: {e}")
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
        
        # Check if label_matrix exists and has the correct shape
        if label_matrix is not None and label_matrix.shape == hist_matrix.shape:
            labels = label_matrix[alt_idx, azi_idx, :]
            if (alt_idx == 18 and azi_idx == 1759) or (alt_idx == 5 and azi_idx == 328):
                print(labels)
            # Plot genuine signals in sky blue
            ax.fill_between(time_axis, 0, signal, where=labels == 1, 
                            color='skyblue', alpha=0.8, label='Genuine (1)')
            # Plot HFR signals in pink
            ax.fill_between(time_axis, 0, signal, where=labels == 2, 
                            color='pink', alpha=0.8, label='HFR (2)')
        
        # Plot the signal outline for clarity
        ax.plot(time_axis, signal, color='black', linewidth=0.75)
        
        ax.set_xlabel("Time Sample Index")
        ax.set_ylabel("Intensity")
        ax.set_ylim(0, 10)
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
    parser = argparse.ArgumentParser(description="Interactively visualize histograms from a .npz file.")
    parser.add_argument("--npz-file", required=True, help="Path to the .npz histogram matrix file.")
    args = parser.parse_args()

    visualize_interactive(args.npz_file)
