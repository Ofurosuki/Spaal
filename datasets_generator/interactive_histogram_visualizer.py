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
            hist_matrix = data['signals'][0]
            vertical_angles = data['vertical_angles']
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

    # Plot the initial histogram
    line, = ax.plot(time_axis, hist_matrix[initial_altitude_idx, initial_azimuth_idx, :])
    ax.set_xlabel("Time Sample Index")
    ax.set_ylabel("Intensity")
    ax.set_ylim(0, 10)  # Assuming intensity is clipped at 9
    ax.grid(True)

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
        valstep=1  # Force integer steps
    )

    slider_azimuth = Slider(
        ax=ax_azimuth,
        label='Azimuth Index',
        valmin=0,
        valmax=num_horizontal_steps - 1,
        valinit=initial_azimuth_idx,
        valstep=1  # Force integer steps
    )

    # 6. Define the function to be called when a slider value changes
    def update(val):
        alt_idx = int(slider_altitude.val)
        azi_idx = int(slider_azimuth.val)

        # Update the y-data of the plot with the new histogram
        line.set_ydata(hist_matrix[alt_idx, azi_idx, :])

        # Update the title to show current indices and angle
        altitude_deg = vertical_angles[alt_idx]
        fig.suptitle(f'Altitude Idx: {alt_idx} (~{altitude_deg:.2f} deg), Azimuth Idx: {azi_idx}')

        # Redraw the plot
        fig.canvas.draw_idle()

    # Initialize title
    update(None)

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
