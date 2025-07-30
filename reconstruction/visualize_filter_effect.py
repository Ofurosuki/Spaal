
import numpy as np
import matplotlib.pyplot as plt
import argparse

def visualize(original_npz: str, filtered_npz: str, output_image_path: str):
    """
    Visualizes the effect of the filter on a sample signal.
    """
    with np.load(original_npz) as data_orig:
        signals_orig = data_orig['signals']
        time_resolution_ns = data_orig['time_resolution_ns']

    with np.load(filtered_npz) as data_filt:
        signals_filt = data_filt['signals']

    # --- Find a signal that was likely attacked to visualize the effect ---
    num_frames, num_channels, num_horizontal, num_samples = signals_orig.shape
    found_signal = False
    
    sample_to_plot_orig = None
    sample_to_plot_filt = None
    
    # Heuristic: find a signal with high frequency content (lots of peaks)
    for frame in range(num_frames):
        for h_idx in range(num_horizontal):
            for v_idx in range(num_channels):
                signal = signals_orig[frame, v_idx, h_idx, :]
                # Simple peak count as a proxy for HFR attack
                raises = np.where((signal[:-1] < 0.1) & (signal[1:] >= 0.1))[0]
                if len(raises) > 10: # Arbitrary threshold for "lots of peaks"
                    sample_to_plot_orig = signal
                    sample_to_plot_filt = signals_filt[frame, v_idx, h_idx, :]
                    found_signal = True
                    break
            if found_signal:
                break
        if found_signal:
            break

    if not found_signal:
        print("Could not find a clear example of an attacked signal to visualize.")
        # Fallback to a default signal if none is found
        sample_to_plot_orig = signals_orig[0, 0, 400, :]
        sample_to_plot_filt = signals_filt[0, 0, 400, :]

    # --- Plotting ---
    time_axis = np.arange(num_samples) * time_resolution_ns

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # Original Signal
    ax1.plot(time_axis, sample_to_plot_orig, label='Original Signal', color='blue')
    ax1.set_title('Signal Before Filtering')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)
    ax1.legend()

    # Filtered Signal
    ax2.plot(time_axis, sample_to_plot_filt, label='Filtered Signal', color='green')
    ax2.set_title('Signal After Notch Filtering')
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_image_path)
    print(f"Visualization saved to {output_image_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize the effect of the notch filter.")
    parser.add_argument("original_npz", type=str, help="Path to the original .npz file.")
    parser.add_argument("filtered_npz", type=str, help="Path to the filtered .npz file.")
    parser.add_argument("output_image", type=str, help="Path to save the output visualization image.")
    args = parser.parse_args()

    visualize(args.original_npz, args.filtered_npz, args.output_image)
