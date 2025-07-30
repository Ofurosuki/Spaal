
import numpy as np
import argparse

class NotchFilter:
    def __init__(self, npz_file_path: str):
        """
        Initializes the filter by loading a .npz file containing LiDAR data.
        """
        with np.load(npz_file_path) as data:
            self.signals = data['signals']
            self.time_resolution_ns = data['time_resolution_ns']
            # Keep other metadata for saving
            self.metadata = {k: v for k, v in data.items() if k != 'signals'}
        
        self.filtered_signals = np.copy(self.signals)

    def apply_filter(self, hfr_freq_mhz: float, bandwidth_mhz: float = 0.5, harmonics: int = 3):
        """
        Applies a notch filter to remove the specified frequency and its harmonics.
        """
        if hfr_freq_mhz <= 0:
            print("Warning: Invalid frequency provided (<= 0). No filtering will be applied.")
            return

        num_frames, num_channels, num_horizontal, num_samples = self.signals.shape
        sample_spacing = self.time_resolution_ns * 1e-9  # in seconds
        
        # Frequency resolution of the FFT
        freqs = np.fft.fftfreq(num_samples, d=sample_spacing)
        
        # --- Define frequencies to notch out ---
        frequencies_to_remove = []
        for i in range(1, harmonics + 1):
            center_freq = hfr_freq_mhz * 1e6 * i  # Convert MHz to Hz
            half_bandwidth = (bandwidth_mhz * 1e6) / 2
            frequencies_to_remove.append((center_freq - half_bandwidth, center_freq + half_bandwidth))

        for frame in range(num_frames):
            for h_idx in range(num_horizontal):
                for v_idx in range(num_channels):
                    signal = self.signals[frame, v_idx, h_idx, :]
                    
                    # --- Apply Notch Filter ---
                    # 1. Go to frequency domain
                    fft_vals = np.fft.fft(signal)
                    
                    # 2. Find and zero out the coefficients for the target frequencies
                    for min_freq, max_freq in frequencies_to_remove:
                        # Positive and negative frequencies
                        indices_to_zero = np.where((np.abs(freqs) >= min_freq) & (np.abs(freqs) <= max_freq))
                        fft_vals[indices_to_zero] = 0
                    
                    # 3. Go back to time domain
                    filtered_signal = np.fft.ifft(fft_vals)
                    self.filtered_signals[frame, v_idx, h_idx, :] = np.real(filtered_signal)

        print(f"Notch filter applied at {hfr_freq_mhz:.2f} MHz with {harmonics} harmonics.")

    def save(self, output_path: str):
        """
        Saves the filtered signal matrix and metadata to a new .npz file.
        """
        np.savez(
            output_path,
            signals=self.filtered_signals,
            **self.metadata
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Apply a notch filter to LiDAR signals to remove HFR noise.")
    parser.add_argument("input_npz", type=str, help="Path to the input .npz file.")
    parser.add_argument("output_npz", type=str, help="Path to save the filtered .npz file.")
    parser.add_argument("--freq", type=float, required=True, help="The HFR frequency to remove in MHz.")
    parser.add_argument("--bw", type=float, default=0.5, help="Bandwidth of the notch filter in MHz.")
    parser.add_argument("--harmonics", type=int, default=3, help="Number of harmonics to remove.")

    args = parser.parse_args()

    print(f"Loading data from {args.input_npz}...")
    notch_filter = NotchFilter(args.input_npz)
    
    print(f"Applying notch filter to remove {args.freq:.2f} MHz...")
    notch_filter.apply_filter(hfr_freq_mhz=args.freq, bandwidth_mhz=args.bw, harmonics=args.harmonics)
    
    print(f"Saving filtered data to {args.output_npz}...")
    notch_filter.save(args.output_npz)
    
    print("Done.")
