
import numpy as np
import argparse

class HFRFrequencyIdentifierFourier:
    def __init__(self, npz_file_path: str):
        """
        Initializes the identifier by loading a .npz file containing LiDAR data.
        """
        with np.load(npz_file_path) as data:
            self.signals = data['signals']
            self.time_resolution_ns = data['time_resolution_ns']

    def _find_peaks(self, signal: np.ndarray, threshold: float = 0.1, min_samples_between_peaks: int = 5) -> list[int]:
        """Finds peak start indices in a signal vector to identify potential attack signals."""
        # Find where the signal crosses the threshold upwards
        raises = np.where((signal[:-1] < threshold) & (signal[1:] >= threshold))[0] + 1
        
        if not raises.any():
            return []
            
        # Filter out peaks that are too close to each other (debounce)
        filtered_peaks = [raises[0]]
        for i in range(1, len(raises)):
            if (raises[i] - raises[i-1]) > min_samples_between_peaks:
                filtered_peaks.append(raises[i])
        return filtered_peaks

    def identify_frequency_fourier(self, peak_detection_threshold: float = 0.1, peak_count_threshold: int = 3) -> float:
        """
        Identifies the HFR frequency using Fourier Transform on potentially attacked signals.
        """
        num_frames, num_channels, num_horizontal, num_samples = self.signals.shape
        
        detected_frequencies = []

        # Sample spacing for FFT
        sample_spacing = self.time_resolution_ns * 1e-9  # in seconds

        for frame in range(num_frames):
            for h_idx in range(num_horizontal):
                for v_idx in range(num_channels):
                    signal = self.signals[frame, v_idx, h_idx, :]
                    
                    # --- Heuristic to identify potentially attacked signals ---
                    # Only perform FFT on signals that have an unusually high number of peaks.
                    peaks = self._find_peaks(signal, threshold=peak_detection_threshold)
                    if len(peaks) < peak_count_threshold:
                        continue

                    # --- Perform FFT ---
                    # We only need the magnitude of the FFT
                    fft_vals = np.fft.fft(signal)
                    fft_mag = np.abs(fft_vals)
                    
                    # Frequencies corresponding to the FFT values
                    freqs = np.fft.fftfreq(num_samples, d=sample_spacing)
                    
                    # We only care about the positive frequencies
                    positive_freq_indices = np.where(freqs > 0)
                    freqs = freqs[positive_freq_indices]
                    fft_mag = fft_mag[positive_freq_indices]

                    if len(freqs) == 0:
                        continue

                    # Find the frequency with the highest magnitude (ignoring DC component)
                    peak_frequency_index = np.argmax(fft_mag)
                    dominant_frequency_hz = freqs[peak_frequency_index]
                    
                    detected_frequencies.append(dominant_frequency_hz)

        if not detected_frequencies:
            return 0.0

        # Return the median of the detected dominant frequencies for robustness
        median_frequency_hz = np.median(detected_frequencies)
        
        return median_frequency_hz / 1e6  # Convert to MHz

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Identify HFR frequency from LiDAR signals using Fourier Transform.")
    parser.add_argument("--input-npz", type=str, help="Path to the input .npz file.")
    parser.add_argument("--threshold", type=float, default=0.1, help="Peak detection threshold to identify candidate signals.")
    parser.add_argument("--peak-count", type=int, default=3, help="Minimum number of peaks to consider a signal for FFT analysis.")

    args = parser.parse_args()

    print(f"Loading data from {args.input_npz}...")
    identifier = HFRFrequencyIdentifierFourier(args.input_npz)
    
    print("Identifying HFR frequency using Fourier Transform...")
    frequency = identifier.identify_frequency_fourier(peak_detection_threshold=args.threshold, peak_count_threshold=args.peak_count)
    
    if frequency > 0:
        print(f"Identified HFR frequency: {frequency:.2f} MHz")
    else:
        print("Could not identify HFR frequency. The signal may not contain a clear HFR attack.")
    
    print("Done.")
