
import numpy as np
import argparse

class HFRFrequencyIdentifier:
    def __init__(self, npz_file_path: str):
        """
        Initializes the identifier by loading a .npz file containing LiDAR data.
        """
        with np.load(npz_file_path) as data:
            self.signals = data['signals']
            self.time_resolution_ns = data['time_resolution_ns']

    def _find_peaks(self, signal: np.ndarray, threshold: float = 0.1, pulse_width_samples: int = 5) -> list[int]:
        """Finds peak start indices in a signal vector."""
        raises = np.where((signal[:-1] < threshold) & (signal[1:] >= threshold))[0] + 1
        
        if not raises.any():
            return []
            
        filtered_peaks = [raises[0]]
        for i in range(1, len(raises)):
            if (raises[i] - raises[i-1]) > pulse_width_samples:
                filtered_peaks.append(raises[i])
        return filtered_peaks

    def identify_frequency(self, threshold: float = 0.1, pulse_width_samples: int = 5) -> float:
        """
        Identifies the High Frequency Repetition (HFR) frequency from the signals.
        """
        num_frames, num_channels, num_horizontal, _ = self.signals.shape
        
        all_peak_intervals = []

        for frame in range(num_frames):
            for h_idx in range(num_horizontal):
                for v_idx in range(num_channels):
                    signal = self.signals[frame, v_idx, h_idx, :]
                    peak_indices = self._find_peaks(signal, threshold=threshold, pulse_width_samples=pulse_width_samples)

                    if len(peak_indices) > 1:
                        intervals = np.diff(peak_indices) * self.time_resolution_ns
                        all_peak_intervals.extend(intervals)

        if not all_peak_intervals:
            return 0.0

        # Simple approach: assume the most common interval corresponds to the HFR frequency.
        # A more robust approach would involve histogramming and finding the peak.
        median_interval_ns = np.median(all_peak_intervals)
        
        if median_interval_ns == 0:
            return 0.0

        # Convert interval in nanoseconds to frequency in MHz
        hfr_frequency_mhz = 1.0 / (median_interval_ns * 1e-9) / 1e6

        return hfr_frequency_mhz

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Identify HFR frequency from LiDAR signals.")
    parser.add_argument("--input-npz", type=str, help="Path to the input .npz file.")
    parser.add_argument("--threshold", type=float, default=0.1, help="Peak detection threshold.")
    parser.add_argument("--pulse-width", type=int, default=5, help="Pulse width in samples for peak filtering.")
    
    args = parser.parse_args()

    print(f"Loading data from {args.input_npz}...")
    identifier = HFRFrequencyIdentifier(args.input_npz)
    
    print("Identifying HFR frequency...")
    frequency = identifier.identify_frequency(threshold=args.threshold, pulse_width_samples=args.pulse_width)
    
    if frequency > 0:
        print(f"Identified HFR frequency: {frequency:.2f} MHz")
    else:
        print("Could not identify HFR frequency.")
    
    print("Done.")
