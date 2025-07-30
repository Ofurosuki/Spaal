
import numpy as np
import argparse

class PeakIntervalReconstructor:
    def __init__(self, npz_file_path: str):
        """
        Initializes the reconstructor by loading a .npz file containing LiDAR data.
        """
        with np.load(npz_file_path) as data:
            self.signals = data['signals']
            self.time_resolution_ns = data['time_resolution_ns']
            self.metadata = {k: v for k, v in data.items() if k != 'signals'}
        
        self.reconstructed_signals = np.copy(self.signals)

    def _find_peaks(self, signal: np.ndarray, threshold: float = 0.1, min_samples_between_peaks: int = 5) -> list[int]:
        """Finds peak start indices in a signal vector."""
        raises = np.where((signal[:-1] < threshold) & (signal[1:] >= threshold))[0] + 1
        if not raises.any():
            return []
        filtered_peaks = [raises[0]]
        for i in range(1, len(raises)):
            if (raises[i] - raises[i-1]) > min_samples_between_peaks:
                filtered_peaks.append(raises[i])
        return np.array(filtered_peaks)

    def reconstruct(self, hfr_freq_mhz: float, tolerance_ns: float = 1.5, min_run_length: int = 3, pulse_width_samples: int = 5):
        """
        Reconstructs the signal by identifying and removing peaks that match the HFR period.
        """
        if hfr_freq_mhz <= 0:
            print("Warning: Invalid frequency provided (<= 0). No reconstruction will be applied.")
            return

        target_period_ns = 1.0 / (hfr_freq_mhz * 1e6) * 1e9
        num_frames, num_channels, num_horizontal, _ = self.signals.shape

        for frame in range(num_frames):
            for h_idx in range(num_horizontal):
                for v_idx in range(num_channels):
                    signal = self.signals[frame, v_idx, h_idx, :]
                    peaks = self._find_peaks(signal)

                    if len(peaks) < min_run_length:
                        continue

                    intervals_ns = np.diff(peaks) * self.time_resolution_ns
                    
                    attack_peak_indices = set()
                    
                    i = 0
                    while i < len(intervals_ns):
                        if abs(intervals_ns[i] - target_period_ns) <= tolerance_ns:
                            # Found a potential start of a run
                            run_start_interval_idx = i
                            j = i + 1
                            while j < len(intervals_ns) and abs(intervals_ns[j] - target_period_ns) <= tolerance_ns:
                                j += 1
                            
                            # A run of intervals has ended at index j-1
                            run_length = j - run_start_interval_idx
                            if run_length >= min_run_length - 1:
                                # This is a confirmed attack run. Add all peaks involved.
                                for k in range(run_start_interval_idx, j + 1):
                                    attack_peak_indices.add(peaks[k])
                            i = j # Continue search from the end of the last run
                        else:
                            i += 1

                    if not attack_peak_indices:
                        continue

                    # Remove the identified attack peaks by zeroing them out
                    clean_signal = self.reconstructed_signals[frame, v_idx, h_idx, :]
                    for peak_idx in sorted(list(attack_peak_indices)):
                        start = max(0, peak_idx - 1)
                        end = min(len(clean_signal), peak_idx + pulse_width_samples)
                        clean_signal[start:end] = 0

        print(f"Peak interval reconstruction applied for {hfr_freq_mhz:.2f} MHz.")

    def save(self, output_path: str):
        """
        Saves the reconstructed signal matrix and metadata to a new .npz file.
        """
        np.savez(
            output_path,
            signals=self.reconstructed_signals,
            **self.metadata
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reconstruct LiDAR signals using the Peak Interval method.")
    parser.add_argument("input_npz", type=str, help="Path to the input .npz file.")
    parser.add_argument("output_npz", type=str, help="Path to save the reconstructed .npz file.")
    parser.add_argument("--freq", type=float, required=True, help="The HFR frequency to target in MHz.")
    parser.add_argument("--tolerance", type=float, default=1.5, help="Tolerance for period matching in nanoseconds.")
    parser.add_argument("--min-run", type=int, default=3, help="Minimum number of consecutive peaks to be considered an attack.")
    
    args = parser.parse_args()

    print(f"Loading data from {args.input_npz}...")
    reconstructor = PeakIntervalReconstructor(args.input_npz)
    
    print(f"Reconstructing signals based on peak intervals for {args.freq:.2f} MHz...")
    reconstructor.reconstruct(hfr_freq_mhz=args.freq, tolerance_ns=args.tolerance, min_run_length=args.min_run)
    
    print(f"Saving reconstructed data to {args.output_npz}...")
    reconstructor.save(args.output_npz)
    
    print("Done.")
