
import numpy as np
import argparse

class RuleBasedReconstructor:
    def __init__(self, npz_file_path: str):
        """
        Initializes the reconstructor by loading a .npz file containing LiDAR data and metadata.
        """
        with np.load(npz_file_path) as data:
            self.signals = data['signals']
            self.initial_azimuth_offset = data['initial_azimuth_offset']
            self.vertical_angles = data['vertical_angles']
            self.fov = data['fov']
            self.time_resolution_ns = data['time_resolution_ns']
        
        self.reconstructed_signals = np.zeros_like(self.signals)
        self._precompute_vertical_neighbors()

    def _precompute_vertical_neighbors(self):
        """
        Precomputes the index of the nearest vertical neighbor for each channel based on angle.
        """
        self._vertical_neighbors = {}
        angles = self.vertical_angles
        for i, angle in enumerate(angles):
            diffs = [abs(angle - other_angle) for other_angle in angles]
            # Find the index of the smallest non-zero difference
            sorted_indices = np.argsort(diffs)
            for j in sorted_indices:
                if i != j:
                    self._vertical_neighbors[i] = j
                    break

    def _distance_from_time(self, time_index: int) -> float:
        """Converts a time index (sample) to a distance in meters."""
        return (time_index * self.time_resolution_ns) * 0.15

    def _find_peaks(self, signal: np.ndarray, threshold: float = 0.1, pulse_width_samples: int = 5) -> list[int]:
        """Finds peak start indices in a signal vector."""
        # Find where the signal crosses the threshold upwards
        raises = np.where((signal[:-1] < threshold) & (signal[1:] >= threshold))[0] + 1
        
        # Filter out peaks that are too close to each other (debounce)
        if not raises.any():
            return []
            
        filtered_peaks = [raises[0]]
        for i in range(1, len(raises)):
            if (raises[i] - raises[i-1]) > pulse_width_samples:
                filtered_peaks.append(raises[i])
        return filtered_peaks

    def reconstruct(self, continuity_threshold_m: float = 1.5, pulse_width_samples: int = 5):
        """
        Performs the rule-based reconstruction of the LiDAR signals.
        """
        num_frames, num_channels, num_horizontal, _ = self.signals.shape
        
        for frame in range(num_frames):
            adopted_distances = np.full((num_channels, num_horizontal), np.nan)
            for h_idx in range(num_horizontal):
                for v_idx in range(num_channels):
                    signal = self.signals[frame, v_idx, h_idx, :]
                    peak_indices = self._find_peaks(signal, pulse_width_samples=pulse_width_samples)

                    if not peak_indices:
                        continue

                    true_peak_index = -1
                    if len(peak_indices) == 1:
                        true_peak_index = peak_indices[0]
                    else:
                        # --- 2D Spatio-temporal Filtering with Weighting ---
                        peak_distances = [self._distance_from_time(p) for p in peak_indices]
                        scores = []

                        # Get neighbor distances
                        h_neighbor_dist = adopted_distances[v_idx, h_idx - 1] if h_idx > 0 else np.nan
                        
                        v_neighbor_idx = self._vertical_neighbors.get(v_idx)
                        v_neighbor_dist = adopted_distances[v_neighbor_idx, h_idx] if v_neighbor_idx is not None else np.nan

                        # Calculate angle deltas for weighting
                        delta_azimuth = self.fov / num_horizontal
                        delta_altitude = abs(self.vertical_angles[v_idx] - self.vertical_angles[v_neighbor_idx]) if v_neighbor_idx is not None else 180

                        for dist in peak_distances:
                            score = 0
                            if not np.isnan(h_neighbor_dist):
                                score += (1 / delta_azimuth) * abs(dist - h_neighbor_dist)
                            if not np.isnan(v_neighbor_dist):
                                score += (1 / delta_altitude) * abs(dist - v_neighbor_dist)
                            scores.append(score)
                        
                        # Fallback to first peak if no neighbors were found
                        if not scores or all(s == 0 for s in scores):
                            true_peak_index = peak_indices[0]
                        else:
                            best_peak_idx_in_list = np.argmin(scores)
                            true_peak_index = peak_indices[best_peak_idx_in_list]

                    # Record and reconstruct the chosen peak
                    if true_peak_index != -1:
                        adopted_distances[v_idx, h_idx] = self._distance_from_time(true_peak_index)
                        
                        # Copy the pulse shape to the reconstructed signal
                        start = true_peak_index
                        end = min(len(signal), start + pulse_width_samples * 2) # capture the whole pulse shape
                        self.reconstructed_signals[frame, v_idx, h_idx, start:end] = signal[start:end]

        return self.reconstructed_signals

    def save(self, output_path: str):
        """
        Saves the reconstructed signal matrix and metadata to a new .npz file.
        """
        metadata_to_save = {
            'initial_azimuth_offset': self.initial_azimuth_offset,
            'vertical_angles': self.vertical_angles,
            'fov': self.fov,
            'time_resolution_ns': self.time_resolution_ns
        }
        np.savez(
            output_path,
            signals=self.reconstructed_signals,
            **metadata_to_save
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reconstruct LiDAR signals by filtering noise.")
    parser.add_argument("input_npz", type=str, help="Path to the input .npz file.")
    parser.add_argument("output_npz", type=str, help="Path to save the reconstructed .npz file.")
    parser.add_argument("--threshold", type=float, default=1.5, help="Continuity threshold in meters.")
    
    args = parser.parse_args()

    print(f"Loading data from {args.input_npz}...")
    reconstructor = RuleBasedReconstructor(args.input_npz)
    
    print("Reconstructing signals...")
    reconstructor.reconstruct(continuity_threshold_m=args.threshold)
    
    print(f"Saving reconstructed data to {args.output_npz}...")
    reconstructor.save(args.output_npz)
    
    print("Done.")
