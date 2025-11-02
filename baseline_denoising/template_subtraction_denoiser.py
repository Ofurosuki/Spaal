import numpy as np
import os
import blosc2
import argparse
import json
from typing import Dict, List
from tqdm import tqdm
import shutil

class TemplateSubtractionDenoiser:
    def __init__(self, input_dir: str, output_dir: str, min_template_samples: int = 2, min_peak_threshold: float = 0.01, chunk_size: int = 20):
        """
        Initialize template subtraction denoiser.

        Parameters:
        -----------
        input_dir : str
            Directory containing sample subdirectories with signal.bl2 and timestamps.bl2
        output_dir : str
            Directory to save denoised results
        min_template_samples : int
            Minimum number of samples required to create a template for a timestamp
        min_peak_threshold : float
            Minimum peak intensity for denoised signal. If max(denoised_signal) < threshold, set signal to 0
        chunk_size : int
            Number of samples to process at once to reduce memory usage (default: 20)
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.min_template_samples = min_template_samples
        self.min_peak_threshold = min_peak_threshold
        self.chunk_size = chunk_size

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Find all sample subdirectories
        self.sample_dirs = self._find_sample_dirs()

        if not self.sample_dirs:
            raise ValueError(f"No sample directories found in {input_dir}")

        print(f"Found {len(self.sample_dirs)} sample directories")

    def _find_sample_dirs(self) -> List[str]:
        """Find all subdirectories containing signal.bl2 and timestamps.bl2"""
        sample_dirs = []
        for subdir in sorted(os.listdir(self.input_dir)):
            subdir_path = os.path.join(self.input_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue

            signal_path = os.path.join(subdir_path, 'signal.bl2')
            timestamps_path = os.path.join(subdir_path, 'timestamps.bl2')

            if os.path.exists(signal_path) and os.path.exists(timestamps_path):
                sample_dirs.append(subdir)

        return sample_dirs

    def _load_bl2(self, file_path: str) -> np.ndarray:
        """Load a blosc2 compressed file"""
        with open(file_path, 'rb') as f:
            return blosc2.unpack_array(f.read())

    def _save_bl2(self, file_path: str, data: np.ndarray):
        """Save data to blosc2 compressed file"""
        packed_data = blosc2.pack_array(data)
        with open(file_path, 'wb') as f:
            f.write(packed_data)

    def build_templates(self) -> Dict[int, np.ndarray]:
        """
        Build templates by averaging signals with the same timestamp using chunked processing.

        Returns:
        --------
        Dict[int, np.ndarray]
            Dictionary mapping timestamp (ns) to average signal template
        """
        print(f"\nBuilding templates from {len(self.sample_dirs)} samples in chunks of {self.chunk_size}...")

        # Running statistics: timestamp -> (sum, count)
        timestamp_stats: Dict[int, tuple[np.ndarray, int]] = {}

        # Process samples in chunks
        num_chunks = (len(self.sample_dirs) + self.chunk_size - 1) // self.chunk_size

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, len(self.sample_dirs))
            chunk_samples = self.sample_dirs[start_idx:end_idx]

            print(f"\nProcessing chunk {chunk_idx + 1}/{num_chunks} (samples {start_idx}-{end_idx-1})...")

            # Dictionary for this chunk: timestamp -> list of signals
            chunk_timestamp_to_signals: Dict[int, List[np.ndarray]] = {}

            # Collect signals from this chunk
            for sample_name in tqdm(chunk_samples, desc=f"Chunk {chunk_idx + 1} - Collecting"):
                sample_dir = os.path.join(self.input_dir, sample_name)

                # Load timestamps and signals
                timestamps = self._load_bl2(os.path.join(sample_dir, 'timestamps.bl2'))
                signals = self._load_bl2(os.path.join(sample_dir, 'signal.bl2'))

                if signals.ndim != 3:
                    print(f"Warning: {sample_name} has unexpected signal shape {signals.shape}, skipping")
                    continue

                channels, horizontal_resolution, num_samples = signals.shape

                # Group signals by timestamp
                for ch_idx in range(channels):
                    for h_idx in range(horizontal_resolution):
                        timestamp = int(timestamps[ch_idx, h_idx])
                        signal = signals[ch_idx, h_idx, :]

                        # Only include signals with non-zero energy
                        if np.sum(signal) > 0.01:
                            if timestamp not in chunk_timestamp_to_signals:
                                chunk_timestamp_to_signals[timestamp] = []
                            chunk_timestamp_to_signals[timestamp].append(signal)

            # Update running statistics with this chunk's data
            for timestamp, signal_list in tqdm(chunk_timestamp_to_signals.items(), desc=f"Chunk {chunk_idx + 1} - Updating stats"):
                chunk_sum = np.sum(signal_list, axis=0)
                chunk_count = len(signal_list)

                if timestamp not in timestamp_stats:
                    # First time seeing this timestamp
                    timestamp_stats[timestamp] = (chunk_sum, chunk_count)
                else:
                    # Update running sum and count
                    old_sum, old_count = timestamp_stats[timestamp]
                    timestamp_stats[timestamp] = (old_sum + chunk_sum, old_count + chunk_count)

            # Clear chunk data to free memory
            chunk_timestamp_to_signals.clear()

        print(f"\nFound {len(timestamp_stats)} unique timestamps")

        # Calculate final templates from accumulated statistics
        templates: Dict[int, np.ndarray] = {}
        timestamps_with_enough_samples = 0

        for timestamp, (signal_sum, signal_count) in tqdm(timestamp_stats.items(), desc="Computing templates"):
            if signal_count >= self.min_template_samples:
                # Calculate average from sum and count
                templates[timestamp] = signal_sum / signal_count
                timestamps_with_enough_samples += 1

        print(f"Created {len(templates)} templates (required min {self.min_template_samples} samples)")
        print(f"  - {timestamps_with_enough_samples} timestamps had enough samples")
        print(f"  - {len(timestamp_stats) - timestamps_with_enough_samples} timestamps had too few samples")

        return templates

    def denoise_samples(self, templates: Dict[int, np.ndarray]):
        """
        Denoise all samples using templates.

        Parameters:
        -----------
        templates : Dict[int, np.ndarray]
            Dictionary mapping timestamp to template signal
        """
        print("\nDenoising samples...")

        for sample_name in tqdm(self.sample_dirs, desc="Processing samples"):
            sample_input_dir = os.path.join(self.input_dir, sample_name)
            sample_output_dir = os.path.join(self.output_dir, sample_name)
            os.makedirs(sample_output_dir, exist_ok=True)

            # Load timestamps and signals
            timestamps = self._load_bl2(os.path.join(sample_input_dir, 'timestamps.bl2'))
            signals = self._load_bl2(os.path.join(sample_input_dir, 'signal.bl2'))

            if signals.ndim != 3:
                print(f"Warning: {sample_name} has unexpected signal shape {signals.shape}, skipping")
                continue

            channels, horizontal_resolution, num_samples = signals.shape
            denoised_signals = np.copy(signals)

            subtracted_count = 0
            no_template_count = 0
            zeroed_count = 0

            # Subtract template from each signal
            for ch_idx in range(channels):
                for h_idx in range(horizontal_resolution):
                    timestamp = int(timestamps[ch_idx, h_idx])

                    if timestamp in templates:
                        # Subtract template and clip to non-negative values
                        denoised_signal = np.maximum(
                            signals[ch_idx, h_idx, :] - templates[timestamp],
                            0.0
                        )

                        # If max intensity is below threshold, treat as noise and zero out
                        if np.max(denoised_signal) < self.min_peak_threshold:
                            denoised_signals[ch_idx, h_idx, :] = 0.0
                            zeroed_count += 1
                        else:
                            denoised_signals[ch_idx, h_idx, :] = denoised_signal

                        subtracted_count += 1
                    else:
                        # No template available, keep original signal
                        no_template_count += 1

            # Save denoised signal
            self._save_bl2(os.path.join(sample_output_dir, 'signal.bl2'), denoised_signals)

            # Copy other files (config.json, labels.bl2, etc.)
            for filename in ['config.json', 'labels.bl2', 'answer_matrix.bl2', 'angles.bl2', 'timestamps.bl2']:
                src = os.path.join(sample_input_dir, filename)
                dst = os.path.join(sample_output_dir, filename)
                if os.path.exists(src):
                    shutil.copy2(src, dst)

            # Debug info for first sample
            if sample_name == self.sample_dirs[0]:
                print(f"\nFirst sample ({sample_name}):")
                print(f"  - Subtracted template from {subtracted_count} pixels")
                print(f"  - Zeroed out (below threshold) {zeroed_count} pixels")
                print(f"  - No template available for {no_template_count} pixels")
                print(f"  - Total pixels: {channels * horizontal_resolution}")

    def run(self):
        """Run the complete denoising pipeline"""
        print("="*70)
        print("Template Subtraction Denoising")
        print("="*70)
        print(f"Input directory:  {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Min template samples: {self.min_template_samples}")
        print(f"Min peak threshold: {self.min_peak_threshold}")
        print(f"Chunk size: {self.chunk_size} samples")
        print("="*70)

        # Step 1: Build templates
        templates = self.build_templates()

        if not templates:
            print("Error: No templates could be created. Check your data.")
            return

        # Step 2: Denoise all samples
        self.denoise_samples(templates)

        # Save template statistics
        stats = {
            'num_templates': len(templates),
            'min_template_samples': self.min_template_samples,
            'min_peak_threshold': self.min_peak_threshold,
            'chunk_size': self.chunk_size,
            'num_samples_processed': len(self.sample_dirs),
            'timestamps': sorted(list(templates.keys()))
        }

        with open(os.path.join(self.output_dir, 'denoising_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)

        print("\n" + "="*70)
        print("Denoising complete!")
        print(f"Denoised samples saved to: {self.output_dir}")
        print(f"Statistics saved to: {os.path.join(self.output_dir, 'denoising_stats.json')}")
        print("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Denoise LiDAR signals using template subtraction based on timestamps."
    )
    parser.add_argument("--input-dir", required=True, type=str,
                        help="Input directory containing sample subdirectories with signal.bl2 and timestamps.bl2")
    parser.add_argument("--output-dir", required=True, type=str,
                        help="Output directory for denoised samples")
    parser.add_argument("--min-template-samples", type=int, default=2,
                        help="Minimum number of samples required to create a template for a timestamp (default: 2)")
    parser.add_argument("--min-peak-threshold", type=float, default=-1.0,
                        help="Minimum peak intensity for denoised signal. If max(denoised_signal) < threshold, set signal to 0 (default: -1.0)")
    parser.add_argument("--chunk-size", type=int, default=20,
                        help="Number of samples to process at once to reduce memory usage (default: 20)")

    args = parser.parse_args()

    try:
        denoiser = TemplateSubtractionDenoiser(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            min_template_samples=args.min_template_samples,
            min_peak_threshold=args.min_peak_threshold,
            chunk_size=args.chunk_size
        )
        denoiser.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
