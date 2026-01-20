import numpy as np
import argparse
import os
import glob
from typing import Tuple
import json
import blosc2
from tqdm import tqdm

def get_peak_time_and_amplitude(signal: np.ndarray, min_peak_amplitude: float = 0.01) -> Tuple[float, float]:
    """
    Finds the interpolated time and amplitude of the highest peak in a signal.
    Returns (0.0, 0.0) if no peak is found.

    Parameters:
    -----------
    signal : np.ndarray
        Signal array
    min_peak_amplitude : float
        Minimum amplitude threshold for peak detection (default: 0.01)
    """
    raises = np.flatnonzero((signal[:-1] < 0.01) & (signal[1:] >= 0.01)) + 1
    if len(raises) == 0:
        return 0.0, 0.0

    peak_values = np.array([np.max(signal[r:min(len(signal), r + 50)]) for r in raises])
    if len(peak_values) == 0:
        return 0.0, 0.0

    peak_amplitude = np.max(peak_values)
    if peak_amplitude < min_peak_amplitude:
        return 0.0, 0.0

    highest_pulse_start_index = raises[np.argmax(peak_values)]
    pulse_region = signal[highest_pulse_start_index:min(len(signal), highest_pulse_start_index + 50)]

    if len(pulse_region) == 0:
        return 0.0, 0.0

    peak_idx_in_region = np.argmax(pulse_region)
    peak_idx_global = highest_pulse_start_index + peak_idx_in_region

    interpolated_time = float(peak_idx_global)
    if 0 < peak_idx_global < len(signal) - 1:
        y0, y1, y2 = signal[peak_idx_global - 1:peak_idx_global + 2]
        if y0 > 0 and y1 > 0 and y2 > 0:
            ln_y0, ln_y1, ln_y2 = np.log(y0), np.log(y1), np.log(y2)
            denominator = (ln_y0 - 2 * ln_y1 + ln_y2)
            if abs(denominator) > 1e-9:
                offset = (ln_y0 - ln_y2) / (2 * denominator)
                interpolated_time = peak_idx_global + offset

    return interpolated_time, peak_amplitude

def reconstruct_point_cloud_from_bl2(sample_dir: str, amplitude_to_intensity_ratio: float = 255.0/10.0, use_answer_matrix: bool = False, min_peak_amplitude: float = 0.01):
    """
    Reconstruct point cloud from bl2 files in a sample directory.

    Parameters:
    -----------
    sample_dir : str
        Path to the sample directory containing bl2 files and config.json
    amplitude_to_intensity_ratio : float
        Ratio to convert signal amplitude to intensity
    use_answer_matrix : bool
        If True, use answer_matrix.bl2 instead of signal.bl2
    min_peak_amplitude : float
        Minimum amplitude threshold for peak detection (default: 0.01)

    Returns:
    --------
    numpy.ndarray
        Point cloud array with shape (N, 5) containing [x, y, z, intensity, ring]
    """
    # Load config
    with open(os.path.join(sample_dir, 'config.json'), 'r') as f:
        config_data = json.load(f)

    # Load signal or answer matrix
    if use_answer_matrix:
        bl2_file = os.path.join(sample_dir, 'answer_matrix.bl2')
        is_prediction = True
    else:
        bl2_file = os.path.join(sample_dir, 'signal.bl2')
        is_prediction = False

    with open(bl2_file, 'rb') as f:
        packed_data = f.read()

    hist_matrix = blosc2.unpack_array(packed_data)

    # Try to load angles.bl2 if available
    angles_file = os.path.join(sample_dir, 'angles.bl2')
    if os.path.exists(angles_file):
        with open(angles_file, 'rb') as f:
            angles_data = f.read()
        azimuth_angles = blosc2.unpack_array(angles_data)
    else:
        azimuth_angles = None

    # Get metadata
    initial_azimuth_offset = config_data.get('initial_azimuth_offset', 0.0)
    vertical_angles = config_data.get('vertical_angles', [])
    fov = config_data.get('fov', 360.0)
    time_resolution_ns = config_data.get('time_resolution_ns', 1.0)

    # Determine if this is prediction data
    is_prediction_local = len(hist_matrix.shape) == 2

    if is_prediction_local:
        channels, horizontal_resolution = hist_matrix.shape
    else:
        channels, horizontal_resolution, _ = hist_matrix.shape

    points = []

    for v_idx in range(channels):
        for h_idx in range(horizontal_resolution):
            if not is_prediction_local:
                signal = hist_matrix[v_idx, h_idx, :]
                highest_peak_time, peak_amplitude = get_peak_time_and_amplitude(signal, min_peak_amplitude)

                if highest_peak_time == 0.0:
                    continue

                intensity = np.clip(peak_amplitude * amplitude_to_intensity_ratio, 0, 255)
            else:
                highest_peak_time = hist_matrix[v_idx, h_idx]
                if highest_peak_time <= 0:
                    continue
                intensity = 100  # Default intensity for predictions

            distance_m = (highest_peak_time * time_resolution_ns) * 0.15

            altitude_deg = vertical_angles[v_idx]

            # Use actual azimuth angle from angles.bl2 if available
            if azimuth_angles is not None:
                actual_azimuth = azimuth_angles[v_idx, h_idx]
                if not np.isnan(actual_azimuth):
                    azimuth_deg = actual_azimuth
                else:
                    # Fallback to synthetic calculation
                    azimuth_deg = (h_idx / horizontal_resolution) * fov + initial_azimuth_offset
            else:
                # No angles.bl2 file, use synthetic calculation
                azimuth_deg = (h_idx / horizontal_resolution) * fov + initial_azimuth_offset

            alpha = np.deg2rad(azimuth_deg)
            omega = np.deg2rad(altitude_deg)

            x = distance_m * np.cos(omega) * np.sin(alpha)
            y = distance_m * np.cos(omega) * np.cos(alpha)
            z = distance_m * np.sin(omega)

            # Store as [x, y, z, intensity, ring]
            points.append([x, y, z, intensity, v_idx])

    return np.array(points, dtype=np.float32)

def convert_bl2_to_bin(input_dir: str, output_dir: str, amplitude_to_intensity_ratio: float = 255.0/10.0, use_answer_matrix: bool = False, format: str = 'nuscenes', min_peak_amplitude: float = 0.01):
    """
    Convert all bl2 datasets in input directory to .bin files.

    Parameters:
    -----------
    input_dir : str
        Path to the directory containing sample directories with bl2 files
    output_dir : str
        Path to the output directory for .bin files
    amplitude_to_intensity_ratio : float
        Ratio to convert signal amplitude to intensity
    use_answer_matrix : bool
        If True, use answer_matrix.bl2 instead of signal.bl2
    format : str
        Output format: 'nuscenes' (5 elements) or 'kitti' (4 elements)
    min_peak_amplitude : float
        Minimum amplitude threshold for peak detection (default: 0.01)
    """
    # Get all sample directories
    sample_dirs = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])

    if not sample_dirs:
        raise FileNotFoundError(f"No sample directories found in {input_dir}")

    print(f"Found {len(sample_dirs)} sample directories")
    print(f"Output format: {format}")
    print(f"Amplitude to intensity ratio: {amplitude_to_intensity_ratio}")
    print(f"Min peak amplitude threshold: {min_peak_amplitude}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each sample
    for sample_name in tqdm(sample_dirs, desc="Converting bl2 to bin"):
        sample_dir = os.path.join(input_dir, sample_name)

        # Reconstruct point cloud
        point_cloud = reconstruct_point_cloud_from_bl2(
            sample_dir,
            amplitude_to_intensity_ratio=amplitude_to_intensity_ratio,
            use_answer_matrix=use_answer_matrix,
            min_peak_amplitude=min_peak_amplitude
        )

        if len(point_cloud) == 0:
            print(f"Warning: No points reconstructed for {sample_name}")
            continue

        # Convert to appropriate format
        if format == 'kitti':
            # KITTI format: x, y, z, intensity (4 elements)
            output_data = point_cloud[:, :4]
        else:
            # nuScenes format: x, y, z, intensity, ring (5 elements)
            output_data = point_cloud

        # Save as .bin file
        output_path = os.path.join(output_dir, f"{sample_name}.bin")
        output_data.tofile(output_path)

        # Print first conversion info
        if sample_name == sample_dirs[0]:
            print(f"\nFirst file: {sample_name}")
            print(f"  Points: {len(point_cloud)}")
            print(f"  Shape: {output_data.shape}")
            print(f"  Output: {output_path}")

    print(f"\nConversion complete!")
    print(f"Converted {len(sample_dirs)} files from {input_dir} to {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert bl2 dataset to .bin files.")
    parser.add_argument("--input-dir", required=True, type=str,
                        help="Path to the input directory containing bl2 sample directories.")
    parser.add_argument("--output-dir", required=True, type=str,
                        help="Path to the output directory for .bin files.")
    parser.add_argument("--amplitude-to-intensity-ratio", type=float, default=255.0/10.0,
                        help="Ratio to convert signal amplitude to intensity (default: 25.5).")
    parser.add_argument("--use-answer-matrix", action='store_true',
                        help="Use answer_matrix.bl2 instead of signal.bl2 for reconstruction.")
    parser.add_argument("--format", type=str, default='nuscenes', choices=['nuscenes', 'kitti'],
                        help="Output format: 'nuscenes' (x,y,z,intensity,ring) or 'kitti' (x,y,z,intensity). Default: nuscenes")
    parser.add_argument("--min-peak-amplitude", type=float, default=0.01,
                        help="Minimum amplitude threshold for peak detection. Points with peak amplitude below this value will be excluded (default: 0.01).")

    args = parser.parse_args()

    convert_bl2_to_bin(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        amplitude_to_intensity_ratio=args.amplitude_to_intensity_ratio,
        use_answer_matrix=args.use_answer_matrix,
        format=args.format,
        min_peak_amplitude=args.min_peak_amplitude
    )
