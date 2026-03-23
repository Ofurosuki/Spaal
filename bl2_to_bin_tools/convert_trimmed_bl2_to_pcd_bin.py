import sys
import os
import argparse
import numpy as np
import blosc2
import struct
from tqdm import tqdm
from typing import Tuple
import glob

# Add parent directory to path to allow importing from other scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PIXEL_X_MAX = 192
PIXEL_Y_MAX = 56

# The angular resolution of the sensor in degrees per pixel.
RESOLUTION = 0.47  # degrees/pixel

def get_angle_from_pixel(pixel_x: int, pixel_y: int) -> tuple[float, float]:
    """
    Converts a pixel coordinate (Pixel_X, Pixel_Y) to a 3D angle 
    (azimuth, altitude) based on a fixed resolution.

    The center of the pixel grid (PIXEL_X_MAX/2, PIXEL_Y_MAX/2) is assumed 
    to correspond to the (0, 0) angle.

    Args:
        pixel_x: The x-coordinate of the pixel (0 to PIXEL_X_MAX).
        pixel_y: The y-coordinate of the pixel (0 to PIXEL_Y_MAX).

    Returns:
        A tuple containing the azimuth and altitude in degrees.
    """
    # Calculate the center of the pixel grid
    center_x = PIXEL_X_MAX / 2
    center_y = PIXEL_Y_MAX / 2

    # Calculate angles by multiplying the pixel's distance from the center 
    # by the angular resolution.
    azimuth = (pixel_x - center_x) * RESOLUTION 
    altitude = (pixel_y - center_y) * RESOLUTION

    return azimuth, altitude
def get_peak_time_and_amplitude(signal: np.ndarray, min_amplitude: float = 0.01) -> Tuple[float, float]:
    """
    Finds the interpolated time and amplitude of the highest peak in a signal.

    Args:
        signal: Input signal array
        min_amplitude: Minimum amplitude threshold for peak detection (default: 0.0, accepts any peak)

    Returns:
        (interpolated_time, peak_amplitude) or (0.0, 0.0) if no peak is found.
    """
    # Find where the signal raises above the threshold
    raises = np.flatnonzero((signal[:-1] < min_amplitude) & (signal[1:] >= min_amplitude)) + 1

    # Check if signal starts above threshold (peak starting from index 0)
    if len(signal) > 0 and signal[0] >= min_amplitude:
        raises = np.concatenate([[0], raises])

    if len(raises) == 0:
        return 0.0, 0.0

    # Find the maximum value (peak) in a window following each raise
    peak_values = np.array([np.max(signal[r:min(len(signal), r + 50)]) for r in raises])
    if len(peak_values) == 0:
        return 0.0, 0.0

    peak_amplitude = np.max(peak_values)
    if peak_amplitude < min_amplitude:
        return 0.0, 0.0

    # Find the start index of the pulse with the highest peak
    highest_pulse_start_index = raises[np.argmax(peak_values)]
    pulse_region = signal[highest_pulse_start_index:min(len(signal), highest_pulse_start_index + 50)]
    
    if len(pulse_region) == 0:
        return 0.0, 0.0
        
    # Find the index of the peak within its local region
    peak_idx_in_region = np.argmax(pulse_region)
    peak_idx_global = highest_pulse_start_index + peak_idx_in_region

    # Perform quadratic interpolation to get a more precise peak time
    interpolated_time = float(peak_idx_global)
    if 0 < peak_idx_global < len(signal) - 1:
        y0, y1, y2 = signal[peak_idx_global - 1:peak_idx_global + 2]
        # Use log-quadratic interpolation for better accuracy with pulse shapes
        if y0 > 0 and y1 > 0 and y2 > 0:
            ln_y0, ln_y1, ln_y2 = np.log(y0), np.log(y1), np.log(y2)
            denominator = (ln_y0 - 2 * ln_y1 + ln_y2)
            if abs(denominator) > 1e-9:
                offset = (ln_y0 - ln_y2) / (2 * denominator)
                interpolated_time = peak_idx_global + offset
    
    return interpolated_time, peak_amplitude

# --- Configuration ---
LIDAR_A_WIDTH = 192
TARGET_WIDTH = 1800
TARGET_HEIGHT = 32
Y_CROP_TOP = 24
X_PADDING_LEFT = (TARGET_WIDTH - LIDAR_A_WIDTH) // 2  # 804
PEAK_THRSHOLD = 20  # Minimum amplitude to consider a peak valid

# --- Helper Functions ---

def load_bl2_file(file_path: str) -> np.ndarray:
    """Loads and unpacks a Blosc2 compressed NumPy array."""
    print(f"Loading data from {file_path}...")
    try:
        with open(file_path, 'rb') as f:
            packed_array = f.read()
            unpacked_array = blosc2.unpack_array(packed_array)
        print(f"Loaded array with shape {unpacked_array.shape}")
        return unpacked_array
    except FileNotFoundError:
        print(f"Error: Input file not found at {file_path}")
        sys.exit(1)

def spherical_to_cartesian(depth, azimuth, altitude):
    """Converts spherical coordinates to Cartesian coordinates."""
    r = depth * 0.15  # Convert depth from bin number to meters
    az_rad = np.deg2rad(azimuth)
    alt_rad = np.deg2rad(altitude)
    x = r * np.cos(alt_rad) * np.sin(az_rad)
    y = r * np.cos(alt_rad) * np.cos(az_rad)
    z = -r * np.sin(alt_rad)
    return x, y, z

def write_binary_file(points: list, output_path: str):
    """Writes points to a binary file in (x, y, z, intensity, ring) format."""
    print(f"Writing {len(points)} points to binary file: {output_path}")
    try:
        with open(output_path, 'wb') as f:
            for point in points:
                packed_data = struct.pack('fffff', point[0], point[1], point[2], point[3], float(point[4]))
                f.write(packed_data)
        print("Binary file writing complete.")
    except Exception as e:
        print(f"An error occurred while writing the binary file: {e}")

# --- Main Conversion Logic ---

def convert_trimmed_bl2_to_bin(bl2_path: str, bin_path: str, z_threshold: float):
    """
    Loads a .bl2 file, saves all points, but calculates intensity multiplier
    based on the original logic (strong signals only).
    """
    bl2_array = load_bl2_file(bl2_path)

    # Pass 1: Extract all points, but separate amplitudes for multiplier calculation
    print("Pass 1: Extracting all points and identifying strong signals for intensity calculation...")
    all_points_info = []
    strong_amplitudes = [] # For calculating the multiplier, as per original logic
    x_start = X_PADDING_LEFT
    x_end = X_PADDING_LEFT + LIDAR_A_WIDTH

    # Use the original threshold for multiplier calculation logic
    ORIGINAL_PEAK_THRESHOLD = 20

    for y_b in tqdm(range(TARGET_HEIGHT), desc="Analyzing rows"):
        for x_b in range(x_start, x_end):
            hist = bl2_array[y_b, x_b, :]
            if np.any(hist > 0):
                # Get peak with no threshold to ensure no points are dropped
                depth, amp = get_peak_time_and_amplitude(hist, PEAK_THRSHOLD)
                
                if amp > 0:
                    x_a = x_b - X_PADDING_LEFT
                    y_a = y_b + Y_CROP_TOP
                    azimuth, altitude = get_angle_from_pixel(x_a, y_a)
                    
                    # Add every valid point to the list for final output
                    all_points_info.append({'depth': depth, 'azimuth': azimuth, 'altitude': altitude, 'amplitude': amp})
                    
                    # But only use strong signals for the multiplier calculation
                    if amp > ORIGINAL_PEAK_THRESHOLD:
                        strong_amplitudes.append(amp)

    if not all_points_info:
        print("No valid points found in the .bl2 file. Writing empty .bin file.")
        write_binary_file([], bin_path)
        return

    # --- Intensity Multiplier Calculation (using original logic) ---
    if strong_amplitudes:
        median_amplitude = np.median(strong_amplitudes)
        if median_amplitude > 0:
            multiplier = 12.0 / median_amplitude
            print(f"Median of strong signals ({len(strong_amplitudes)} points) is {median_amplitude:.2f}. Calculated intensity multiplier: {multiplier:.4f}")
        else:
            # This case is unlikely if strong_amplitudes is not empty and threshold > 0
            multiplier = 1.0
    else:
        print("Warning: No strong signals found for intensity normalization. Using default multiplier of 1.0")
        multiplier = 1.0

    # Pass 2: Build final point cloud for ALL points, applying the calculated multiplier
    print(f"Pass 2: Building final point cloud for {len(all_points_info)} points...")
    points_to_write = []
    for point_info in tqdm(all_points_info, desc="Generating points", leave=False):
        x, y, z = spherical_to_cartesian(point_info['depth'], point_info['azimuth'], point_info['altitude'])
        
        # Apply Z-coordinate filtering for ground points
        if z <= z_threshold:
            continue # Skip this point if it's considered ground

        intensity = point_info['amplitude'] * multiplier
        ring = 0
        points_to_write.append((x, y, z, intensity, ring))

    print(f"Generated {len(points_to_write)} points. (All filtering was disabled)")
    write_binary_file(points_to_write, bin_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert .bl2 files from an input directory to .pcd.bin files in an output directory, optionally filtering ground points."
    )
    parser.add_argument("input_dir", help="Path to the source directory containing .bl2 files (trimmed format).")
    parser.add_argument("output_dir", help="Path for the destination directory to save .pcd.bin files.")
    parser.add_argument(
        "--z_ground_threshold",
        type=float,
        default=-2.0,
        help="Z-coordinate threshold in meters. Points with Z <= this value are considered ground and removed. (Default: -2.0)"
    )
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    bl2_files = sorted(glob.glob(os.path.join(args.input_dir, '**', '*.bl2'), recursive=True))

    if not bl2_files:
        print(f"No .bl2 files found in the input directory: {args.input_dir}")
        sys.exit(0)

    for input_bl2_path in tqdm(bl2_files, desc="Converting .bl2 to .pcd.bin"):
        # Determine the base name for the output file
        if os.path.basename(input_bl2_path) == 'signal.bl2':
            # If it's a 'signal.bl2' file inside a directory, use the parent directory's name
            base_name = os.path.basename(os.path.dirname(input_bl2_path))
        else:
            # Otherwise, use the file's own name (without extension)
            base_name = os.path.splitext(os.path.basename(input_bl2_path))[0]
            
        output_bin_filename = base_name + '.pcd.bin'
        output_bin_path = os.path.join(args.output_dir, output_bin_filename)
        convert_trimmed_bl2_to_bin(input_bl2_path, output_bin_path, args.z_ground_threshold)