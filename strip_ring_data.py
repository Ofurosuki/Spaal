
import numpy as np
import argparse
import os

def strip_ring_data(input_path: str, output_path: str, normalize_intensity: bool = True, shift_x: float = 0.0):
    """
    Reads a 5xN .bin file, strips the 5th element (ring data),
    optionally normalizes intensity and shifts x-coordinate, and writes the resulting 4xN data to a new .bin file.

    Args:
        input_path: Path to input 5xN .bin file
        output_path: Path to output 4xN .bin file
        normalize_intensity: If True, normalizes intensity (4th column) by dividing by 255.0
        shift_x: Amount to shift x-coordinate (e.g., 12.0 for KITTI compatibility)
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    try:
        # Read the 5xN data
        data_5xN = np.fromfile(input_path, dtype=np.float32).reshape(-1, 5)

        # Strip the 5th column (ring data)
        data_4xN = data_5xN[:, :4].copy()  # Make a copy to avoid modifying original

        # Shift x-coordinate if requested (for KITTI compatibility)
        if shift_x != 0.0:
            data_4xN[:, 0] = data_4xN[:, 0] + shift_x

        # Normalize intensity if requested
        if normalize_intensity:
            # Intensity is in column 3 (0-indexed)
            intensity_before = data_4xN[:, 3].copy()
            data_4xN[:, 3] = data_4xN[:, 3] / 255.0

            # Optional: Print normalization info for debugging
            # print(f"  Intensity normalized: [{intensity_before.min():.3f}, {intensity_before.max():.3f}] -> [{data_4xN[:, 3].min():.3f}, {data_4xN[:, 3].max():.3f}]")

        # Write the 4xN data to the output file
        data_4xN.tofile(output_path)

        messages = []
        if shift_x != 0.0:
            messages.append(f"x+{shift_x}m")
        if normalize_intensity:
            messages.append("intensity normalized")

        msg_str = " (" + ", ".join(messages) + ")" if messages else ""
        print(f"Successfully processed {os.path.basename(input_path)}{msg_str}")
    except Exception as e:
        print(f"Error processing file {input_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Strips the 5th element (ring data) from a 5xN .bin file to create a 4xN .bin file, with optional intensity normalization."
    )
    parser.add_argument(
        "input_file",
        help="Path to the input 5xN .bin file."
    )
    parser.add_argument(
        "output_file",
        help="Path to the output 4xN .bin file."
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Don't normalize intensity values (default: normalize by dividing by 255.0)."
    )
    parser.add_argument(
        "--shift-x",
        type=float,
        default=0.0,
        help="Amount to shift x-coordinate in meters (e.g., 12.0 for KITTI compatibility, default: 0.0)."
    )
    args = parser.parse_args()

    strip_ring_data(args.input_file, args.output_file, normalize_intensity=not args.no_normalize, shift_x=args.shift_x)
