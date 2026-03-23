import os
import sys
import glob
import argparse
import subprocess

# This script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# The script to run on each file
TARGET_SCRIPT = os.path.join(SCRIPT_DIR, 'convert_trimmed_bl2_to_pcd_bin.py')

def main():
    """
    Batch converts (id)-data.bl2 files to .pcd.bin files using the
    convert_trimmed_bl2_to_pcd_bin.py script.
    """
    parser = argparse.ArgumentParser(
        description="Batch run convert_trimmed_bl2_to_pcd_bin.py on all '*-data.bl2' files in a directory.",
        epilog="Example: python batch_convert_bl2_to_pcd_bin.py /path/to/your/data"
    )
    parser.add_argument(
        "root_dir",
        help="The root directory to search for '*-data.bl2' files recursively."
    )
    args = parser.parse_args()

    # --- Validation ---
    if not os.path.isdir(args.root_dir):
        print(f"Error: Root directory not found: '{args.root_dir}'")
        sys.exit(1)
    
    if not os.path.isfile(TARGET_SCRIPT):
        print(f"Error: Target script not found: '{TARGET_SCRIPT}'")
        sys.exit(1)

    # --- Find Files ---
    search_pattern = os.path.join(args.root_dir, '**', 'data.bl2')
    bl2_files = glob.glob(search_pattern, recursive=True)

    if not bl2_files:
        print(f"No 'data.bl2' files found in '{args.root_dir}'.")
        sys.exit(0)

    print("==========================================")
    print(f"Found {len(bl2_files)} files to process.")
    print("==========================================")

    # --- Process Files ---
    for input_path in bl2_files:
        # Assume the output file has the same name, with the extension changed.
        # e.g., /path/to/123-data.bl2 -> /path/to/123-data.pcd.bin
        output_path = input_path.replace('.bl2', '.pcd.bin')

        print(f"\n---> Processing: {os.path.basename(input_path)}")
        print(f"     Input:  {input_path}")
        print(f"     Output: {output_path}")

        try:
            command = [
                sys.executable,  # Use the same python interpreter that is running this script
                TARGET_SCRIPT,
                input_path,
                output_path
            ]
            
            # The target script overwrites by default, so no need to delete old files.
            # We let the subprocess output stream directly to the console.
            subprocess.run(
                command,
                check=True,  # Raise an exception if the script fails
            )
            print(f"     SUCCESS: Finished processing {os.path.basename(input_path)}")

        except subprocess.CalledProcessError as e:
            print(f"     ERROR: Failed to process {os.path.basename(input_path)}.")
            print(f"     Return code: {e.returncode}")
            print("     --- STDOUT ---")
            print(e.stdout)
            print("     --- STDERR ---")
            print(e.stderr)
        except Exception as e:
            print(f"     An unexpected Python error occurred: {e}")

    print("\n==========================================")
    print("Batch processing complete.")
    print("==========================================")


if __name__ == "__main__":
    main()
