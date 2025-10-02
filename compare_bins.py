import os
import numpy as np
import argparse

class BinComparer:
    """
    Compares two .pcd.bin files to see if their contents are identical.
    """
    def __init__(self, file1_path: str, file2_path: str):
        self.file1_path = file1_path
        self.file2_path = file2_path

        if not os.path.isfile(self.file1_path):
            raise FileNotFoundError(f"File not found: {self.file1_path}")
        if not os.path.isfile(self.file2_path):
            raise FileNotFoundError(f"File not found: {self.file2_path}")

    def are_equal(self) -> bool:
        """
        Performs the comparison and prints the result.
        It first prints the first 20 points of each file, then compares them.
        If files are different, it prints the first differing elements.
        
        Returns:
            True if files are identical, False otherwise.
        """
        # 1. Check file sizes for a quick check.
        size1 = os.path.getsize(self.file1_path)
        size2 = os.path.getsize(self.file2_path)

        # 2. Load data and print preview
        try:
            scan1 = np.fromfile(self.file1_path, dtype=np.float32)
            scan2 = np.fromfile(self.file2_path, dtype=np.float32)

            print("-" * 30)
            print(f"Preview of {self.file1_path} ({size1} bytes):")
            try:
                points1_preview = scan1.reshape((-1, 5))
                print(points1_preview[:20])
            except ValueError:
                print("Could not reshape file 1 to points (expected 5 floats per point). Showing flat data.")
                print(scan1[:100]) # 20 points * 5 values

            print("-" * 30)
            print(f"Preview of {self.file2_path} ({size2} bytes):")
            try:
                points2_preview = scan2.reshape((-1, 5))
                print(points2_preview[:20])
            except ValueError:
                print("Could not reshape file 2 to points (expected 5 floats per point). Showing flat data.")
                print(scan2[:100]) # 20 points * 5 values
            print("-" * 30)

        except Exception as e:
            print(f"An error occurred while reading files: {e}")
            return False

        # 3. Compare sizes and content
        if size1 != size2:
            print(f"Files are different: Size mismatch ({size1} bytes vs {size2} bytes).")
            return False

        if np.array_equal(scan1, scan2):
            print("Files are identical.")
            return True
        else:
            print("Files are different: Content mismatch.")
            try:
                # Reshape to (N, 5) to find which points/fields differ
                points1 = scan1.reshape((-1, 5))
                points2 = scan2.reshape((-1, 5))

                # Find the first index of a mismatch
                diff_mask = (points1 != points2)
                first_diff_point_idx = np.where(diff_mask)[0][0]
                
                print(f"First difference found at point index {first_diff_point_idx}:")
                print(f"  - File 1 point: {points1[first_diff_point_idx]}")
                print(f"  - File 2 point: {points2[first_diff_point_idx]}")

            except ValueError:
                # Fallback for when reshape fails
                diff_indices = np.where(scan1 != scan2)[0]
                if diff_indices.size > 0:
                    first_diff_idx = diff_indices[0]
                    print(f"First difference in flat array at index {first_diff_idx}:")
                    print(f"  - File 1 value: {scan1[first_diff_idx]}")
                    print(f"  - File 2 value: {scan2[first_diff_idx]}")
            return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare two .pcd.bin files for equality.')
    parser.add_argument('file1', type=str, help='Path to the first .pcd.bin file.')
    parser.add_argument('file2', type=str, help='Path to the second .pcd.bin file.')
    args = parser.parse_args()

    try:
        comparer = BinComparer(args.file1, args.file2)
        comparer.are_equal()
    except FileNotFoundError as e:
        print(e)
