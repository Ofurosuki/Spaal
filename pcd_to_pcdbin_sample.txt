import os
import numpy as np
import open3d as o3d
import argparse

class PCDToNuScenesConverter:
    """
    Converts .pcd files to nuScenes-compatible .pcd.bin files.
    """
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = input_dir
        self.output_dir = output_dir

        if not os.path.isdir(self.input_dir):
            raise ValueError(f"Input directory not found: {self.input_dir}")
        
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output will be saved to: {self.output_dir}")

    def convert(self):
        """
        Iterates through .pcd files in the input directory and converts them.
        """
        pcd_files = sorted([f for f in os.listdir(self.input_dir) if f.endswith('.pcd')])
        if not pcd_files:
            print(f"No .pcd files found in {self.input_dir}")
            return

        print(f"Found {len(pcd_files)} .pcd files to convert.")

        for pcd_file in pcd_files:
            input_path = os.path.join(self.input_dir, pcd_file)
            output_filename = os.path.splitext(pcd_file)[0] + '.pcd.bin'
            output_path = os.path.join(self.output_dir, output_filename)
            
            print(f"Converting {input_path} -> {output_path} ...")

            try:
                pcd = o3d.io.read_point_cloud(input_path)
                points = np.asarray(pcd.points)

                # nuScenes .bin format is typically [x, y, z, intensity, ring_index].
                # A standard .pcd file may not have intensity or ring_index.
                # We will create dummy values for them if they are not present.
                
                num_points = points.shape[0]
                
                # Create dummy intensity (e.g., 100.0) and ring_index (e.g., 0)
                # In a real scenario, you might need a more sophisticated way to get these values.
                intensity = np.full((num_points, 1), 100.0, dtype=np.float32)
                ring_index = np.zeros((num_points, 1), dtype=np.float32)

                # Combine to create the (N, 5) array for nuScenes
                nuscenes_points = np.hstack((points, intensity, ring_index)).astype(np.float32)

                # Save to binary file
                nuscenes_points.tofile(output_path)
                print("  -> Success.")

            except Exception as e:
                print(f"  -> Failed to convert {pcd_file}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert .pcd files to nuScenes .pcd.bin format.')
    parser.add_argument('input_dir', type=str, help='Directory containing the input .pcd files.')
    parser.add_argument('output_dir', type=str, help='Directory to save the output .pcd.bin files.')
    args = parser.parse_args()

    converter = PCDToNuScenesConverter(args.input_dir, args.output_dir)
    converter.convert()
