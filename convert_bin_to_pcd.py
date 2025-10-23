import numpy as np
import open3d as o3d
import sys

if len(sys.argv) < 2:
    print("Usage: python convert_bin_to_pcd.py <input.bin> [output.pcd]")
    sys.exit(1)

input_bin = sys.argv[1]
output_pcd = sys.argv[2] if len(sys.argv) > 2 else input_bin.replace('.bin', '.pcd')

print(f"Converting {input_bin} to {output_pcd}...")

# Load binary point cloud
raw_data = np.fromfile(input_bin, dtype=np.float32)
points = raw_data.reshape(-1, 4)

print(f"Loaded {len(points):,} points")
print(f"Data shape: {points.shape}")

# Extract coordinates and intensity
xyz = points[:, :3]
intensity = points[:, 3]

print(f"\nIntensity statistics:")
print(f"  Min: {np.min(intensity):.4f}")
print(f"  Max: {np.max(intensity):.4f}")
print(f"  Mean: {np.mean(intensity):.4f}")
print(f"  Median: {np.median(intensity):.4f}")

# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)

# Convert intensity to colors (grayscale)
# Normalize intensity to [0, 1] range
if np.max(intensity) > 0:
    intensity_normalized = intensity / np.max(intensity)
else:
    intensity_normalized = intensity

colors = np.stack([intensity_normalized, intensity_normalized, intensity_normalized], axis=1)
pcd.colors = o3d.utility.Vector3dVector(colors)

# Save as PCD
o3d.io.write_point_cloud(output_pcd, pcd)

print(f"\nSaved to {output_pcd}")
print(f"Point cloud has {len(pcd.points):,} points with intensity as color")
