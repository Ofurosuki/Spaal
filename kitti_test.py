import numpy as np
import open3d as o3d

# .binファイルのパス
bin_file_path = 'D:/testing/velodyne/000801.bin'

# float32型としてバイナリを読み込み、Nx4の形状に変形
try:
    point_cloud = np.fromfile(bin_file_path, dtype=np.float32).reshape((-1, 4))
    
    # (N, 4) という形状と、最初の点の [x, y, z, r] を表示
    print(f"Shape: {point_cloud.shape}")
    if point_cloud.shape[0] > 0:
        print(f"First point (x, y, z, intensity): {point_cloud[0]}")

except FileNotFoundError:
    print(f"Error: File not found at {bin_file_path}")
except ValueError as e:
    print(f"Error: Could not reshape array. Is the file format correct? {e}")

    # Visualize the point cloud using Open3D
if point_cloud.shape[0] > 0:
    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    
    # Assign points (x, y, z) to the PointCloud object
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    # Add a coordinate frame to the visualization
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, coordinate_frame])
    # Visualize the point cloud
    