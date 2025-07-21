import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import simple_pcd_viewer as spv

from .velo_point import VeloPoint


def visualize(points: list[VeloPoint]):
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(np.array([[p.x, p.y, p.z] for p in points]))
    # cmap = plt.get_cmap("jet")
    # pcd.colors = o3d.utility.Vector3dVector(np.array([cmap(p.intensity)[:3] for p in points]))
    # o3d.visualization.draw_geometries([pcd])
    cmap = plt.get_cmap("jet")
    p = np.array([
        [p.x, p.y, p.z] for p in points
    ])
    c = np.array([
        cmap(p.intensity)[:3] for p in points
    ])
    rule = spv.single.PcdReadingRule(colorize_type=spv.single.ColorizeType.RGB)
    vis = spv.single.SingleFrameVisualizer(np.hstack([p,c]), rule=rule)
    vis.show(show_controller=True)

def visualize_comparison(original_points: np.ndarray, simulated_points, no_signal_points: np.ndarray = np.array([])):
    original_pcd = o3d.geometry.PointCloud()
    original_pcd.points = o3d.utility.Vector3dVector(original_points)
    original_pcd.paint_uniform_color([0, 0, 1])  # Blue for original

    simulated_pcd = o3d.geometry.PointCloud()
    
    # Handle both list of VeloPoint and numpy array for simulated_points
    if isinstance(simulated_points, list) and len(simulated_points) > 0 and isinstance(simulated_points[0], VeloPoint):
        simulated_points_np = np.array([[p.x, p.y, p.z] for p in simulated_points])
    elif isinstance(simulated_points, np.ndarray):
        simulated_points_np = simulated_points
    else:
        simulated_points_np = np.array([])

    if simulated_points_np.size > 0:
        simulated_pcd.points = o3d.utility.Vector3dVector(simulated_points_np)
        simulated_pcd.paint_uniform_color([1, 0, 0])  # Red for simulated

    geometries_to_draw = [original_pcd, simulated_pcd]

    if no_signal_points.size > 0:
        print(f"Visualizing {len(no_signal_points)} no-signal points.")
        no_signal_pcd = o3d.geometry.PointCloud()
        no_signal_pcd.points = o3d.utility.Vector3dVector(no_signal_points)
        no_signal_pcd.paint_uniform_color([0, 1, 0])  # Green for no signal
        geometries_to_draw.append(no_signal_pcd)

    o3d.visualization.draw_geometries(geometries_to_draw)
