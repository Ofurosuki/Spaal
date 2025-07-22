import numpy as np
from spaal2.core.dummy_lidar.dummy_lidar_vlp16_pcd import PcdLidarVLP16
from spaal2.core import (
    PreciseDuration, 
    visualize_comparison,
    DummyOutdoor,
    apply_noise,
    DummySpooferAdaptiveHFR
)
import open3d as o3d
import matplotlib.pyplot as plt


def run_pcd_lidar_example_with_hfr_spoofer():
    # Path to the PCD file
    # The user provided this path, assuming it exists.
    #pcd_file_path = "C:/Users/nextr/spaal2-core/example/1464001237.670017000.pcd"
    pcd_file_path = "C:/Users/nextr/spaal2-core/example/1464002254.951256000.pcd"

    # LiDAR position and orientation
    lidar_position = np.array([0.0, 0.0, 0.0])
    lidar_rotation = np.array([0.0, 0.0, 0.0])

    # Create a PcdLidarVLP16 instance
    lidar = PcdLidarVLP16(
        pcd_file_path=pcd_file_path,
        lidar_position=lidar_position,
        lidar_rotation=lidar_rotation,
        base_timestamp=PreciseDuration(nanoseconds=0),
        time_resolution_ns=0.2 # ここで分解能を設定
    )
    
    # Setup outdoor environment and HFR spoofer
    outdoor = DummyOutdoor(50.0, 0.8)
    spoofer = DummySpooferAdaptiveHFR(
        frequency=20 * 1e6, 
        duration=PreciseDuration(milliseconds=20),
        spoofer_distance_m=10.0,
        pulse_width=PreciseDuration(nanoseconds=5),
        time_resolution_ns=lidar.time_resolution_ns, # LiDARの分解能と統一
    )

    print("Running PCD LiDAR example with HFR spoofer...")

    point_list = []
    signal_for_point = []  # 各点に対応するsignalを保存
    # Perform a full scan
    Triggered = False
    for i in range(lidar.max_index):
        try:
            config, signal = lidar.scan()

            # Trigger the spoofer at a specific angle (e.g., altitude 1 degree, azimuth ~180 degrees)
            if config.altitude == 100 and abs(config.azimuth - 0) == 7000:
                spoofer.trigger(config, signal)
                Triggered = True

            # Apply outdoor conditions and noise to the legitimate signal
            #signal = apply_noise(outdoor.apply(signal), ratio=0.1)
            
            # Get the spoofer's signal and add noise
            external_signal = apply_noise(spoofer.get_range_signal(config.start_timestamp, config.accept_duration), ratio=0.01)
            
            # Combine the legitimate and spoofed signals
            signal_combined = np.maximum(signal, external_signal)
            signal_combined = np.clip(signal_combined, 0, 9)

            # Receive points from the combined signal
            points = lidar.receive(config, signal_combined)
            if points:
                point_list.extend(points)
                # 各点に対応するsignalを保存（各点に同じsignalを紐付ける）
                signal_for_point.extend([signal_combined.copy()] * len(points))

        except StopIteration:
            print("End of scan.")
            break
    
    print(f"Detected {len(point_list)} points.")
    original_points = np.asarray(lidar.point_cloud.points)

    no_signal_points = lidar.get_no_signal_points()
    print(f"Number of no-signal points: {len(no_signal_points)}")

    # Convert the list of VeloPoint objects to a NumPy array for visualization
    simulated_points_np = np.array([[p.x, p.y, p.z] for p in point_list])

    visualize_comparison(original_points, simulated_points_np, no_signal_points)

    # --- Open3Dでインタラクティブ可視化 ---
    if len(point_list) == 0:
        print("No points to visualize.")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(simulated_points_np)
    colors = np.tile([0.5, 0.5, 0.5], (len(point_list), 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # original_pointsもOpen3D点群として作成（青色）
    pcd_original = o3d.geometry.PointCloud()
    pcd_original.points = o3d.utility.Vector3dVector(original_points)
    colors_original = np.tile([0.2, 0.2, 1.0], (len(original_points), 1))
    pcd_original.colors = o3d.utility.Vector3dVector(colors_original)

    current_index = [0]  # クロージャで書き換え可能に

    # 球体を毎回新規生成し、絶対座標に配置する
    def create_highlight_sphere(center, color, radius=0.5):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.paint_uniform_color(color)
        sphere.translate(center)
        return sphere

    highlight_sphere_sim = create_highlight_sphere(simulated_points_np[current_index[0]], [1, 0, 0], radius=0.5)
    highlight_sphere_org = create_highlight_sphere(original_points[current_index[0]], [0, 1, 0], radius=0.5)

    def highlight_point(vis):
        nonlocal highlight_sphere_sim, highlight_sphere_org
        # シミュレート点群の色更新
        colors = np.tile([0.5, 0.5, 0.5], (len(point_list), 1))
        colors[current_index[0]] = [1, 0, 0]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        vis.update_geometry(pcd)
        # original点群の色は固定なので更新不要

        # 球体の位置更新（remove→addで再生成）
        vis.remove_geometry(highlight_sphere_sim, reset_bounding_box=False)
        highlight_sphere_sim = create_highlight_sphere(simulated_points_np[current_index[0]], [1, 0, 0], radius=0.5)
        vis.add_geometry(highlight_sphere_sim, reset_bounding_box=False)

        vis.remove_geometry(highlight_sphere_org, reset_bounding_box=False)
        highlight_sphere_org = create_highlight_sphere(original_points[current_index[0]], [0, 1, 0], radius=0.5)
        vis.add_geometry(highlight_sphere_org, reset_bounding_box=False)

        vis.poll_events()
        vis.update_renderer()

    def on_key_q(vis):
        current_index[0] = max(0, current_index[0] - 1)
        highlight_point(vis)
        print(f"Current index: {current_index[0]}")
        return False

    def on_key_e(vis):
        current_index[0] = min(len(point_list) - 1, current_index[0] + 1)
        highlight_point(vis)
        print(f"Current index: {current_index[0]}")
        return False

    def on_key_v(vis):
        idx = current_index[0]
        signal = signal_for_point[idx]
        plt.figure()
        plt.plot(signal)
        plt.title(f"Signal for point index {idx}")
        plt.xlabel("Sample Index")
        plt.ylabel("Signal Value")
        plt.show()
        return False

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="PCD LiDAR Points Interactive Viewer")
    vis.add_geometry(pcd)
    vis.add_geometry(pcd_original)
    vis.add_geometry(highlight_sphere_sim)
    vis.add_geometry(highlight_sphere_org)
    highlight_point(vis)
    vis.register_key_callback(ord("Q"), on_key_q)
    vis.register_key_callback(ord("E"), on_key_e)
    vis.register_key_callback(ord("V"), on_key_v)
    print("操作方法: Q/Eキーで点を移動、Vキーでsignalを可視化、ESCで終了")
    vis.run()
    vis.destroy_window()

    # --- 元のdivergent points出力処理はそのまま ---
    distance_threshold = 0.05  # 5 cm
    divergent_points_output = []

    min_len = min(len(original_points), len(simulated_points_np))

    for i in range(min_len):
        original_point = original_points[i]
        simulated_point = simulated_points_np[i]
        distance = np.linalg.norm(original_point - simulated_point)

        if distance > distance_threshold:
            divergent_points_output.append(
                f"Index: {i}, Original: {original_point}, Simulated: {simulated_point}, Distance: {distance:.4f}"
            )

    if divergent_points_output:
        with open("divergent_points.txt", "w") as f:
            f.write("\n".join(divergent_points_output))
        print(f"Found {len(divergent_points_output)} divergent points. Details saved to divergent_points.txt")
    else:
        print("No divergent points found within the specified threshold.")

if __name__ == "__main__":
    run_pcd_lidar_example_with_hfr_spoofer()
