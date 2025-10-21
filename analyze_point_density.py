import numpy as np
import matplotlib.pyplot as plt
import argparse

def analyze_point_density(bin_file_path: str, target_resolution_deg: float = 0.1):
    """
    Analyze how many points fall into each grid cell and simulate thinning.
    """

    # Load point cloud
    raw_data = np.fromfile(bin_file_path, dtype=np.float32)
    if raw_data.size % 4 == 0:
        points = raw_data.reshape(-1, 4)[:, :3]
    else:
        points = raw_data.reshape(-1, 5)[:, :3]

    print(f"Original points: {len(points):,}")

    # Calculate spherical coordinates
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    distances = np.linalg.norm(points, axis=1)
    azimuth_deg = np.rad2deg(np.arctan2(x, y))
    elevation_deg = np.rad2deg(np.arcsin(z / distances))

    # Grid discretization with target resolution
    horizontal_resolution = int(target_resolution_deg * 100)
    azimuth_keys = ((azimuth_deg * 100) / horizontal_resolution).astype(int) * horizontal_resolution
    elevation_keys = (elevation_deg * 100).astype(int)

    # Count points per grid cell
    grid_cells = {}
    for i in range(len(points)):
        key = (azimuth_keys[i], elevation_keys[i])
        if key not in grid_cells:
            grid_cells[key] = []
        grid_cells[key].append(i)

    # Analyze distribution
    points_per_cell = [len(indices) for indices in grid_cells.values()]

    print(f"\n=== Grid Cell Analysis (Resolution: {target_resolution_deg}deg) ===")
    print(f"Total grid cells: {len(grid_cells):,}")
    print(f"Grid cells with points: {len(grid_cells):,}")
    print(f"Average points per cell: {np.mean(points_per_cell):.2f}")
    print(f"Median points per cell: {np.median(points_per_cell):.1f}")
    print(f"Max points per cell: {np.max(points_per_cell)}")
    print(f"Cells with 1 point: {sum(1 for x in points_per_cell if x == 1):,} ({sum(1 for x in points_per_cell if x == 1)/len(points_per_cell)*100:.1f}%)")
    print(f"Cells with 2-5 points: {sum(1 for x in points_per_cell if 2 <= x <= 5):,} ({sum(1 for x in points_per_cell if 2 <= x <= 5)/len(points_per_cell)*100:.1f}%)")
    print(f"Cells with 6-10 points: {sum(1 for x in points_per_cell if 6 <= x <= 10):,} ({sum(1 for x in points_per_cell if 6 <= x <= 10)/len(points_per_cell)*100:.1f}%)")
    print(f"Cells with >10 points: {sum(1 for x in points_per_cell if x > 10):,} ({sum(1 for x in points_per_cell if x > 10)/len(points_per_cell)*100:.1f}%)")

    # Current method: keep only nearest point per cell
    current_method_points = len(grid_cells)
    current_loss = len(points) - current_method_points
    print(f"\n=== Current Method (Nearest Point Only) ===")
    print(f"Points kept: {current_method_points:,}")
    print(f"Points lost: {current_loss:,} ({current_loss/len(points)*100:.1f}%)")

    # Simulate thinning to target resolution
    print(f"\n=== Thinning Simulation ===")

    # Strategy 1: Keep nearest point in each cell
    thinned_points_nearest = []
    for cell_indices in grid_cells.values():
        # Find nearest point in this cell
        cell_distances = distances[cell_indices]
        nearest_idx = cell_indices[np.argmin(cell_distances)]
        thinned_points_nearest.append(nearest_idx)

    print(f"Strategy 1 (Nearest): {len(thinned_points_nearest):,} points ({len(thinned_points_nearest)/len(points)*100:.1f}%)")
    print(f"  Loss from original: {len(points) - len(thinned_points_nearest):,} points ({(len(points) - len(thinned_points_nearest))/len(points)*100:.1f}%)")

    # Strategy 2: Keep first point in each cell (original order)
    thinned_points_first = []
    for cell_indices in grid_cells.values():
        thinned_points_first.append(cell_indices[0])

    print(f"Strategy 2 (First): {len(thinned_points_first):,} points ({len(thinned_points_first)/len(points)*100:.1f}%)")

    # Strategy 3: Stratified sampling (keep one point per cell, distributed)
    # This is equivalent to Strategy 1 for single-point-per-cell

    # Calculate actual angular spacing in thinned data
    thinned_azimuth = azimuth_deg[thinned_points_nearest]
    thinned_azimuth_sorted = np.sort(thinned_azimuth)
    thinned_azimuth_diffs = np.diff(thinned_azimuth_sorted)
    meaningful_diffs = thinned_azimuth_diffs[(thinned_azimuth_diffs > 0.01) & (thinned_azimuth_diffs < 1.0)]

    if len(meaningful_diffs) > 0:
        print(f"\n=== Thinned Point Cloud Angular Spacing ===")
        print(f"Mean azimuth spacing: {np.mean(meaningful_diffs):.4f}deg")
        print(f"Median azimuth spacing: {np.median(meaningful_diffs):.4f}deg")
        print(f"Target: {target_resolution_deg:.4f}deg")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Histogram of points per cell
    ax = axes[0, 0]
    ax.hist(points_per_cell, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(points_per_cell), color='r', linestyle='--',
               label=f'Mean: {np.mean(points_per_cell):.1f}')
    ax.axvline(np.median(points_per_cell), color='g', linestyle='--',
               label=f'Median: {np.median(points_per_cell):.1f}')
    ax.set_xlabel('Points per Grid Cell')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of Points per Grid Cell ({target_resolution_deg}deg resolution)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Scatter plot of grid occupancy
    ax = axes[0, 1]
    cell_azimuth = [k[0]/100 for k in grid_cells.keys()]
    cell_elevation = [k[1]/100 for k in grid_cells.keys()]
    cell_counts = [len(indices) for indices in grid_cells.values()]
    scatter = ax.scatter(cell_azimuth, cell_elevation, c=cell_counts,
                        s=1, cmap='hot', alpha=0.5)
    plt.colorbar(scatter, ax=ax, label='Points per cell')
    ax.set_xlabel('Azimuth (degrees)')
    ax.set_ylabel('Elevation (degrees)')
    ax.set_title('Grid Cell Occupancy')
    ax.grid(True, alpha=0.3)

    # Plot 3: CDF of points per cell
    ax = axes[1, 0]
    sorted_counts = np.sort(points_per_cell)
    cdf = np.arange(1, len(sorted_counts)+1) / len(sorted_counts) * 100
    ax.plot(sorted_counts, cdf, linewidth=2)
    ax.axhline(50, color='r', linestyle='--', alpha=0.5, label='50th percentile')
    ax.axhline(90, color='orange', linestyle='--', alpha=0.5, label='90th percentile')
    ax.axvline(1, color='g', linestyle='--', alpha=0.5, label='1 point/cell')
    ax.set_xlabel('Points per Grid Cell')
    ax.set_ylabel('Cumulative Percentage (%)')
    ax.set_title('CDF: Points per Cell')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Comparison of methods
    ax = axes[1, 1]
    methods = ['Original', 'Current\n(Nearest)', 'After\nThinning']
    point_counts = [len(points), current_method_points, len(thinned_points_nearest)]
    colors = ['blue', 'orange', 'green']
    bars = ax.bar(methods, point_counts, color=colors, alpha=0.7)

    # Add percentages on bars
    for i, (bar, count) in enumerate(zip(bars, point_counts)):
        height = bar.get_height()
        percentage = count / len(points) * 100
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}\n({percentage:.1f}%)',
                ha='center', va='bottom')

    ax.set_ylabel('Number of Points')
    ax.set_title('Point Count Comparison')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('point_density_analysis.png', dpi=150)
    print(f"\n=== Plot saved to point_density_analysis.png ===")

    # Key insight
    print(f"\n=== Key Insight ===")
    cells_with_multiple = sum(1 for x in points_per_cell if x > 1)
    total_extra_points = sum(max(0, x-1) for x in points_per_cell)

    print(f"Grid cells with multiple points: {cells_with_multiple:,} ({cells_with_multiple/len(points_per_cell)*100:.1f}%)")
    print(f"Total 'extra' points (beyond first): {total_extra_points:,}")
    print(f"These {total_extra_points:,} points will be lost in current method")
    print(f"\nIf we thin to {target_resolution_deg}deg first:")
    print(f"  - We intentionally remove ~{len(points) - len(thinned_points_nearest):,} points")
    print(f"  - But we can choose WHICH points to keep (nearest, highest intensity, etc.)")
    print(f"  - Grid mapping then has minimal collision (1-2 points per cell)")
    print(f"  - Result: Better control over which points survive")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze point density per grid cell")
    parser.add_argument("--bin-file", type=str, required=True, help="Path to .bin file")
    parser.add_argument("--resolution", type=float, default=0.1,
                       help="Target grid resolution in degrees")

    args = parser.parse_args()
    analyze_point_density(args.bin_file, args.resolution)
