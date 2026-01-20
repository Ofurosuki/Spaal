"""
KITTI Coordinate System Transformations

KITTI uses different coordinate systems for different sensors:

1. Camera Coordinate System (used in annotations):
   - X: right
   - Y: down
   - Z: forward
   - Right-handed system

2. Velodyne (LiDAR) Coordinate System:
   - X: forward
   - Y: left
   - Z: up
   - Right-handed system

This module provides transformations between these coordinate systems.
"""

import numpy as np
from typing import Tuple
from kitti_eval import BBox3D


class CoordinateTransform:
    """KITTI coordinate system transformations"""

    @staticmethod
    def camera_to_velodyne(x_cam: float, y_cam: float, z_cam: float) -> Tuple[float, float, float]:
        """
        Transform from camera coordinates to Velodyne (LiDAR) coordinates

        Camera:   X=right, Y=down,  Z=forward
        Velodyne: X=forward, Y=left, Z=up

        Transformation:
            X_vel =  Z_cam  (camera forward → velodyne forward)
            Y_vel = -X_cam  (camera right → velodyne left, negated)
            Z_vel = -Y_cam  (camera down → velodyne up, negated)

        Args:
            x_cam, y_cam, z_cam: Coordinates in camera system

        Returns:
            (x_vel, y_vel, z_vel): Coordinates in Velodyne system
        """
        x_vel = z_cam
        y_vel = -x_cam
        z_vel = -y_cam

        return x_vel, y_vel, z_vel

    @staticmethod
    def velodyne_to_camera(x_vel: float, y_vel: float, z_vel: float) -> Tuple[float, float, float]:
        """
        Transform from Velodyne (LiDAR) coordinates to camera coordinates

        Velodyne: X=forward, Y=left, Z=up
        Camera:   X=right, Y=down,  Z=forward

        Transformation (inverse of camera_to_velodyne):
            X_cam = -Y_vel  (velodyne left → camera right, negated)
            Y_cam = -Z_vel  (velodyne up → camera down, negated)
            Z_cam =  X_vel  (velodyne forward → camera forward)

        Args:
            x_vel, y_vel, z_vel: Coordinates in Velodyne system

        Returns:
            (x_cam, y_cam, z_cam): Coordinates in camera system
        """
        x_cam = -y_vel
        y_cam = -z_vel
        z_cam = x_vel

        return x_cam, y_cam, z_cam

    @staticmethod
    def transform_bbox_camera_to_velodyne(bbox_cam: BBox3D) -> BBox3D:
        """
        Transform a 3D bounding box from camera to Velodyne coordinates

        Args:
            bbox_cam: BBox3D in camera coordinates

        Returns:
            BBox3D in Velodyne coordinates
        """
        # Transform center position
        x_vel, y_vel, z_vel = CoordinateTransform.camera_to_velodyne(
            bbox_cam.x, bbox_cam.y, bbox_cam.z
        )

        # Transform dimensions
        # Camera: (w, l, h) corresponds to (width=X, length=Z, height=Y)
        # Velodyne: Need to remap to (width=Y, length=X, height=Z)
        w_cam, l_cam, h_cam = bbox_cam.w, bbox_cam.l, bbox_cam.h

        # Remap dimensions:
        # - Camera width (X-axis) → Velodyne width (Y-axis)
        # - Camera length (Z-axis) → Velodyne length (X-axis)
        # - Camera height (Y-axis) → Velodyne height (Z-axis)
        w_vel = w_cam  # width: X-axis (camera) → Y-axis (velodyne)
        l_vel = l_cam  # length: Z-axis (camera) → X-axis (velodyne)
        h_vel = h_cam  # height: Y-axis (camera) → Z-axis (velodyne)

        # Transform rotation
        # Camera: rotation around Y-axis (down)
        # Velodyne: need rotation around Z-axis (up)
        # Since Y_cam maps to -Z_vel, the rotation direction is preserved
        # but we need to account for the axis transformation
        ry_vel = -bbox_cam.ry  # Negate rotation due to coordinate flip

        # Create new BBox3D in Velodyne coordinates
        bbox_vel = BBox3D(
            x=x_vel,
            y=y_vel,
            z=z_vel,
            w=w_vel,
            l=l_vel,
            h=h_vel,
            ry=ry_vel,
            score=bbox_cam.score,
            class_name=bbox_cam.class_name
        )

        return bbox_vel

    @staticmethod
    def transform_bbox_velodyne_to_camera(bbox_vel: BBox3D) -> BBox3D:
        """
        Transform a 3D bounding box from Velodyne to camera coordinates

        Args:
            bbox_vel: BBox3D in Velodyne coordinates

        Returns:
            BBox3D in camera coordinates
        """
        # Transform center position
        x_cam, y_cam, z_cam = CoordinateTransform.velodyne_to_camera(
            bbox_vel.x, bbox_vel.y, bbox_vel.z
        )

        # Transform dimensions (inverse of camera_to_velodyne)
        w_vel, l_vel, h_vel = bbox_vel.w, bbox_vel.l, bbox_vel.h

        w_cam = w_vel
        l_cam = l_vel
        h_cam = h_vel

        # Transform rotation (inverse)
        ry_cam = -bbox_vel.ry

        # Create new BBox3D in camera coordinates
        bbox_cam = BBox3D(
            x=x_cam,
            y=y_cam,
            z=z_cam,
            w=w_cam,
            l=l_cam,
            h=h_cam,
            ry=ry_cam,
            score=bbox_vel.score,
            class_name=bbox_vel.class_name
        )

        return bbox_cam

    @staticmethod
    def transform_points_camera_to_velodyne(points_cam: np.ndarray) -> np.ndarray:
        """
        Transform point cloud from camera to Velodyne coordinates

        Args:
            points_cam: Nx3 or Nx4 array in camera coordinates

        Returns:
            points_vel: Nx3 or Nx4 array in Velodyne coordinates
        """
        points_vel = points_cam.copy()

        # Apply coordinate transformation to XYZ
        x_vel = points_cam[:, 2]   # Z_cam → X_vel (forward)
        y_vel = -points_cam[:, 0]  # -X_cam → Y_vel (left)
        z_vel = -points_cam[:, 1]  # -Y_cam → Z_vel (up)

        points_vel[:, 0] = x_vel
        points_vel[:, 1] = y_vel
        points_vel[:, 2] = z_vel

        # Keep intensity/other features if present (column 3+)

        return points_vel

    @staticmethod
    def transform_points_velodyne_to_camera(points_vel: np.ndarray) -> np.ndarray:
        """
        Transform point cloud from Velodyne to camera coordinates

        Args:
            points_vel: Nx3 or Nx4 array in Velodyne coordinates

        Returns:
            points_cam: Nx3 or Nx4 array in camera coordinates
        """
        points_cam = points_vel.copy()

        # Apply inverse coordinate transformation
        x_cam = -points_vel[:, 1]  # -Y_vel → X_cam (right)
        y_cam = -points_vel[:, 2]  # -Z_vel → Y_cam (down)
        z_cam = points_vel[:, 0]   # X_vel → Z_cam (forward)

        points_cam[:, 0] = x_cam
        points_cam[:, 1] = y_cam
        points_cam[:, 2] = z_cam

        # Keep intensity/other features if present (column 3+)

        return points_cam


def test_transformations():
    """Test coordinate transformations"""
    print("Testing KITTI Coordinate Transformations")
    print("=" * 80)

    # Test point transformation
    print("\n1. Point Transformation Test")
    print("-" * 80)

    # Camera: point at (1, -2, 10) = (right=1m, down=2m, forward=10m)
    point_cam = np.array([1.0, -2.0, 10.0])
    print(f"Camera coordinates:   X={point_cam[0]:6.2f} (right), "
          f"Y={point_cam[1]:6.2f} (down), Z={point_cam[2]:6.2f} (forward)")

    # Transform to Velodyne
    x_vel, y_vel, z_vel = CoordinateTransform.camera_to_velodyne(*point_cam)
    print(f"Velodyne coordinates: X={x_vel:6.2f} (forward), "
          f"Y={y_vel:6.2f} (left), Z={z_vel:6.2f} (up)")

    # Verify:
    # Camera (1, -2, 10) should map to Velodyne (10, -1, 2)
    # X_vel = 10 (forward from camera's Z)
    # Y_vel = -1 (left, negated from camera's right)
    # Z_vel = 2 (up, negated from camera's down)

    # Transform back
    x_cam_back, y_cam_back, z_cam_back = CoordinateTransform.velodyne_to_camera(x_vel, y_vel, z_vel)
    print(f"Back to camera:       X={x_cam_back:6.2f}, Y={y_cam_back:6.2f}, Z={z_cam_back:6.2f}")
    print(f"Round-trip error:     {np.linalg.norm(point_cam - np.array([x_cam_back, y_cam_back, z_cam_back])):.6f}")

    # Test bounding box transformation
    print("\n2. Bounding Box Transformation Test")
    print("-" * 80)

    # Camera coordinates: Car at (5, 1.5, 20) with size (1.8, 4.5, 1.6) and rotation 0.5
    bbox_cam = BBox3D(
        x=5.0,      # 5m to the right
        y=1.5,      # 1.5m down (below camera)
        z=20.0,     # 20m forward
        w=1.8,      # width (X-axis)
        l=4.5,      # length (Z-axis)
        h=1.6,      # height (Y-axis)
        ry=0.5,     # rotation around Y-axis
        class_name='Car'
    )

    print("Camera BBox:")
    print(f"  Position: ({bbox_cam.x:.2f}, {bbox_cam.y:.2f}, {bbox_cam.z:.2f})")
    print(f"  Size (w,l,h): ({bbox_cam.w:.2f}, {bbox_cam.l:.2f}, {bbox_cam.h:.2f})")
    print(f"  Rotation: {bbox_cam.ry:.3f} rad")

    # Transform to Velodyne
    bbox_vel = CoordinateTransform.transform_bbox_camera_to_velodyne(bbox_cam)

    print("\nVelodyne BBox:")
    print(f"  Position: ({bbox_vel.x:.2f}, {bbox_vel.y:.2f}, {bbox_vel.z:.2f})")
    print(f"  Size (w,l,h): ({bbox_vel.w:.2f}, {bbox_vel.l:.2f}, {bbox_vel.h:.2f})")
    print(f"  Rotation: {bbox_vel.ry:.3f} rad")

    # Transform back
    bbox_cam_back = CoordinateTransform.transform_bbox_velodyne_to_camera(bbox_vel)

    print("\nBack to Camera BBox:")
    print(f"  Position: ({bbox_cam_back.x:.2f}, {bbox_cam_back.y:.2f}, {bbox_cam_back.z:.2f})")
    print(f"  Size (w,l,h): ({bbox_cam_back.w:.2f}, {bbox_cam_back.l:.2f}, {bbox_cam_back.h:.2f})")
    print(f"  Rotation: {bbox_cam_back.ry:.3f} rad")

    pos_error = np.linalg.norm([bbox_cam.x - bbox_cam_back.x,
                                 bbox_cam.y - bbox_cam_back.y,
                                 bbox_cam.z - bbox_cam_back.z])
    print(f"\nRound-trip position error: {pos_error:.6f}")

    print("\n" + "=" * 80)
    print("Transformation tests completed!")


if __name__ == "__main__":
    test_transformations()
