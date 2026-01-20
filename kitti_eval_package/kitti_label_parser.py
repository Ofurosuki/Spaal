"""
KITTI Label File Parser

Parse KITTI official label format (.txt files) for 3D object detection evaluation.

KITTI Label Format (15 values per object):
type truncated occluded alpha bbox_left bbox_top bbox_right bbox_bottom
dimensions_h dimensions_w dimensions_l location_x location_y location_z rotation_y

Additional 16th value for detections: score
"""

import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

from kitti_eval import BBox3D, Detection, GroundTruth
from coordinate_transform import CoordinateTransform


class KITTILabelParser:
    """Parser for KITTI label files"""

    # KITTI difficulty criteria
    MIN_HEIGHT = {'EASY': 40, 'MODERATE': 25, 'HARD': 25}
    MAX_OCCLUSION = {'EASY': 0, 'MODERATE': 1, 'HARD': 2}
    MAX_TRUNCATION = {'EASY': 0.15, 'MODERATE': 0.3, 'HARD': 0.5}

    @staticmethod
    def parse_label_line(line: str, is_detection: bool = False) -> Dict:
        """
        Parse a single line from KITTI label file

        Args:
            line: Line from label file
            is_detection: Whether this is a detection (has score) or ground truth

        Returns:
            Dictionary with parsed values
        """
        parts = line.strip().split()

        if len(parts) < 15:
            return None

        obj = {}

        # Type
        obj['type'] = parts[0]

        # Truncation (0.0 to 1.0)
        obj['truncation'] = float(parts[1])

        # Occlusion (0, 1, 2, 3)
        obj['occlusion'] = int(parts[2])

        # Alpha (observation angle)
        obj['alpha'] = float(parts[3])

        # 2D bounding box (left, top, right, bottom)
        obj['bbox_2d'] = [
            float(parts[4]),  # left
            float(parts[5]),  # top
            float(parts[6]),  # right
            float(parts[7])   # bottom
        ]

        # 3D dimensions (height, width, length) in meters
        obj['h'] = float(parts[8])
        obj['w'] = float(parts[9])
        obj['l'] = float(parts[10])

        # 3D location (x, y, z) in camera coordinates
        obj['x'] = float(parts[11])
        obj['y'] = float(parts[12])
        obj['z'] = float(parts[13])

        # Rotation around Y-axis
        obj['ry'] = float(parts[14])

        # Score (for detections only)
        if is_detection and len(parts) >= 16:
            obj['score'] = float(parts[15])
        else:
            obj['score'] = 1.0

        return obj

    @staticmethod
    def determine_difficulty(obj: Dict) -> str:
        """
        Determine difficulty level based on KITTI criteria

        Args:
            obj: Parsed object dictionary

        Returns:
            Difficulty level: 'EASY', 'MODERATE', 'HARD', or 'IGNORE'
        """
        # Get 2D bbox height
        bbox_height = obj['bbox_2d'][3] - obj['bbox_2d'][1]

        # Check if object should be ignored (DontCare or too small)
        if obj['type'] == 'DontCare' or bbox_height < KITTILabelParser.MIN_HEIGHT['HARD']:
            return 'IGNORE'

        # Determine difficulty
        truncation = obj['truncation']
        occlusion = obj['occlusion']

        if (bbox_height >= KITTILabelParser.MIN_HEIGHT['EASY'] and
            occlusion <= KITTILabelParser.MAX_OCCLUSION['EASY'] and
            truncation <= KITTILabelParser.MAX_TRUNCATION['EASY']):
            return 'EASY'
        elif (bbox_height >= KITTILabelParser.MIN_HEIGHT['MODERATE'] and
              occlusion <= KITTILabelParser.MAX_OCCLUSION['MODERATE'] and
              truncation <= KITTILabelParser.MAX_TRUNCATION['MODERATE']):
            return 'MODERATE'
        elif (bbox_height >= KITTILabelParser.MIN_HEIGHT['HARD'] and
              occlusion <= KITTILabelParser.MAX_OCCLUSION['HARD'] and
              truncation <= KITTILabelParser.MAX_TRUNCATION['HARD']):
            return 'HARD'
        else:
            return 'IGNORE'

    @staticmethod
    def camera_to_lidar_coords(x: float, y: float, z: float, h: float) -> Tuple[float, float, float]:
        """
        Convert from KITTI camera coordinates to Velodyne (LiDAR) coordinates

        KITTI camera: X=right, Y=down, Z=forward
        Velodyne:     X=forward, Y=left, Z=up

        Note: Camera Y-coordinate represents the bottom of the object,
        so we adjust by h/2 to get the center.

        Args:
            x, y, z: Camera coordinates (y is bottom of object)
            h: Object height

        Returns:
            (x_lidar, y_lidar, z_lidar) in Velodyne coordinates (center of object)
        """
        # Adjust camera Y to center of object (from bottom)
        y_center = y - h / 2

        # Use CoordinateTransform for the conversion
        x_lidar, y_lidar, z_lidar = CoordinateTransform.camera_to_velodyne(x, y_center, z)

        return x_lidar, y_lidar, z_lidar

    @staticmethod
    def parse_ground_truth_file(filepath: str, coordinate_system: str = 'camera') -> List[GroundTruth]:
        """
        Parse KITTI ground truth label file

        Args:
            filepath: Path to label .txt file
            coordinate_system: 'camera' or 'lidar'

        Returns:
            List of GroundTruth objects
        """
        ground_truths = []

        if not os.path.exists(filepath):
            return ground_truths

        with open(filepath, 'r') as f:
            for line in f:
                obj = KITTILabelParser.parse_label_line(line.strip(), is_detection=False)

                if obj is None:
                    continue

                # Skip DontCare objects
                if obj['type'] == 'DontCare':
                    continue

                # Determine difficulty
                difficulty = KITTILabelParser.determine_difficulty(obj)

                if difficulty == 'IGNORE':
                    continue

                # Convert coordinates if needed
                if coordinate_system == 'lidar':
                    x, y, z = KITTILabelParser.camera_to_lidar_coords(
                        obj['x'], obj['y'], obj['z'], obj['h']
                    )
                else:
                    x, y, z = obj['x'], obj['y'], obj['z']

                # Create BBox3D
                bbox = BBox3D(
                    x=x,
                    y=y,
                    z=z,
                    w=obj['w'],
                    l=obj['l'],
                    h=obj['h'],
                    ry=obj['ry'],
                    class_name=obj['type']
                )

                # Create GroundTruth
                gt = GroundTruth(
                    bbox=bbox,
                    class_name=obj['type'],
                    difficulty=difficulty,
                    truncation=obj['truncation'],
                    occlusion=obj['occlusion']
                )

                ground_truths.append(gt)

        return ground_truths

    @staticmethod
    def parse_detection_file(filepath: str, coordinate_system: str = 'camera') -> List[Detection]:
        """
        Parse KITTI detection result file

        Args:
            filepath: Path to detection .txt file
            coordinate_system: 'camera' or 'lidar'

        Returns:
            List of Detection objects
        """
        detections = []

        if not os.path.exists(filepath):
            return detections

        with open(filepath, 'r') as f:
            for line in f:
                obj = KITTILabelParser.parse_label_line(line.strip(), is_detection=True)

                if obj is None:
                    continue

                # Skip DontCare
                if obj['type'] == 'DontCare':
                    continue

                # Convert coordinates if needed
                if coordinate_system == 'lidar':
                    x, y, z = KITTILabelParser.camera_to_lidar_coords(
                        obj['x'], obj['y'], obj['z'], obj['h']
                    )
                else:
                    x, y, z = obj['x'], obj['y'], obj['z']

                # Create BBox3D
                bbox = BBox3D(
                    x=x,
                    y=y,
                    z=z,
                    w=obj['w'],
                    l=obj['l'],
                    h=obj['h'],
                    ry=obj['ry'],
                    score=obj['score'],
                    class_name=obj['type']
                )

                # Create Detection
                det = Detection(
                    bbox=bbox,
                    score=obj['score'],
                    class_name=obj['type']
                )

                detections.append(det)

        return detections

    @staticmethod
    def load_all_labels(
        label_dir: str,
        is_detection: bool = False,
        coordinate_system: str = 'camera',
        file_indices: List[int] = None
    ) -> Dict[str, List]:
        """
        Load all label files from a directory

        Args:
            label_dir: Directory containing label .txt files
            is_detection: Whether files contain detections (with scores)
            coordinate_system: 'camera' or 'lidar'
            file_indices: List of file indices to load (e.g., [0, 1, 2])
                         If None, loads all files

        Returns:
            Dictionary mapping sample_id -> List[GroundTruth/Detection]
        """
        labels = {}

        # Get all .txt files
        label_files = sorted(Path(label_dir).glob('*.txt'))

        for label_file in label_files:
            # Get sample ID from filename (e.g., '000000.txt' -> '000000')
            sample_id = label_file.stem

            # Check if this index should be loaded
            if file_indices is not None:
                try:
                    file_idx = int(sample_id)
                    if file_idx not in file_indices:
                        continue
                except ValueError:
                    continue

            # Parse file
            if is_detection:
                labels[sample_id] = KITTILabelParser.parse_detection_file(
                    str(label_file), coordinate_system
                )
            else:
                labels[sample_id] = KITTILabelParser.parse_ground_truth_file(
                    str(label_file), coordinate_system
                )

        return labels


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python kitti_label_parser.py <path_to_label_file>")
        sys.exit(1)

    filepath = sys.argv[1]

    print(f"Parsing: {filepath}")
    print("\nGround Truth Objects:")
    print("-" * 80)

    gts = KITTILabelParser.parse_ground_truth_file(filepath)

    for i, gt in enumerate(gts):
        print(f"{i+1}. {gt.class_name} [{gt.difficulty}]")
        print(f"   Location: ({gt.bbox.x:.2f}, {gt.bbox.y:.2f}, {gt.bbox.z:.2f})")
        print(f"   Size: {gt.bbox.w:.2f} x {gt.bbox.l:.2f} x {gt.bbox.h:.2f}")
        print(f"   Rotation: {gt.bbox.ry:.2f} rad")
        print(f"   Truncation: {gt.truncation:.2f}, Occlusion: {gt.occlusion}")
        print()

    print(f"\nTotal: {len(gts)} objects")
