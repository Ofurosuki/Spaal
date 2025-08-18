import numpy as np
import numpy.typing as npt
from typing import Optional
import open3d as o3d
import os

from spaal2.core import PreciseDuration, MeasurementConfig, VeloPoint

class FireAngle:
    def __init__(self, v_angle: float, h_offset: float) -> None:
        self.v_angle = v_angle
        self.h_offset = h_offset

class PcdLidarVLP32c:
    fire_angles: list[FireAngle] = [
        FireAngle(-25, 1.4), FireAngle(-1, -4.2),
        FireAngle(-1.667, 1.4), FireAngle(-15.639, -1.4),
        FireAngle(-11.31, 1.4), FireAngle(0, -1.4),
        FireAngle(-0.667, 4.2), FireAngle(-8.843, -1.4),
        FireAngle(-7.254, 1.4), FireAngle(0.333, -4.2),
        FireAngle(-0.333, 1.4), FireAngle(-6.148, -1.4),
        FireAngle(-5.333, 4.2), FireAngle(1.333, -1.4),
        FireAngle(0.667, 4.2), FireAngle(-4, -1.4),
        FireAngle(-4.667, 1.4), FireAngle(1.667, -4.2),
        FireAngle(1, 1.4), FireAngle(-3.667, -4.2),
        FireAngle(-3.333, 4.2), FireAngle(3.333, -1.4),
        FireAngle(2.333, 1.4), FireAngle(-2.667, -1.4),
        FireAngle(-3, 1.4), FireAngle(7, -1.4),
        FireAngle(4.667, 1.4), FireAngle(-2.333, -4.2),
        FireAngle(-2, 4.2), FireAngle(15, -1.4),
        FireAngle(10.333, 1.4), FireAngle(-1.333, -1.4)
    ]
    vertical_angles: list[float] = [fa.v_angle for fa in fire_angles]


    def __init__(self, pcd_file_path: str, lidar_position: np.ndarray, lidar_rotation: np.ndarray, base_timestamp: PreciseDuration = PreciseDuration(nanoseconds=0), amplitude: float = 1.0, pulse_width: PreciseDuration = PreciseDuration(nanoseconds=10), time_resolution_ns: float = 1.0) -> None:
        self.index: int = 0
        self.max_index: int = int(360 // 0.2 * 32)
        self.accept_window = PreciseDuration(nanoseconds=800)
        self.base_timestamp = base_timestamp
        self.amplitude = amplitude
        self.pulse_width = pulse_width
        self.time_resolution_ns = time_resolution_ns
        self.lidar_position = lidar_position
        self.lidar_rotation = lidar_rotation
        self.pcd_file_path = pcd_file_path

        if not os.path.exists(pcd_file_path) and pcd_file_path:
             raise FileNotFoundError(f"PCD file not found at {pcd_file_path}")
        
        if os.path.exists(pcd_file_path):
            self.point_cloud = o3d.io.read_point_cloud(pcd_file_path)
        else:
            self.point_cloud = o3d.geometry.PointCloud()

        self.points = np.asarray(self.point_cloud.points)

        self.all_point_indices = set(range(len(self.points)))
        self.detected_point_indices = set()

        if len(self.points) > 0:
            first_point = self.points[0]
            self.initial_azimuth_offset = np.rad2deg(np.arctan2(first_point[1], first_point[0]))
            print(f"Initial azimuth offset: {self.initial_azimuth_offset} degrees") 
        else:
            self.initial_azimuth_offset = 0

        self.depth_map, self.original_point_indices_map = self._create_depth_map()
        self.no_signal_scan_angles: list[tuple[int, int]] = []

    def _create_depth_map(self):
        depth_map = {}
        original_point_indices_map = {}
        horizontal_resolution = 20
        
        for i, point in enumerate(self.points):
            x, y, z = point
            if np.linalg.norm([x, y, z]) == 0:
                continue
            
            depth = np.linalg.norm(point)
            azimuth_deg = np.rad2deg(np.arctan2(x, y))
            altitude_deg = np.rad2deg(np.arcsin(z / depth))

            closest_vertical_angle = min(self.vertical_angles, key=lambda v_angle: abs(v_angle - altitude_deg))
            
            azimuth_index = round(azimuth_deg * 100 / horizontal_resolution)
            discretized_azimuth = (azimuth_index * horizontal_resolution)

            azimuth_key = int(discretized_azimuth % 36000)
            altitude_key = int(closest_vertical_angle * 100)

            if (azimuth_key, altitude_key) not in depth_map or depth < depth_map[(azimuth_key, altitude_key)]:
                depth_map[(azimuth_key, altitude_key)] = depth
                original_point_indices_map[(azimuth_key, altitude_key)] = i
                
        return depth_map, original_point_indices_map

    def set_amplitude(self, amplitude: float):
        self.amplitude = amplitude

    def new_frame(self, base_timestamp: PreciseDuration) -> "PcdLidarVLP32c":
        return PcdLidarVLP32c(
            pcd_file_path=self.pcd_file_path,
            lidar_position=self.lidar_position, 
            lidar_rotation=self.lidar_rotation,
            base_timestamp=base_timestamp, 
            amplitude=self.amplitude, 
            pulse_width=self.pulse_width,
            time_resolution_ns=self.time_resolution_ns
        )

    def _get_current_angle(self) -> tuple[int, int]:
        horizontal_index = self.index // 32
        vertical_index = self.index % 32
        fire_angle = self.fire_angles[vertical_index]
        
        lidar_azimuth_deg = horizontal_index * 0.2 + fire_angle.h_offset
        
        absolute_azimuth_deg = lidar_azimuth_deg + self.initial_azimuth_offset
        
        altitude = int(fire_angle.v_angle * 100)

        horizontal_resolution = 20
        azimuth_index = round(absolute_azimuth_deg * 100 / horizontal_resolution)
        discretized_azimuth = (azimuth_index * horizontal_resolution)
        
        lookup_key = int(discretized_azimuth % 36000)

        return lookup_key, altitude

    def _get_current_timestamp(self) -> int:
        horizontal_index = self.index // 32
        vertical_index = int((self.index % 32) // 2)
        return horizontal_index * 55296 + vertical_index * 2304 + self.base_timestamp.in_nanoseconds

    def scan(self) -> tuple[MeasurementConfig, npt.NDArray[np.float64]]:
        if self.index >= self.max_index:
            raise StopIteration()
        azimuth, altitude = self._get_current_angle()
        timestamp = self._get_current_timestamp()

        depth = self.depth_map.get((azimuth, altitude))
        
        signal_length = int(self.accept_window.in_nanoseconds / self.time_resolution_ns)
        signal = np.zeros((signal_length, ))

        if depth is not None:
            point_idx = self.original_point_indices_map.get((azimuth, altitude))
            if point_idx is not None:
                self.detected_point_indices.add(point_idx)
            
            time_of_flight_index = int(depth / (0.15 * self.time_resolution_ns))
            if time_of_flight_index < signal_length:
                pulse_width_indices = int(self.pulse_width.in_nanoseconds / self.time_resolution_ns)
                signal[time_of_flight_index:time_of_flight_index + pulse_width_indices] = self.amplitude
        else:
            self.no_signal_scan_angles.append((azimuth, altitude))

        self.index += 1
        return MeasurementConfig(
            start_timestamp=PreciseDuration(nanoseconds=timestamp),
            accept_duration=self.accept_window,
            azimuth=azimuth,
            altitude=altitude,
        ), signal

    def get_no_signal_points(self) -> np.ndarray:
        undetected_indices = self.all_point_indices - self.detected_point_indices
        if not undetected_indices:
            return np.array([])
        return self.points[list(undetected_indices)]

    def receive(self, config: MeasurementConfig, signal: npt.NDArray[np.float64]) -> list[VeloPoint]:
        raises = np.flatnonzero(
            (signal[:-1] < 0.01) & (signal[1:] >= 0.01)
        ) + 1
        
        if len(raises) == 0:
            return []

        peaks = np.empty_like(raises, dtype=np.float64)
        for i in range(len(raises)):
            peaks[i] = np.max(
                signal[raises[i]:min(len(signal), raises[i] + 50)]
            )

        highest_peak_index = np.argmax(peaks)
        highest_peak = peaks[highest_peak_index]
        highest_peak_time = raises[highest_peak_index]

        intensity = int(min(highest_peak * 255, 255))
        distance_m = (highest_peak_time * self.time_resolution_ns) * 0.15
        alpha = np.deg2rad(config.azimuth / 100.0)
        omega = np.deg2rad(config.altitude / 100.0)
        x = distance_m * np.sin(alpha) * np.cos(omega)
        y = distance_m * np.cos(alpha) * np.cos(omega)
        z = distance_m * np.sin(omega)

        return [
            VeloPoint(
                intensity=intensity,
                channel=0,
                timestamp=config.start_timestamp.in_nanoseconds,
                azimuth=config.azimuth,
                altitude=config.altitude,
                distance_m=distance_m,
                x=x,
                y=y,
                z=z,
            )
        ]

    def get_point_count(self) -> int:
        return self.max_index