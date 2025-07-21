import numpy as np
import numpy.typing as npt
from typing import Optional
import open3d as o3d

from spaal2.core import PreciseDuration, MeasurementConfig, VeloPoint

class PcdLidarVLP16:
    vertical_angles: list[int] = [-15, 1, -13, 3, -11, 5, -9, 7, -7, 9, -5, 11, -3, 13, -1, 15]

    def __init__(self, pcd_file_path: str, lidar_position: np.ndarray, lidar_rotation: np.ndarray, base_timestamp: PreciseDuration = PreciseDuration(nanoseconds=0), amplitude: float = 1.0, pulse_width: PreciseDuration = PreciseDuration(nanoseconds=10), time_resolution_ns: float = 1.0) -> None:
        self.index: int = 0
        self.max_index: int = int(360 // 0.2 * 16)
        self.accept_window = PreciseDuration(nanoseconds=800)
        self.base_timestamp = base_timestamp
        self.amplitude = amplitude
        self.pulse_width = pulse_width
        self.time_resolution_ns = time_resolution_ns
        self.lidar_position = lidar_position
        self.lidar_rotation = lidar_rotation

        self.point_cloud = o3d.io.read_point_cloud(pcd_file_path)
        self.points = np.asarray(self.point_cloud.points)

        self.all_point_indices = set(range(len(self.points)))
        self.detected_point_indices = set()

        # Set the initial azimuth offset so the scan starts at the first point's angle
        first_point = self.points[0]
        self.initial_azimuth_offset = np.rad2deg(np.arctan2(first_point[1], first_point[0]))
        print(f"Initial azimuth offset: {self.initial_azimuth_offset} degrees")

        self.depth_map, self.original_point_indices_map = self._create_depth_map()
        self.no_signal_scan_angles: list[tuple[int, int]] = [] # Store (azimuth, altitude) for no signal

    def _create_depth_map(self):
        depth_map = {}
        original_point_indices_map = {}
        # Discretize azimuth and altitude to match LiDAR's scanning pattern
        horizontal_resolution = 20  # 0.2 degrees in hundredths

        for i, point in enumerate(self.points):
            x, y, z = point
            if np.linalg.norm([x, y, z]) == 0:
                continue

            depth = np.linalg.norm(point)
            # Y軸を0度とする座標系に合わせるため、arctan2の引数を(x, y)に変更
            azimuth_deg = np.rad2deg(np.arctan2(x, y))
            altitude_deg = np.rad2deg(np.arcsin(z / depth))

            # Find the closest vertical angle in the VLP-16 spec
            closest_vertical_angle = min(self.vertical_angles, key=lambda v_angle: abs(v_angle - altitude_deg))

            # Discretize the azimuth to the LiDAR's horizontal resolution
            # and handle the wrap-around at 360 degrees
            azimuth_index = round(azimuth_deg * 100 / horizontal_resolution)
            discretized_azimuth = (azimuth_index * horizontal_resolution)

            azimuth_key = int(discretized_azimuth % 36000)
            altitude_key = int(closest_vertical_angle * 100)

            # If the key already exists, only update if the new point is closer
            if (azimuth_key, altitude_key) not in depth_map or depth < depth_map[(azimuth_key, altitude_key)]:
                depth_map[(azimuth_key, altitude_key)] = depth
                original_point_indices_map[(azimuth_key, altitude_key)] = i

        return depth_map, original_point_indices_map

    def set_amplitude(self, amplitude: float):
        self.amplitude = amplitude
        print(f"LiDAR: VLP16 (PCD)")

    def new_frame(self, base_timestamp: PreciseDuration) -> "PcdLidarVLP16":
        return PcdLidarVLP16(
            pcd_file_path="",
            lidar_position=self.lidar_position,
            lidar_rotation=self.lidar_rotation,
            base_timestamp=base_timestamp,
            amplitude=self.amplitude,
            pulse_width=self.pulse_width,
            time_resolution_ns=self.time_resolution_ns
        )

    def _get_current_angle(self) -> tuple[int, int]:
        horizontal_index = self.index // 16
        vertical_index = self.index % 16
        altitude = self._vertical_index_to_altitude(vertical_index)

        relative_azimuth_deg = horizontal_index * 0.2
        absolute_azimuth_deg = relative_azimuth_deg + self.initial_azimuth_offset

        horizontal_resolution = 20
        azimuth_index = round(absolute_azimuth_deg * 100 / horizontal_resolution)
        discretized_azimuth = (azimuth_index * horizontal_resolution)

        lookup_key = int(discretized_azimuth % 36000)

        return lookup_key, altitude

    def _vertical_index_to_altitude(self, index: int) -> int:
        return self.vertical_angles[index] * 100

    def _get_current_timestamp(self) -> int:
        horizontal_index = self.index // 16
        vertical_index = self.index % 16
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
                signal[raises[i]:min(len(signal), raises[i] + 10)]
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