import numpy as np
import numpy.typing as npt
from typing import Optional, List
import open3d as o3d
import os

from spaal2.core import PreciseDuration, MeasurementConfig, VeloPoint

class PcdLidarVLP16AmplitudeAuth:
    """
    A VLP-16 LiDAR simulator that reads from PCD files and uses a multi-pulse
    amplitude authentication mechanism to reject spoofed signals.
    """
    vertical_angles: list[int] = [-15, 1, -13, 3, -11, 5, -9, 7, -7, 9, -5, 11, -3, 13, -1, 15]

    def __init__(self, pcd_file_path: Optional[str], lidar_position: np.ndarray, lidar_rotation: np.ndarray, base_timestamp: PreciseDuration = PreciseDuration(nanoseconds=0), amplitude: float = 1.0, pulse_width: PreciseDuration = PreciseDuration(nanoseconds=10), time_resolution_ns: float = 1.0) -> None:
        self.index: int = 0
        self.max_index: int = int(360 // 0.2 * 16)
        self.accept_window = PreciseDuration(nanoseconds=800)
        self.base_timestamp = base_timestamp
        self.amplitude = amplitude
        self.pulse_width = pulse_width
        self.time_resolution_ns = time_resolution_ns
        self.lidar_position = lidar_position
        self.lidar_rotation = lidar_rotation
        self.pcd_files: list[str] = []
        self.pcd_file_path = pcd_file_path

        # --- Amplitude Authentication Parameters ---
        self.auth_amplitude_ratios: List[float] = [1.0, 0.6]
        self.auth_pulse_interval_ns: float = 20.0
        # ------------------------------------------

        if pcd_file_path:
            if not os.path.exists(pcd_file_path):
                 raise FileNotFoundError(f"PCD file not found at {pcd_file_path}")
            self._read_pcd(pcd_file_path)
        else:
            self.points = np.array([])
            self.all_point_indices = set()
            self.initial_azimuth_offset = 0
            self.depth_map = {}
            self.original_point_indices_map = {}
            self.no_signal_scan_angles = []
        
        self.detected_point_indices = set()

    def _read_pcd(self, file_path: str):
        self.point_cloud = o3d.io.read_point_cloud(file_path)
        self.points = np.asarray(self.point_cloud.points)
        self.all_point_indices = set(range(len(self.points)))
        if len(self.points) > 0:
            first_point = self.points[0]
            self.initial_azimuth_offset = np.rad2deg(np.arctan2(first_point[1], first_point[0]))
        else:
            self.initial_azimuth_offset = 0
        self.depth_map, self.original_point_indices_map = self._create_depth_map()
        self.no_signal_scan_angles: list[tuple[int, int]] = []

    def set_pcd_files(self, pcd_files: list[str]):
        self.pcd_files = pcd_files

    def _create_depth_map(self):
        depth_map = {}
        original_point_indices_map = {}
        horizontal_resolution = 20
        for i, point in enumerate(self.points):
            x, y, z = point
            if np.linalg.norm([x, y, z]) == 0:
                continue
            depth = np.linalg.norm(point)
            azimuth_deg = np.rad2deg(np.arctan2(y, x))
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

    def new_frame(self, frame_num: int = 0, base_timestamp: PreciseDuration = PreciseDuration(nanoseconds=0)) -> "PcdLidarVLP16AmplitudeAuth":
        if not self.pcd_files:
            if self.pcd_file_path:
                self.pcd_files.append(self.pcd_file_path)
            else:
                raise ValueError("PCD file list is not set and no initial pcd_file_path was provided.")
        pcd_file_to_load = self.pcd_files[frame_num % len(self.pcd_files)]
        self._read_pcd(pcd_file_to_load)
        self.index = 0
        self.base_timestamp = base_timestamp
        self.detected_point_indices = set()
        return self

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
            pulse_width_indices = int(self.pulse_width.in_nanoseconds / self.time_resolution_ns)
            interval_indices = int(self.auth_pulse_interval_ns / self.time_resolution_ns)

            for i, ratio in enumerate(self.auth_amplitude_ratios):
                pulse_amplitude = self.amplitude * ratio
                pulse_start_index = time_of_flight_index + (i * interval_indices)
                if pulse_start_index + pulse_width_indices < signal_length:
                    signal[pulse_start_index : pulse_start_index + pulse_width_indices] = pulse_amplitude
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
        # --- Amplitude Ratio Verification Logic ---
        raises = np.flatnonzero((signal[:-1] < 0.01) & (signal[1:] >= 0.01)) + 1
        if len(raises) < len(self.auth_amplitude_ratios):
            return []

        peaks = {r: np.max(signal[r:min(len(signal), r + 50)]) for r in raises}
        peak_times = sorted(peaks.keys())
        
        interval_indices = int(self.auth_pulse_interval_ns / self.time_resolution_ns)
        tolerance_indices = int(interval_indices * 0.2) # 20% tolerance for timing
        
        valid_points = []
        used_peak_times = set()

        for i in range(len(peak_times)):
            if peak_times[i] in used_peak_times:
                continue

            # Find a sequence of peaks with the correct timing
            sequence_times = [peak_times[i]]
            for j in range(1, len(self.auth_amplitude_ratios)):
                expected_next_time = sequence_times[-1] + interval_indices
                found_next = False
                for k in range(i + 1, len(peak_times)):
                    if peak_times[k] in used_peak_times:
                        continue
                    if abs(peak_times[k] - expected_next_time) <= tolerance_indices:
                        sequence_times.append(peak_times[k])
                        found_next = True
                        break
                if not found_next:
                    break
            
            # If a full sequence is found, check amplitude ratios
            if len(sequence_times) == len(self.auth_amplitude_ratios):
                base_amplitude = peaks[sequence_times[0]]
                if base_amplitude < 0.1: continue # Avoid division by zero / noise

                ratios_match = True
                for k in range(1, len(self.auth_amplitude_ratios)):
                    expected_ratio = self.auth_amplitude_ratios[k] / self.auth_amplitude_ratios[0]
                    actual_ratio = peaks[sequence_times[k]] / base_amplitude
                    if abs(actual_ratio - expected_ratio) > 0.15: # 15% tolerance for ratio
                        ratios_match = False
                        break
                
                if ratios_match:
                    # This is a valid point, create VeloPoint
                    for t in sequence_times:
                        used_peak_times.add(t)

                    highest_peak_time = sequence_times[0]
                    highest_peak = base_amplitude
                    intensity = int(min(highest_peak * 255, 255))
                    distance_m = (highest_peak_time * self.time_resolution_ns) * 0.15
                    alpha = np.deg2rad(config.azimuth / 100.0)
                    omega = np.deg2rad(config.altitude / 100.0)
                    x = distance_m * np.cos(omega) * np.cos(alpha)
                    y = distance_m * np.cos(omega) * np.sin(alpha)
                    z = distance_m * np.sin(omega)

                    valid_points.append(VeloPoint(
                        intensity=intensity, channel=0,
                        timestamp=config.start_timestamp.in_nanoseconds,
                        azimuth=config.azimuth, altitude=config.altitude,
                        distance_m=distance_m, x=x, y=y, z=z,
                    ))
        return valid_points

    def get_point_count(self) -> int:
        return self.max_index
