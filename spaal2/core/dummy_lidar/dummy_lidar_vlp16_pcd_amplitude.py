import numpy as np
import numpy.typing as npt
from typing import Optional, Tuple, List
import open3d as o3d
import os
import time
from numpy.random import default_rng

from spaal2.core import PreciseDuration, MeasurementConfig, VeloPoint
from spaal2.core.noise_utils import detect_echo, accumulate_signal_list
from spaal2.core.dummy_lidar.echo import Echo
from spaal2.core.dummy_lidar.echogroup import EchoGroup

class PcdLidarVLP16Amplitude:
    vertical_angles: list[int] = [-15, 1, -13, 3, -11, 5, -9, 7, -7, 9, -5, 11, -3, 13, -1, 15]

    def __init__(
        self,
        pcd_file_path: Optional[str],
        lidar_position: np.ndarray,
        lidar_rotation: np.ndarray,
        base_timestamp: PreciseDuration = PreciseDuration(nanoseconds=0),
        amplitude: float = 1.0,
        pulse_width: PreciseDuration = PreciseDuration(nanoseconds=10),
        time_resolution_ns: float = 1.0,
        # Fingerprint parameters
        pulse_num: int = 1,
        min_interval: PreciseDuration = PreciseDuration(nanoseconds=100),
        max_interval: PreciseDuration = PreciseDuration(nanoseconds=200),
        consider_amp: bool = False,
        min_amp_diff_ratio: float = 0.2,
        max_amp_diff_ratio: float = 0.4,
        max_torelance_error: PreciseDuration = PreciseDuration(nanoseconds=5),
        max_amp_torelance_error: float = 0.1,
        num_accumulation: int = 1,
        acc_interval: PreciseDuration = PreciseDuration(nanoseconds=0),
        thd_factor: float = 0.5,
        use_height_estimation: bool = False,
        pulse_half_width: PreciseDuration = PreciseDuration(nanoseconds=5),
        # Fixed fingerprint parameters
        gt_intervals_ns: Optional[List[int]] = None,
        gt_amps_ratio: Optional[List[float]] = None,
    ):
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

        # Fingerprint attributes
        self.pulse_num = pulse_num
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.consider_amp = consider_amp
        self.min_amp_diff_ratio = min_amp_diff_ratio
        self.max_amp_diff_ratio = max_amp_diff_ratio
        self.max_torelance_error = max_torelance_error
        self.max_amp_torelance_error = max_amp_torelance_error
        self.num_accumulation = num_accumulation
        self.acc_interval = acc_interval
        self.thd_factor = thd_factor
        self.use_height_estimation = use_height_estimation
        self.pulse_half_width = pulse_half_width
        self.gt_intervals_ns = gt_intervals_ns
        self.gt_amps_ratio = gt_amps_ratio
        
        self.rng = default_rng()
        
        pulse_width_indices = int(self.pulse_width.in_nanoseconds / self.time_resolution_ns)
        self.pulse_shape = np.ones(pulse_width_indices) * self.amplitude

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
            if np.linalg.norm([x, y, z]) == 0: continue
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

    def new_frame(self, frame_num: int = 0, base_timestamp: PreciseDuration = PreciseDuration(nanoseconds=0)) -> "PcdLidarVLP16Amplitude":
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
        signal = np.zeros(signal_length)
        gt_intervals = []
        gt_amps_ratio = []

        if depth is not None:
            point_idx = self.original_point_indices_map.get((azimuth, altitude))
            if point_idx is not None: self.detected_point_indices.add(point_idx)

            time_of_flight_index = int(depth / (0.15 * self.time_resolution_ns))
            pulse_width_indices = int(self.pulse_width.in_nanoseconds / self.time_resolution_ns)
            base_signal = np.zeros(signal_length)
            if time_of_flight_index < signal_length:
                base_signal[time_of_flight_index:time_of_flight_index + pulse_width_indices] = self.pulse_shape

            if self.pulse_num <= 1:
                signal = base_signal
                gt_amps_ratio = [1.0]
            else:
                # Determine intervals
                if self.gt_intervals_ns:
                    gt_intervals = self.gt_intervals_ns
                else:
                    gt_intervals = np.random.randint(self.min_interval.in_nanoseconds, self.max_interval.in_nanoseconds, self.pulse_num - 1).cumsum().tolist()

                # Determine amplitude ratios
                if self.consider_amp:
                    if self.gt_amps_ratio:
                        gt_amps_ratio = self.gt_amps_ratio
                    else:
                        gt_amps_ratio = self.rng.permutation(np.hstack([1.0, self.rng.random(self.pulse_num - 1) * (self.max_amp_diff_ratio - self.min_amp_diff_ratio) + (1 - self.max_amp_diff_ratio)])).tolist()
                else:
                    gt_amps_ratio = np.ones(self.pulse_num).tolist()

                # Build signal
                signal = base_signal.copy() * gt_amps_ratio[0]
                for i in range(self.pulse_num - 1):
                    mean = time_of_flight_index + gt_intervals[i]
                    start = mean - pulse_width_indices // 2
                    end = start + pulse_width_indices
                    if start >= 0 and end < signal_length:
                        signal[start:end] += self.pulse_shape * gt_amps_ratio[i+1]
        else:
            self.no_signal_scan_angles.append((azimuth, altitude))

        self.index += 1
        return MeasurementConfig(
            start_timestamp=PreciseDuration(nanoseconds=timestamp),
            accept_duration=self.accept_window,
            azimuth=azimuth,
            altitude=altitude,
            gt_intervals=gt_intervals,
            consider_amp=self.consider_amp,
            gt_amps_ratio=gt_amps_ratio,
            amp_torelance_error_ratio=self.max_amp_torelance_error,
            torelance_error=self.max_torelance_error,
            num_accumulation=self.num_accumulation,
            accumulation_interval=self.acc_interval,
        ), signal

    def receive(self, config: MeasurementConfig, signal_list: list[npt.NDArray[np.float64]]) -> Tuple[list[VeloPoint], list[EchoGroup]]:
        if len(signal_list) == 1:
            signal = signal_list[0]
        else:
            signal = accumulate_signal_list(signal_list)
        
        max_height = self.amplitude * len(signal_list)
        effective_echoes = detect_echo(signal, max_height, self.thd_factor, self.use_height_estimation, self.pulse_half_width.in_nanoseconds)

        if not effective_echoes: return [], []

        pulse_num = len(config.gt_intervals) + 1
        if pulse_num < 2:
            effective_echoes_group = [EchoGroup([x]) for x in effective_echoes]
        else:
            certified_echoes: list[EchoGroup] = []
            error = config.torelance_error.in_nanoseconds
            peaks = np.array([x.peak_position for x in effective_echoes])
            
            for i in range(len(effective_echoes) - pulse_num + 1):
                if peaks[i] + config.gt_intervals[-1] - error > len(signal):
                    break
                
                certified = True
                fingerprint_echoes: list[Echo] = [effective_echoes[i]]
                first_echo_amp = effective_echoes[i].peak_height
                if first_echo_amp == 0: continue

                for pulse_index in range(pulse_num - 1):
                    base_position = config.gt_intervals[pulse_index]
                    applicable_echoes_indices = np.flatnonzero(np.abs((peaks - peaks[i]) - base_position) <= error)
                    applicable_echoes_indices = applicable_echoes_indices[applicable_echoes_indices != i]

                    if len(applicable_echoes_indices) == 0:
                        certified = False
                        break
                    
                    selected_echo_idx = -1
                    if config.consider_amp:
                        for a_echo_idx in applicable_echoes_indices:
                            actual_ratio = effective_echoes[a_echo_idx].peak_height / first_echo_amp
                            ideal_ratio = config.gt_amps_ratio[pulse_index + 1] / config.gt_amps_ratio[0]
                            if abs(actual_ratio - ideal_ratio) <= config.amp_torelance_error_ratio:
                                selected_echo_idx = a_echo_idx
                                break
                        if selected_echo_idx == -1:
                            certified = False
                            break
                    else:
                        selected_echo_idx = applicable_echoes_indices[0]

                    fingerprint_echoes.append(effective_echoes[selected_echo_idx])

                if certified:
                    certified_echoes.append(EchoGroup(fingerprint_echoes))
            effective_echoes_group = certified_echoes
        
        if not effective_echoes_group: return [], []
        
        effective_echoes_group.sort(key=lambda x: (-x[0].peak_height, x[0].peak_position))
        effective_echoes_group = effective_echoes_group[:10]

        first_echo_of_strongest_group = effective_echoes_group[0][0]
        fired_pulse_height = (config.gt_amps_ratio[0] * self.amplitude) if (config.consider_amp and config.gt_amps_ratio) else self.amplitude
        highest_peak = first_echo_of_strongest_group.peak_height / fired_pulse_height
        highest_peak_time = first_echo_of_strongest_group.peak_position

        distance_m = (highest_peak_time * self.time_resolution_ns) * 0.15
        intensity = int(min(highest_peak * 255, 255))
        
        alpha = np.deg2rad(config.azimuth / 100.0)
        omega = np.deg2rad(config.altitude / 100.0)
        x = distance_m * np.sin(alpha) * np.cos(omega)
        y = distance_m * np.cos(alpha) * np.cos(omega)
        z = distance_m * np.sin(omega)

        return [VeloPoint(intensity=intensity, channel=0, timestamp=config.start_timestamp.in_nanoseconds, azimuth=config.azimuth, altitude=config.altitude, distance_m=distance_m, x=x, y=y, z=z)], effective_echoes_group

    def get_point_count(self) -> int:
        return self.max_index

    def get_undetected_points(self) -> np.ndarray:
        undetected_indices = list(self.all_point_indices - self.detected_point_indices)
        return self.points[undetected_indices]