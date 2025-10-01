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
        FireAngle(-30.67, 0), FireAngle(-9.33, 0),
        FireAngle(-29.33, 0), FireAngle(-8.0, 0),
        FireAngle(-28.0, 0), FireAngle(-6.66, 0),
        FireAngle(-26.66, 0), FireAngle(-5.33, 0),
        FireAngle(-25.33, 0), FireAngle(-4.0, 0),
        FireAngle(-24.0, 0), FireAngle(-2.67, 0),
        FireAngle(-22.67, 0), FireAngle(-1.33, 0),
        FireAngle(-21.33, 0), FireAngle(0.0, 0),
        FireAngle(-20.0, 0), FireAngle(1.33, 0),
        FireAngle(-18.67, 0), FireAngle(2.67, 0),
        FireAngle(-17.33, 0), FireAngle(4.0, 0),
        FireAngle(-16.0, 0),
        FireAngle(5.33, 0), FireAngle(-14.67, 0),
        FireAngle(6.67, 0), FireAngle(-13.33, 0),
        FireAngle(8.0, 0), FireAngle(-12.0, 0),
        FireAngle(9.33, 0), FireAngle(-10.67, 0),
        FireAngle(10.67, 0)
    ]
    vertical_angles: list[float] = [fa.v_angle for fa in fire_angles]


    def __init__(self, pcd_file_path: Optional[str], lidar_position: np.ndarray, lidar_rotation: np.ndarray, base_timestamp: PreciseDuration = PreciseDuration(nanoseconds=0), amplitude: float = 1.0, pulse_width: PreciseDuration = PreciseDuration(nanoseconds=10), time_resolution_ns: float = 1.0) -> None:
        self.index: int = 0
        self.max_index: int = int(360 / 0.2 * 32)
        self.accept_window = PreciseDuration(nanoseconds=800)
        self.base_timestamp = base_timestamp
        self.amplitude = amplitude
        self.pulse_width = pulse_width
        self.time_resolution_ns = time_resolution_ns
        self.lidar_position = lidar_position
        self.lidar_rotation = lidar_rotation
        self.pcd_files: list[str] = []
        self.pcd_file_path = pcd_file_path

        self.sync_angle_step = 0.2  # degrees

        # Create a sorted list of vertical angles for the new scan pattern
        self.sorted_vertical_angles = sorted(self.vertical_angles, reverse=True)
        self.horizontal_steps = self.max_index // 32  # Should be 1800 for VLP-32c

        # Azimuth-based time perturbation settings
        self.perturbation_azimuth_thresholds_deg: list[float] = []
        self.perturbation_times_ns: list[float] = []

        if pcd_file_path:
            if not os.path.exists(pcd_file_path):
                 raise FileNotFoundError(f"PCD file not found at {pcd_file_path}")
            self._read_pcd(pcd_file_path)
        else:
            # Handle case where no file is provided initially
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
            # Re-calculate offset for each file
            self.initial_azimuth_offset = np.rad2deg(np.arctan2(first_point[1], first_point[0]))
        else:
            self.initial_azimuth_offset = 0

        self.depth_map, self.original_point_indices_map = self._create_depth_map()
        self.no_signal_scan_angles: list[tuple[int, int]] = []

    def set_pcd_files(self, pcd_files: list[str]):
        self.pcd_files = pcd_files
    
    def set_sync_angle_step(self, angle_step: float):
        self.sync_angle_step = angle_step

    def set_azimuth_time_perturbation(self, thresholds_deg: list[float], times_ns: list[float]):
        """
        Sets multiple time perturbations based on azimuth angles. The perturbations are step-wise.
        For example, if thresholds_deg=[90, 180] and times_ns=[20, 40], then for an azimuth `a`:
        - if a < 90, no perturbation is added.
        - if 90 <= a < 180, 20ns is added.
        - if a >= 180, 40ns is added.

        Parameters
        ----------
        thresholds_deg : list[float]
            A list of azimuth angle thresholds (in degrees).
        times_ns : list[float]
            A list of delay times in nanoseconds to add. Must have the same length as thresholds_deg.
        """
        if len(thresholds_deg) != len(times_ns):
            raise ValueError("thresholds_deg and times_ns must have the same length.")
        
        # Sort by threshold to ensure correct application
        sorted_pairs = sorted(zip(thresholds_deg, times_ns))
        self.perturbation_azimuth_thresholds_deg = [p[0] for p in sorted_pairs]
        self.perturbation_times_ns = [p[1] for p in sorted_pairs]

    def get_azimuth_index(self, angle_deg: float) -> int:
        horizontal_resolution = 20
        absolute_azimuth_deg = angle_deg + self.initial_azimuth_offset
        if absolute_azimuth_deg < 0:
            absolute_azimuth_deg += 360
        elif absolute_azimuth_deg >= 360:
            absolute_azimuth_deg -= 360
        azimuth_index = absolute_azimuth_deg * 100
        if azimuth_index % horizontal_resolution != 0:
            azimuth_index = azimuth_index - (azimuth_index % horizontal_resolution)
        #print(f"Azimuth index for angle {angle_deg} degrees: {azimuth_index}")
        return azimuth_index

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

    def new_frame(self, frame_num: int = 0, base_timestamp: PreciseDuration = PreciseDuration(nanoseconds=0)) -> "PcdLidarVLP32c":
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
        # Determine which altitude ring and azimuth step we are on based on the new ideal scan pattern
        altitude_index = self.index // self.horizontal_steps
        azimuth_index_in_ring = self.index % self.horizontal_steps

        # Get the corresponding vertical angle from the pre-sorted list
        v_angle = self.sorted_vertical_angles[altitude_index]
        altitude_key = int(v_angle * 100)

        # Calculate the azimuth key based on the ideal horizontal step
        horizontal_resolution = 20  # 0.2 degrees * 100
        lidar_azimuth_deg = azimuth_index_in_ring * 0.2
        
        # Apply the initial offset for the current frame to get the absolute angle
        absolute_azimuth_deg = lidar_azimuth_deg + self.initial_azimuth_offset
        
        # Discretize the azimuth to create the lookup key for the depth map
        azimuth_index = round(absolute_azimuth_deg * 100 / horizontal_resolution)
        discretized_azimuth = (azimuth_index * horizontal_resolution)
        azimuth_key = int(discretized_azimuth % 36000)

        return azimuth_key, altitude_key

    def _get_current_timestamp(self) -> int:
        horizontal_steps_per_ring = self.max_index // 32
        azimuth_step_index = self.index % horizontal_steps_per_ring
        current_azimuth_deg = azimuth_step_index * 0.2
        altitude_step_index = self.index // horizontal_steps_per_ring

        # 水平角が20度増加するごとにタイムスタンプを20ns増加させる
        self.sync_angle_step = 5 # degrees
        time_increase_per_step = 20  # ns
        timestamp = (current_azimuth_deg // self.sync_angle_step) * time_increase_per_step

        # 高度リングが変わるごとに5060nsの時間を追加
        timestamp += altitude_step_index * 50234

        # Check for azimuth-based time perturbation
        # time_perturbation = 0.0
        # if self.perturbation_azimuth_thresholds_deg:
        #     for i in range(len(self.perturbation_azimuth_thresholds_deg)):
        #         if current_azimuth_deg >= self.perturbation_azimuth_thresholds_deg[i]:
        #             time_perturbation = self.perturbation_times_ns[i]
        
        # timestamp += time_perturbation
        
        return int(timestamp) + self.base_timestamp.in_nanoseconds

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
            time_of_flight = depth / (0.15 * self.time_resolution_ns) 
            if time_of_flight_index < signal_length:
                pulse_width_indices = int(self.pulse_width.in_nanoseconds / self.time_resolution_ns)
                if pulse_width_indices > 0:
                    # The peak of the pulse is at the time of flight
                    #mu = time_of_flight_index
                    mu = time_of_flight

                    # Calculate sigma so the FWHM matches the configured pulse_width
                    fwhm = pulse_width_indices
                    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

                    # Define the window to generate the pulse over
                    start_idx = max(0, int(mu - fwhm))
                    end_idx = min(signal_length, int(mu + fwhm))
                    
                    if start_idx < end_idx:
                        pulse_indices = np.arange(start_idx, end_idx)
                        
                        # Generate the Gaussian pulse
                        gaussian_pulse = self.amplitude * np.exp(-((pulse_indices - mu)**2) / (2 * (sigma**2)))
                        
                        # Use += in case of future overlapping effects
                        signal[pulse_indices] += gaussian_pulse
                elif time_of_flight_index < signal_length:
                    # Fallback for zero pulse width
                    signal[time_of_flight_index] += self.amplitude
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
