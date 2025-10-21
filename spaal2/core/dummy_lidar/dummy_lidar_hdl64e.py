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

class PcdLidarHDL64E:
    # HDL-64E fire angles extracted from KITTI point cloud data
    # VCF: Vertical Correction Factor (v_angle)
    # RCF: Rotational Correction Factor (h_offset)
    # Note: KITTI data has minimal h_offset (already corrected)
    fire_angles: list[FireAngle] = [
        FireAngle(3.2580518723, -0.0019178391),  # Channel 0
        FireAngle(2.2037801743, 0.0003433228),   # Channel 1
        FireAngle(1.7058248520, 0.0002212524),   # Channel 2
        FireAngle(1.1367645264, -0.0017490387),  # Channel 3
        FireAngle(0.8440742493, -0.0018463135),  # Channel 4
        FireAngle(0.4481000900, 0.0001792908),   # Channel 5
        FireAngle(0.1777248383, 0.0027389526),   # Channel 6
        FireAngle(-0.1850357056, 0.0013809204),  # Channel 7
        FireAngle(-0.5362186432, -0.0029830933), # Channel 8
        FireAngle(-0.9345235825, -0.0001039505), # Channel 9
        FireAngle(-1.2323884964, 0.0020675659),  # Channel 10
        FireAngle(-1.5841431618, 0.0011596680),  # Channel 11
        FireAngle(-1.9751014709, -0.0001299381), # Channel 12
        FireAngle(-2.3321866989, -0.0012130737), # Channel 13
        FireAngle(-2.6619057655, 0.0002975464),  # Channel 14
        FireAngle(-2.9901218414, 0.0007591248),  # Channel 15
        FireAngle(-3.3205828667, -0.0013275146), # Channel 16
        FireAngle(-3.6402072906, -0.0009424686), # Channel 17
        FireAngle(-3.9481477737, 0.0025863647),  # Channel 18
        FireAngle(-4.2664375305, -0.0010204315), # Channel 19
        FireAngle(-4.5766868591, 0.0022029877),  # Channel 20
        FireAngle(-4.8997979164, -0.0003738403), # Channel 21
        FireAngle(-5.2262482643, -0.0003833771), # Channel 22
        FireAngle(-5.5500288010, -0.0010881424), # Channel 23
        FireAngle(-5.8619031906, -0.0016117096), # Channel 24
        FireAngle(-6.1563215256, 0.0008049011),  # Channel 25
        FireAngle(-6.4209241867, -0.0005798340), # Channel 26
        FireAngle(-6.6594710350, 0.0034103394),  # Channel 27
        FireAngle(-6.8932523727, 0.0005264282),  # Channel 28
        FireAngle(-7.1558170319, 0.0026206970),  # Channel 29
        FireAngle(-7.4280138016, -0.0009574890), # Channel 30
        FireAngle(-7.8201732635, 0.0000457764),  # Channel 31
        FireAngle(-8.4200296402, 0.0011215210),  # Channel 32
        FireAngle(-8.9318990707, -0.0007858276), # Channel 33
        FireAngle(-9.3866062164, -0.0003433228), # Channel 34
        FireAngle(-9.7722358704, 0.0012168884),  # Channel 35
        FireAngle(-10.2273349762, 0.0011940002), # Channel 36
        FireAngle(-10.8219451904, -0.0003356934),# Channel 37
        FireAngle(-11.3157062531, 0.0009346008), # Channel 38
        FireAngle(-11.7347698212, 0.0010781288), # Channel 39
        FireAngle(-12.1810989380, 0.0009231567), # Channel 40
        FireAngle(-12.6001729965, 0.0001068115), # Channel 41
        FireAngle(-13.1180887222, -0.0020141602),# Channel 42
        FireAngle(-13.6323451996, 0.0004081726), # Channel 43
        FireAngle(-14.1992654800, 0.0006427765), # Channel 44
        FireAngle(-14.6381416321, -0.0001716614),# Channel 45
        FireAngle(-15.1361141205, -0.0021486282),# Channel 46
        FireAngle(-15.4987249374, -0.0005149841),# Channel 47
        FireAngle(-16.1125431061, 0.0005569458), # Channel 48
        FireAngle(-16.6406726837, 0.0001831055), # Channel 49
        FireAngle(-17.2057380676, 0.0008087158), # Channel 50
        FireAngle(-17.6588668823, 0.0012111664), # Channel 51
        FireAngle(-18.1662464142, -0.0003929138),# Channel 52
        FireAngle(-18.5672683716, -0.0004024506),# Channel 53
        FireAngle(-19.0264759064, -0.0010604858),# Channel 54
        FireAngle(-19.5698242188, -0.0004730225),# Channel 55
        FireAngle(-20.0774421692, -0.0008888245),# Channel 56
        FireAngle(-20.6752262115, -0.0003051758),# Channel 57
        FireAngle(-21.1592216492, 0.0015869141), # Channel 58
        FireAngle(-21.5547599792, -0.0015563965),# Channel 59
        FireAngle(-21.9995670319, 0.0017395020), # Channel 60
        FireAngle(-22.6350288391, -0.0025634766),# Channel 61
        FireAngle(-23.1590080261, 0.0034179688), # Channel 62
        FireAngle(-23.6364269257, 0.0011444092), # Channel 63
    ]
    vertical_angles: list[float] = [fa.v_angle for fa in fire_angles]


    def __init__(self, pcd_file_path: Optional[str], lidar_position: np.ndarray, lidar_rotation: np.ndarray, base_timestamp: PreciseDuration = PreciseDuration(nanoseconds=0), amplitude: float = 1.0, pulse_width: PreciseDuration = PreciseDuration(nanoseconds=10), time_resolution_ns: float = 1.0, intensity_to_amplitude_ratio: float = 40.0/255.0, initial_point_offset: int = 0, scan_mode: str = 'horizontal', horizontal_resolution_deg: float = 0.1) -> None:
        self.index: int = 0
        self.horizontal_resolution_deg = horizontal_resolution_deg  # Default 0.1° for HDL-64E
        self.max_index: int = int(360 / self.horizontal_resolution_deg * 64)  # 64 channels
        self.accept_window = PreciseDuration(nanoseconds=800)
        self.base_timestamp = base_timestamp
        self.amplitude = amplitude
        self.pulse_width = pulse_width
        self.time_resolution_ns = time_resolution_ns
        self.lidar_position = lidar_position
        self.lidar_rotation = lidar_rotation
        self.pcd_files: list[str] = []
        self.pcd_file_path = pcd_file_path
        self.intensity_to_amplitude_ratio = intensity_to_amplitude_ratio
        self.intensities: Optional[np.ndarray] = None
        self.initial_point_offset = initial_point_offset
        self.scan_mode = scan_mode

        self.sync_angle_step = horizontal_resolution_deg  # degrees

        # Create a sorted list of vertical angles for the new scan pattern
        self.sorted_vertical_angles = sorted(self.vertical_angles, reverse=True)
        self.horizontal_steps = self.max_index // 64  # Should be 1800 for HDL-64E

        # Create mapping from sorted vertical angle index to original fire_angle
        self.sorted_v_angle_to_fire_angle = {}
        for i, v_angle in enumerate(self.sorted_vertical_angles):
            for fire_angle in self.fire_angles:
                if abs(fire_angle.v_angle - v_angle) < 1e-6:
                    self.sorted_v_angle_to_fire_angle[i] = fire_angle
                    break

        # Azimuth-based time perturbation settings
        self.perturbation_azimuth_thresholds_deg: list[float] = []
        self.perturbation_times_ns: list[float] = []

        if pcd_file_path:
            if not os.path.exists(pcd_file_path):
                 raise FileNotFoundError(f"File not found at {pcd_file_path}")
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
        print(f"LiDAR: HDL-64E initialized with {len(self.points) if hasattr(self, 'points') else 0} points")

    def _parse_pcd_file(self, file_path: str):
        with open(file_path, 'r') as f:
            lines = f.readlines()

        header_end_idx = 0
        fields = []
        data_type = ''
        num_points = 0
        for i, line in enumerate(lines):
            line_s = line.strip()
            if line_s.startswith('FIELDS'):
                fields = line_s.split(' ')[1:]
            elif line_s.startswith('POINTS'):
                num_points = int(line_s.split(' ')[1])
            elif line_s.startswith('DATA'):
                data_type = line_s.split(' ')[1]
                header_end_idx = i + 1
                break

        if data_type != 'ascii':
            # For binary, fallback to open3d and no intensity
            pcd = o3d.io.read_point_cloud(file_path)
            return np.asarray(pcd.points), None

        if not fields:
            raise ValueError("Could not find FIELDS in PCD header")

        try:
            x_idx = fields.index('x')
            y_idx = fields.index('y')
            z_idx = fields.index('z')
            intensity_idx = fields.index('intensity')
            has_intensity = True
        except ValueError:
            has_intensity = False

        # Pre-allocate numpy arrays
        points = np.zeros((num_points, 3), dtype=np.float32)
        intensities = np.zeros(num_points, dtype=np.float32) if has_intensity else None

        point_idx = 0
        for i in range(header_end_idx, len(lines)):
            line = lines[i].strip()
            if not line:
                continue

            vals = line.split(' ')
            if point_idx < num_points:
                points[point_idx] = [float(vals[x_idx]), float(vals[y_idx]), float(vals[z_idx])]
                if has_intensity:
                    intensities[point_idx] = float(vals[intensity_idx])
                point_idx += 1

        # Trim if num_points was an overestimate
        if point_idx < num_points:
            points = points[:point_idx]
            if has_intensity:
                intensities = intensities[:point_idx]

        return points, intensities

    def _read_pcd(self, file_path: str):
        if file_path.endswith('.pcd'):
            try:
                points, intensities = self._parse_pcd_file(file_path)
            except Exception as e:
                print(f"Failed to parse PCD with custom parser: {e}. Falling back to open3d.")
                pcd = o3d.io.read_point_cloud(file_path)
                points = np.asarray(pcd.points)
                intensities = None
        elif file_path.endswith('.bin'):
            # KITTI dataset format: x, y, z, intensity (4 columns)
            raw_data = np.fromfile(file_path, dtype=np.float32)
            if raw_data.size % 4 != 0:
                raw_data = raw_data[:-(raw_data.size % 4)]
            point_cloud = raw_data.reshape(-1, 4)
            points = point_cloud[:, :3]
            intensities = point_cloud[:, 3]
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

        if points is None:
            raise ValueError(f"Could not read points from {file_path}")

        if self.initial_point_offset != 0:
            points = np.roll(points, -self.initial_point_offset, axis=0)
            if intensities is not None:
                intensities = np.roll(intensities, -self.initial_point_offset, axis=0)

        self.points = points
        self.intensities = intensities

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
        horizontal_resolution = int(self.horizontal_resolution_deg * 100)
        absolute_azimuth_deg = angle_deg + self.initial_azimuth_offset
        if absolute_azimuth_deg < 0:
            absolute_azimuth_deg += 360
        elif absolute_azimuth_deg >= 360:
            absolute_azimuth_deg -= 360
        azimuth_index = absolute_azimuth_deg * 100
        if azimuth_index % horizontal_resolution != 0:
            azimuth_index = azimuth_index - (azimuth_index % horizontal_resolution)
        return azimuth_index

    def _create_depth_map(self):
        if len(self.points) == 0:
            return {}, {}

        points = self.points

        # 1. Vectorized calculations
        depth = np.linalg.norm(points, axis=1)

        # Avoid division by zero for points at the origin
        non_zero_depth_mask = depth > 1e-6

        if not np.any(non_zero_depth_mask):
            return {}, {}

        points = points[non_zero_depth_mask]
        depth = depth[non_zero_depth_mask]
        original_indices = np.arange(len(self.points))[non_zero_depth_mask]

        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        azimuth_deg = np.rad2deg(np.arctan2(x, y))
        altitude_deg = np.rad2deg(np.arcsin(z / depth))

        # 2. Optimize closest_vertical_angle lookup
        vertical_angles_np = np.array(self.vertical_angles)
        diffs = np.abs(altitude_deg[:, np.newaxis] - vertical_angles_np)
        closest_v_angle_indices = np.argmin(diffs, axis=1)
        closest_vertical_angles = vertical_angles_np[closest_v_angle_indices]

        # 3. Vectorized key creation
        horizontal_resolution = int(self.horizontal_resolution_deg * 100)
        azimuth_indices = np.round(azimuth_deg * 100 / horizontal_resolution)
        discretized_azimuths = azimuth_indices * horizontal_resolution

        azimuth_keys = (discretized_azimuths % 36000).astype(int)
        altitude_keys = (closest_vertical_angles * 100).astype(int)

        # 4. Filter for minimum depth per key using sorting
        # Sort by azimuth, then altitude, then depth.
        sort_indices = np.lexsort((depth, altitude_keys, azimuth_keys))

        sorted_azimuth_keys = azimuth_keys[sort_indices]
        sorted_altitude_keys = altitude_keys[sort_indices]

        # Find the first occurrence of each unique (azimuth, altitude) pair.
        keys_arr = np.c_[sorted_azimuth_keys, sorted_altitude_keys]
        _, first_occurrence_indices = np.unique(keys_arr, axis=0, return_index=True)

        # Get the original indices from the *unsorted* data that correspond to our final selection
        final_indices = sort_indices[first_occurrence_indices]

        # Create the maps using the final selected indices
        final_azimuth_keys = azimuth_keys[final_indices]
        final_altitude_keys = altitude_keys[final_indices]
        final_depths = depth[final_indices]
        final_original_indices = original_indices[final_indices]

        depth_map = {}
        original_point_indices_map = {}

        intensities = self.intensities
        has_intensities = intensities is not None

        if has_intensities:
            # Filter intensities based on the non_zero_depth_mask as well
            intensities = intensities[non_zero_depth_mask]
            final_intensities = intensities[final_indices]

        for i in range(len(final_azimuth_keys)):
            key = (final_azimuth_keys[i], final_altitude_keys[i])
            pcd_intensity = final_intensities[i] if has_intensities else None
            depth_map[key] = (final_depths[i], pcd_intensity)
            original_point_indices_map[key] = final_original_indices[i]

        return depth_map, original_point_indices_map

    def set_intensity_to_amplitude_ratio(self, ratio: float):
        self.intensity_to_amplitude_ratio = ratio

    def set_amplitude(self, amplitude: float):
        self.amplitude = amplitude

    def new_frame_from_points(self, points: np.ndarray, intensities: np.ndarray, base_timestamp: PreciseDuration = PreciseDuration(nanoseconds=0)) -> "PcdLidarHDL64E":
        self.points = points
        self.intensities = intensities

        if self.initial_point_offset != 0:
            self.points = np.roll(self.points, -self.initial_point_offset, axis=0)
            if self.intensities is not None:
                self.intensities = np.roll(self.intensities, -self.initial_point_offset, axis=0)

        self.all_point_indices = set(range(len(self.points)))

        if len(self.points) > 0:
            first_point = self.points[0]
            self.initial_azimuth_offset = np.rad2deg(np.arctan2(first_point[1], first_point[0]))
        else:
            self.initial_azimuth_offset = 0

        self.depth_map, self.original_point_indices_map = self._create_depth_map()
        self.no_signal_scan_angles: list[tuple[int, int]] = []

        self.index = 0
        self.base_timestamp = base_timestamp
        self.detected_point_indices = set()
        return self

    def new_frame(self, frame_num: int = 0, base_timestamp: PreciseDuration = PreciseDuration(nanoseconds=0)) -> "PcdLidarHDL64E":
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
        # Determine which altitude ring and azimuth step we are on based on the scan pattern
        if self.scan_mode == 'vertical':
            azimuth_index_in_ring = self.index // 64
            altitude_index = self.index % 64
        else:  # horizontal
            altitude_index = self.index // self.horizontal_steps
            azimuth_index_in_ring = self.index % self.horizontal_steps

        # Get the corresponding fire_angle to access both v_angle and h_offset
        fire_angle = self.sorted_v_angle_to_fire_angle.get(altitude_index)
        if fire_angle is None:
            # Fallback: use sorted vertical angle without h_offset
            v_angle = self.sorted_vertical_angles[altitude_index]
            h_offset = 0.0
        else:
            v_angle = fire_angle.v_angle
            h_offset = fire_angle.h_offset

        altitude_key = int(v_angle * 100)

        # Calculate the azimuth key based on the configured horizontal resolution
        horizontal_resolution = int(self.horizontal_resolution_deg * 100)  # e.g., 0.1° * 100 = 10
        lidar_azimuth_deg = azimuth_index_in_ring * self.horizontal_resolution_deg

        # Apply h_offset (Rotational Correction Factor)
        lidar_azimuth_deg += h_offset

        # Apply the initial offset for the current frame to get the absolute angle
        absolute_azimuth_deg = lidar_azimuth_deg + self.initial_azimuth_offset

        # Discretize the azimuth to create the lookup key for the depth map
        azimuth_index = round(absolute_azimuth_deg * 100 / horizontal_resolution)
        discretized_azimuth = (azimuth_index * horizontal_resolution)
        azimuth_key = int(discretized_azimuth % 36000)

        return azimuth_key, altitude_key

    def _get_current_timestamp(self) -> int:
        horizontal_steps_per_ring = self.max_index // 64

        if self.scan_mode == 'vertical':
            num_vertical_channels = 64
            azimuth_step_index = self.index // num_vertical_channels
            altitude_step_index = self.index % num_vertical_channels

            # Get current vertical angle from the pre-sorted list
            current_vertical_angle = self.sorted_vertical_angles[altitude_step_index]

            # Timestamp logic similar to horizontal mode, but for vertical direction
            time_increase_per_step = 20  # ns
            timestamp = (current_vertical_angle // self.sync_angle_step) * time_increase_per_step

            # Add a large time jump for each new azimuth column
            time_jump_per_azimuth = 50234
            timestamp += azimuth_step_index * time_jump_per_azimuth

            current_azimuth_deg = azimuth_step_index * self.horizontal_resolution_deg + 8

        else: # horizontal
            azimuth_step_index = self.index % horizontal_steps_per_ring
            altitude_step_index = self.index // horizontal_steps_per_ring
            current_azimuth_deg = azimuth_step_index * self.horizontal_resolution_deg + 8

            # Timestamp increases as azimuth increases
            time_increase_per_step = 20  # ns
            timestamp = (current_azimuth_deg // self.sync_angle_step) * time_increase_per_step

            # Add time jump for each altitude ring change
            timestamp += altitude_step_index * 50234

        return int(timestamp) + self.base_timestamp.in_nanoseconds

    def scan(self) -> tuple[MeasurementConfig, npt.NDArray[np.float64]]:
        if self.index >= self.max_index:
            raise StopIteration()
        azimuth, altitude = self._get_current_angle()
        timestamp = self._get_current_timestamp()

        depth_info = self.depth_map.get((azimuth, altitude))

        signal_length = int(self.accept_window.in_nanoseconds / self.time_resolution_ns)
        signal = np.zeros((signal_length, ))

        if depth_info is not None:
            depth, pcd_intensity = depth_info
            point_idx = self.original_point_indices_map.get((azimuth, altitude))
            if point_idx is not None:
                self.detected_point_indices.add(point_idx)

            time_of_flight_index = int(depth / (0.15 * self.time_resolution_ns))
            time_of_flight = depth / (0.15 * self.time_resolution_ns)
            if time_of_flight_index < signal_length:
                pulse_width_indices = int(self.pulse_width.in_nanoseconds / self.time_resolution_ns)
                if pulse_width_indices > 0:
                    # The peak of the pulse is at the time of flight
                    mu = time_of_flight

                    # Calculate sigma so the FWHM matches the configured pulse_width
                    fwhm = pulse_width_indices
                    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

                    # Define the window to generate the pulse over
                    start_idx = max(0, int(mu - fwhm))
                    end_idx = min(signal_length, int(mu + fwhm))

                    if start_idx < end_idx:
                        pulse_indices = np.arange(start_idx, end_idx)

                        pulse_amplitude = self.amplitude
                        if pcd_intensity is not None:
                            pulse_amplitude = pcd_intensity * self.intensity_to_amplitude_ratio

                        # Generate the Gaussian pulse
                        gaussian_pulse = pulse_amplitude * np.exp(-((pulse_indices - mu)**2) / (2 * (sigma**2)))

                        signal[pulse_indices] += gaussian_pulse
                elif time_of_flight_index < signal_length:
                    # Fallback for zero pulse width
                    pulse_amplitude = self.amplitude
                    if pcd_intensity is not None:
                        pulse_amplitude = pcd_intensity * self.intensity_to_amplitude_ratio
                    signal[time_of_flight_index] += pulse_amplitude
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
