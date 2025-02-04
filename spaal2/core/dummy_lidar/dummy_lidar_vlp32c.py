import numpy as np
import numpy.typing as npt

from spaal2.core import PreciseDuration, MeasurementConfig, VeloPoint

class FireAngle:
    def __init__(self, v_angle: float, h_offset: float) -> None:
        self.v_angle = v_angle
        self.h_offset = h_offset


class DummyLidarVLP32c:
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
        FireAngle(10.333, 1.4), FireAngle(-1.333, -1.4)]

    def __init__(self) -> None:
        self.index: int = 0
        self.max_index: int = int(360 // 0.2 * 32)
        self.accept_window = PreciseDuration(nanoseconds=800)
        print(f"LiDAR: VLP32c")

    def _get_current_angle(self) -> tuple[int, int]:
        """
        Returns the current azimuth and altitude.
        """
        horizontal_index = self.index // 32
        vertical_index = self.index % 32
        fire_angle = self.fire_angles[vertical_index]
        azimuth = int(horizontal_index * 20 + fire_angle.h_offset * 100)
        altitude = int(fire_angle.v_angle * 100)
        return azimuth, altitude

    def _get_current_timestamp(self) -> int:
        """
        Returns the current timestamp.
        """
        horizontal_index = self.index // 32
        vertical_index = int((self.index % 32) // 2)
        return horizontal_index * 55296 + vertical_index * 2304

    def scan(self) -> tuple[MeasurementConfig, npt.NDArray[np.float64]]:
        if self.index >= self.max_index:
            raise StopIteration()
        azimuth, altitude = self._get_current_angle()
        timestamp = self._get_current_timestamp()

        signal = np.zeros((self.accept_window.in_nanoseconds, ))
        signal[:10] = 1.0

        self.index += 1
        return MeasurementConfig(
            start_timestamp=PreciseDuration(nanoseconds=timestamp),
            accept_duration=self.accept_window,
            azimuth=azimuth,
            altitude=altitude,
        ), signal

    def receive(self, config: MeasurementConfig, signal: npt.NDArray[np.float64]) -> list[VeloPoint]:
        raises = np.flatnonzero(
            (signal[:-1] < 0.5) & (signal[1:] >= 0.5)
        ) + 1
        peaks = np.empty_like(raises, dtype=object)
        for i in range(len(raises)):
            peaks[i] = np.max(
                signal[raises[i]:min(len(signal), raises[i] + 10)]
            )
        if peaks.size == 0:
            return []

        # get the highest peak
        highest_peak_index = np.argmax(peaks)
        highest_peak = peaks[highest_peak_index]
        highest_peak_time = raises[highest_peak_index]

        intensity = int(min(highest_peak * 255, 255))
        distance_m = highest_peak_time * 0.15
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