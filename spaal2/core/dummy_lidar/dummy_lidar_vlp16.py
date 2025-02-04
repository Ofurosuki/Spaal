import numpy as np
import numpy.typing as npt

from spaal2.core import PreciseDuration, MeasurementConfig, VeloPoint

class DummyLidarVLP16:
    vertical_angles: list[int] = [-15, 1, -13, 3, -11, 5, -9, 7, -7, 9, -5, 11, -3, 13, -1, 15]

    def __init__(self, base_timestamp: PreciseDuration = PreciseDuration(nanoseconds=0)) -> None:
        self.index: int = 0
        self.max_index: int = int(360 // 0.2 * 16)
        self.accept_window = PreciseDuration(nanoseconds=800)
        self.base_timestamp = base_timestamp
        print(f"LiDAR: VLP16")

    def new_frame(self, base_timestamp: PreciseDuration) -> "DummyLidarVLP16":
        """
        新しいフレームを作成する
        """
        return DummyLidarVLP16(base_timestamp=base_timestamp)

    def _get_current_angle(self) -> tuple[int, int]:
        """
        Returns the current azimuth and altitude.
        """
        horizontal_index = self.index // 16
        vertical_index = self.index % 16
        return horizontal_index * 20, self._vertical_index_to_altitude(vertical_index)

    def _vertical_index_to_altitude(self, index: int) -> int:
        """
        Returns the altitude for the given vertical index.
        """
        return self.vertical_angles[index] * 100

    def _get_current_timestamp(self) -> int:
        """
        Returns the current timestamp.
        """
        horizontal_index = self.index // 16
        vertical_index = self.index % 16
        return horizontal_index * 55296 + vertical_index * 2304 + self.base_timestamp.in_nanoseconds

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
            (signal[:-1] < 0.01) & (signal[1:] >= 0.01)
        ) + 1
        peaks = np.empty_like(raises, dtype=object)
        for i in range(len(raises)):
            peaks[i] = np.max(
                signal[raises[i]:min(len(signal), raises[i] + 10)]
            )

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