import numpy as np
import numpy.typing as npt

from spaal2.core import PreciseDuration, MeasurementConfig, VeloPoint

class DummyLidarVLP16:
    vertical_angles: list[int] = [-15, 1, -13, 3, -11, 5, -9, 7, -7, 9, -5, 11, -3, 13, -1, 15]

    def __init__(self, base_timestamp: PreciseDuration = PreciseDuration(nanoseconds=0), amplitude: float = 1.0, pulse_width: PreciseDuration = PreciseDuration(nanoseconds=10), time_resolution_ns: float = 1.0) -> None:
        self.index: int = 0
        self.max_index: int = int(360 // 0.2 * 16)
        self.accept_window = PreciseDuration(nanoseconds=800)
        self.base_timestamp = base_timestamp
        self.amplitude = amplitude
        self.pulse_width = pulse_width
        self.time_resolution_ns = time_resolution_ns

    def set_amplitude(self, amplitude: float):
        self.amplitude = amplitude

    def new_frame(self, base_timestamp: PreciseDuration) -> "DummyLidarVLP16":
        """
        新しいフレームを作成する
        """
        return DummyLidarVLP16(
            base_timestamp=base_timestamp, 
            amplitude=self.amplitude, 
            pulse_width=self.pulse_width,
            time_resolution_ns=self.time_resolution_ns
        )

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

        signal_length = int(self.accept_window.in_nanoseconds / self.time_resolution_ns)
        signal = np.zeros((signal_length, ))
        pulse_width_indices = int(self.pulse_width.in_nanoseconds / self.time_resolution_ns)
        if pulse_width_indices > 0:
            signal[:pulse_width_indices] = self.amplitude

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