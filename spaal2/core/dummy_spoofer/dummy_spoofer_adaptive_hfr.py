import numpy as np
import numpy.typing as npt
import math
from typing import Optional

from spaal2.core.dummy_spoofer import DummySpooferInterface
from spaal2.core.measurement_config import MeasurementConfig, PreciseDuration

class DummySpooferAdaptiveHFR(DummySpooferInterface):
    """
    Adaptive HFR Spoofer
    """
    def __init__(self, 
                 frequency: float,
                 duration: PreciseDuration,
                 spoofer_distance_m: float,
                 pulse_width: PreciseDuration,
                 amplitude: float = 9.0,
                 time_resolution_ns: float = 1.0,
                 debug: bool = False) -> None:
        """
        Parameters
        ----------
        frequency : float
            パルスの周波数(Hz)
        duration : PreciseDuration
            トリガーされてからの攻撃継続時間
        spoofer_distance_m : float
            SpooferとLiDARの距離(m)
        pulse_width : PreciseDuration
            パルスの幅
        time_resolution_ns : float, optional
            時間分解能 (ns), by default 1.0
        debug : bool, optional
            デバッグ情報を表示するかどうか, by default False
        """
        self.frequency = frequency
        self.duration = duration
        self.distance_m = spoofer_distance_m
        self.pulse_width = pulse_width
        self.amplitude = amplitude
        self.time_resolution_ns = time_resolution_ns
        self.pulse_period_ns = 1 / self.frequency * 1e9
        self.trigger_time: Optional[PreciseDuration] = None

        self._precompute_pulse_shape()

    def set_amplitude(self, amplitude: float):
        self.amplitude = amplitude
        self._precompute_pulse_shape()

    def _precompute_pulse_shape(self):
        # パルス形状を事前計算 (分解能1.0nsを基準とする)
        sigma = self.pulse_width.in_nanoseconds / (2 * np.sqrt(2 * np.log2(2)))
        pulse_x = np.arange(-3 * sigma, 3 * sigma, 1.0) #基準の分解能で計算
        # ガウス分布
        pulse_shape = self.amplitude * np.exp(-((pulse_x) ** 2) / (2 * sigma ** 2))

        # 5000nsのバッファを事前計算し、必要な部分だけを逐一切り出すようにする
        max_duration_ns = 5000
        self.pulse_sequence_buffer = np.zeros((max_duration_ns, ))
        for i in range(math.ceil(max_duration_ns / 1e9 * self.frequency)):
            index_min = round(i * self.pulse_period_ns)
            index_max = min(index_min + pulse_shape.size, max_duration_ns)
            self.pulse_sequence_buffer[index_min:index_max] = pulse_shape[:index_max - index_min]

    def trigger(self, config: MeasurementConfig, signal: npt.NDArray[np.float64]):
        if self.trigger_time is not None:
            return
        
        # delay
        delay_indices = int(self.distance_m / (0.15 * self.time_resolution_ns))
        new_signal = np.zeros_like(signal)
        if delay_indices < len(signal):
            new_signal[delay_indices:] = signal[:-delay_indices]
        signal = new_signal

        # find the first peak
        raises = np.flatnonzero(
            (signal[:-1] < 0.5) & (signal[1:] >= 0.5)
        ) + 1
        if raises.size == 0:
            return
        peak_index = raises[0]
        peak_time_ns = peak_index * self.time_resolution_ns
        self.trigger_time = config.start_timestamp + PreciseDuration(nanoseconds=peak_time_ns)
        print(f"Triggered at {self.trigger_time.in_nanoseconds}ns")

    def get_range_signal(self, start_timestamp: PreciseDuration, duration: PreciseDuration) -> npt.NDArray[np.float64]:
        output_length = int(duration.in_nanoseconds / self.time_resolution_ns)
        
        if self.trigger_time is None:
            return np.zeros((output_length, ))
        
        attack_start_time = self.trigger_time
        attack_end_time = self.trigger_time + self.duration
        if start_timestamp + duration <= attack_start_time:
            return np.zeros((output_length, ))
        if start_timestamp >= attack_end_time:
            self.trigger_time = None
            return np.zeros((output_length, ))
        
        # --- Optimized Interpolation ---
        # Create the target time points for the output signal
        target_time_points = np.arange(output_length) * self.time_resolution_ns

        # Calculate the corresponding time points in the buffer's reference frame
        phase_ns = (start_timestamp - attack_start_time).in_nanoseconds
        sample_times = phase_ns + target_time_points

        # Find the indices and weights for linear interpolation
        t_floor = np.floor(sample_times)
        interp_weights = sample_times - t_floor

        # Get the buffer values at floor and ceil positions, wrapping around the buffer
        buffer_len = len(self.pulse_sequence_buffer)
        indices_floor = t_floor.astype(np.int64) % buffer_len
        indices_ceil = (t_floor + 1).astype(np.int64) % buffer_len

        vals_floor = self.pulse_sequence_buffer[indices_floor]
        vals_ceil = self.pulse_sequence_buffer[indices_ceil]

        # Perform the interpolation
        signal = vals_floor + interp_weights * (vals_ceil - vals_floor)
        # --- End of Optimization ---

        if start_timestamp < attack_start_time:
            clear_until_index = int((attack_start_time - start_timestamp).in_nanoseconds / self.time_resolution_ns)
            if clear_until_index < len(signal):
                signal[:clear_until_index] = 0.0
        if start_timestamp + duration > attack_end_time:
            clear_from_index = int((attack_end_time - start_timestamp).in_nanoseconds / self.time_resolution_ns)
            if clear_from_index < len(signal):
                signal[clear_from_index:] = 0.0
                
        return signal