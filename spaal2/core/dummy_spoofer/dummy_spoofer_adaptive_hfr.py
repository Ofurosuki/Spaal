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
        debug : bool, optional
            デバッグ情報を表示するかどうか, by default False
        """
        self.frequency = frequency
        self.duration = duration
        self.distance_m = spoofer_distance_m
        self.pulse_width = pulse_width
        self.pulse_period_ns = 1 / self.frequency * 1e9
        self.trigger_time: Optional[PreciseDuration] = None
        if debug:
            print(f"Spoofer: Adaptive HFR")
            print(f"\tfrequency: {self.frequency / 1e6} MHz")
            print(f"\tduration: {self.duration}")
            print(f"\tpulse_width: {self.pulse_width}")

        # パルス形状を事前計算
        sigma = pulse_width.in_nanoseconds / (2 * np.sqrt(2 * np.log2(2)))
        pulse_x = np.arange(-3 * sigma, 3 * sigma, 1.0)
        # ガウス分布
        pulse_shape = 9 * np.exp(-((pulse_x) ** 2) / (2 * sigma ** 2))

        # 5000nsのバッファを事前計算し、必要な部分だけを逐一切り出すようにする
        max_duration = PreciseDuration(nanoseconds=5000)
        self.pulse_sequence_buffer = np.zeros((max_duration.in_nanoseconds, ))
        for i in range(math.ceil(max_duration.in_nanoseconds / 1e9 * self.frequency)):
            index_min = round(i * self.pulse_period_ns)
            index_max = min(index_min + pulse_shape.size, max_duration.in_nanoseconds)
            self.pulse_sequence_buffer[index_min:index_max] = pulse_shape[:index_max - index_min]

    def trigger(self, config: MeasurementConfig, signal: npt.NDArray[np.float64]):
        if self.trigger_time is not None:
            return
        
        # delay
        new_signal = np.zeros_like(signal)
        delay = int(self.distance_m / 0.15)
        new_signal[delay:] = signal[:-delay]
        signal = new_signal

        # find the first peak
        raises = np.flatnonzero(
            (signal[:-1] < 0.5) & (signal[1:] >= 0.5)
        ) + 1
        if raises.size == 0:
            return
        peak_index = raises[0]
        peak_time = config.start_timestamp + PreciseDuration(nanoseconds=peak_index)

        self.trigger_time = peak_time
        print(f"Triggered at {self.trigger_time.in_nanoseconds}ns")

    def get_range_signal(self, start_timestamp: PreciseDuration, duration: PreciseDuration) -> npt.NDArray[np.float64]:
        if self.trigger_time is None:
            return np.zeros((duration.in_nanoseconds, ))
        
        attack_start_time = self.trigger_time
        attack_end_time = self.trigger_time + self.duration
        if start_timestamp + duration <= attack_start_time:
            return np.zeros((duration.in_nanoseconds, ))
        if start_timestamp >= attack_end_time:
            self.trigger_time = None
            return np.zeros((duration.in_nanoseconds, ))
        
        phase = round( (start_timestamp - attack_start_time).in_nanoseconds % self.pulse_period_ns )
        signal = self.pulse_sequence_buffer[phase:phase+duration.in_nanoseconds]
        if start_timestamp < attack_start_time:
            signal[:attack_start_time.in_nanoseconds - start_timestamp.in_nanoseconds] = 0.0
        if start_timestamp + duration > attack_end_time:
            signal[attack_end_time.in_nanoseconds - start_timestamp.in_nanoseconds:] = 0.0
        return signal