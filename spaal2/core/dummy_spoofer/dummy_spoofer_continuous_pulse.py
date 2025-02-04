import numpy as np
import numpy.typing as npt
import math

from spaal2.core.dummy_spoofer import DummySpooferInterface
from spaal2.core.measurement_config import MeasurementConfig, PreciseDuration

class DummySpooferContinuousPulse(DummySpooferInterface):
    """
    周期的なパルスを発生させるSpoofer
    """
    def __init__(self, frequency: float, pulse_width: PreciseDuration, debug: bool = False) -> None:
        """
        Parameters
        ----------
        frequency : float
            パルスの周波数(Hz)
        pulse_width : PreciseDuration
            パルスの幅
        debug : bool, optional
            デバッグ情報を表示するかどうか, by default False
        """
        self.frequency = frequency
        self.pulse_width = pulse_width
        self.pulse_period_ns = math.floor(1 / self.frequency * 1e9)
        if debug:
            print(f"Spoofer: Continuous Pulse")
            print(f"\tfrequency: {self.frequency / 1e6} MHz")
            print(f"\tpulse_width: {self.pulse_width}")
        
        # パルス形状を事前計算
        sigma = pulse_width.in_nanoseconds / (2 * np.sqrt(2 * np.log2(2)))
        pulse_x = np.arange(-3 * sigma, 3 * sigma, 1.0)
        # ガウス分布
        pulse_shape = 9 * np.exp(-((pulse_x) ** 2) / (2 * sigma ** 2))
        one_pulse_sequence = np.zeros((self.pulse_period_ns, ))
        one_pulse_sequence[:pulse_shape.size] = pulse_shape

        # 5000nsのバッファを事前計算し、必要な部分だけを逐一切り出すようにする
        max_duration = PreciseDuration(nanoseconds=5000)
        self.pulse_sequence_buffer = np.tile(
            one_pulse_sequence, 
            math.ceil(max_duration.in_nanoseconds / self.pulse_period_ns)
        )

    def trigger(self, config: MeasurementConfig, signal: npt.NDArray[np.float64]):
        # 非同期なので何もしない
        pass

    def get_range_signal(self, start_timestamp: PreciseDuration, duration: PreciseDuration) -> npt.NDArray[np.float64]:
        # 開始時刻から位相を計算し、バッファから切り出す
        phase = start_timestamp.in_nanoseconds % self.pulse_period_ns
        return self.pulse_sequence_buffer[phase:phase+duration.in_nanoseconds]