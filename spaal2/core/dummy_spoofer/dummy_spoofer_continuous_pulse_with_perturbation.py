import numpy as np
import numpy.typing as npt
import math
import random

from spaal2.core.dummy_spoofer import DummySpooferInterface
from spaal2.core.measurement_config import MeasurementConfig, PreciseDuration

class DummySpooferContinuousPulseWithPerturbation(DummySpooferInterface):
    """
    周期的なパルスを発生させるSpoofer（摂動あり）
    """
    def __init__(self, frequency: float, pulse_width: PreciseDuration, perturbation_ns: float, amplitude: float = 9.0, debug: bool = False) -> None:
        """
        Parameters
        ----------
        frequency : float
            パルスの周波数(Hz)
        pulse_width : PreciseDuration
            パルスの幅
        perturbation_ns : float
            パルスタイミングの摂動の大きさ（ナノ秒、標準偏差）
        debug : bool, optional
            デバッグ情報を表示するかどうか, by default False
        """
        self.frequency = frequency
        self.pulse_width = pulse_width
        self.perturbation_ns = perturbation_ns
        self.amplitude = amplitude
        self.pulse_period_ns = math.floor(1 / self.frequency * 1e9)

        self._precompute_pulse_shape()

    def set_amplitude(self, amplitude: float):
        self.amplitude = amplitude
        self._precompute_pulse_shape()

    def _precompute_pulse_shape(self):
        # パルス形状を事前計算
        sigma = self.pulse_width.in_nanoseconds / (2 * np.sqrt(2 * np.log2(2)))
        pulse_x = np.arange(-3 * sigma, 3 * sigma, 1.0)
        # ガウス分布
        pulse_shape = self.amplitude * np.exp(-((pulse_x) ** 2) / (2 * sigma ** 2))
        one_pulse_sequence = np.zeros((self.pulse_period_ns, ))
        one_pulse_sequence[:pulse_shape.size] = pulse_shape

        # 5000ns + 摂動の4σ分のバッファを事前計算
        max_duration_ns = 5000 + math.ceil(self.perturbation_ns * 4)
        self.pulse_sequence_buffer = np.tile(
            one_pulse_sequence, 
            math.ceil(max_duration_ns / self.pulse_period_ns)
        )

    def trigger(self, config: MeasurementConfig, signal: npt.NDArray[np.float64]):
        # 非同期なので何もしない
        pass

    def get_range_signal(self, start_timestamp: PreciseDuration, duration: PreciseDuration) -> npt.NDArray[np.float64]:
        # 摂動を加える
        perturbation = int(random.gauss(0, self.perturbation_ns))
        
        # 開始時刻から位相を計算し、バッファから切り出す
        phase = (start_timestamp.in_nanoseconds + perturbation) % self.pulse_period_ns

        # 位相が負にならないように調整
        if phase < 0:
            phase += self.pulse_period_ns
            
        end_index = phase + duration.in_nanoseconds
        
        # バッファの範囲外アクセスを防ぐ
        if end_index > len(self.pulse_sequence_buffer):
            # このケースは通常発生しないはずだが、念のため
            signal_part = self.pulse_sequence_buffer[phase:]
            padding = np.zeros(end_index - len(self.pulse_sequence_buffer))
            return np.concatenate([signal_part, padding])

        return self.pulse_sequence_buffer[phase:end_index]
