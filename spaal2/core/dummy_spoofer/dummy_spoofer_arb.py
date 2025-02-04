import numpy as np
import numpy.typing as npt
import math
from typing import Optional

from spaal2.core.dummy_spoofer import DummySpooferInterface
from spaal2.core.measurement_config import MeasurementConfig, PreciseDuration

class DummySpooferArb(DummySpooferInterface):
    def __init__(self, arb: npt.NDArray[np.float64], period: PreciseDuration, delay: PreciseDuration) -> None:
        self.distance_m = 10.0
        self.trigger_time: Optional[PreciseDuration] = None
        self.arb = arb
        self.period = period
        self.period_per_sample = int(period.in_nanoseconds // arb.size)
        self.delay = delay

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

    def _get_arb_data_in_ns(self, duration: PreciseDuration) -> float:
        if duration.is_negative or duration >= self.period:
            return 0.0

        return self.arb[math.floor(duration.in_nanoseconds / self.period_per_sample)]

    def get_range_signal(self, start_timestamp: PreciseDuration, duration: PreciseDuration) -> npt.NDArray[np.float64]:
        if self.trigger_time is None:
            return np.zeros((duration.in_nanoseconds, ))

        arb_start = self.trigger_time + self.delay + PreciseDuration(nanoseconds=int(self.distance_m / 0.15))
        arb_end = arb_start + self.period
        if start_timestamp >= arb_end:
            self.trigger_time = None
            return np.zeros((duration.in_nanoseconds, ))

        if start_timestamp >= arb_start and start_timestamp + duration <= arb_end:
            # in the middle of the arb
            start_index_ns = (start_timestamp - arb_start).in_nanoseconds
            arb_start_index = math.floor(start_index_ns / self.period_per_sample)
            arb_end_index = math.floor((start_index_ns + duration.in_nanoseconds - 1) / self.period_per_sample)
            extended_arb = np.repeat(self.arb[arb_start_index:arb_end_index+1], self.period_per_sample)
            return extended_arb[start_index_ns - arb_start_index * self.period_per_sample:start_index_ns - arb_start_index * self.period_per_sample + duration.in_nanoseconds]

        result = np.zeros((duration.in_nanoseconds, ))
        for i in range(duration.in_nanoseconds):
            if start_timestamp + PreciseDuration(nanoseconds=i) >= arb_start:
                result[i] = self._get_arb_data_in_ns(start_timestamp + PreciseDuration(nanoseconds=i) - arb_start)
            else:
                result[i] = 0.0
        return result