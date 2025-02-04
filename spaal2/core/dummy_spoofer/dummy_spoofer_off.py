import numpy as np
import numpy.typing as npt

from spaal2.core.dummy_spoofer import DummySpooferInterface
from spaal2.core.measurement_config import MeasurementConfig, PreciseDuration

class DummySpooferOff(DummySpooferInterface):
    """
    何もしないSpoofer
    """
    def __init__(self) -> None:
        pass

    def trigger(self, config: MeasurementConfig, signal: npt.NDArray[np.float64]):
        pass

    def get_range_signal(self, start_timestamp: PreciseDuration, duration: PreciseDuration) -> npt.NDArray[np.float64]:
        return np.zeros((duration.in_nanoseconds, ))
