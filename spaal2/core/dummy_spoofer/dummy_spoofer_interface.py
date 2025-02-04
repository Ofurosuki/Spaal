from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt

from spaal2.core.measurement_config import MeasurementConfig, PreciseDuration

class DummySpooferInterface(ABC):
    @abstractmethod
    def trigger(self, config: MeasurementConfig, signal: npt.NDArray[np.float64]):
        pass

    @abstractmethod
    def get_range_signal(self, start_timestamp: PreciseDuration, duration: PreciseDuration) -> npt.NDArray[np.float64]:
        pass
