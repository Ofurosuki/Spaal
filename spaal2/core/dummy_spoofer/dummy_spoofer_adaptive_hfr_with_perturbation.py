import numpy as np
import numpy.typing as npt
import math
from typing import Optional

from spaal2.core.dummy_spoofer import DummySpooferInterface
from spaal2.core.measurement_config import MeasurementConfig, PreciseDuration

class DummySpooferAdaptiveHFRWithPerturbation(DummySpooferInterface):
    """
    Adaptive HFR Spoofer with random time perturbation on each pulse.
    """
    def __init__(self,
                 frequency: float,
                 duration: PreciseDuration,
                 spoofer_distance_m: float,
                 pulse_width: PreciseDuration,
                 perturbation_ns: float,
                 amplitude: float = 9.0,
                 debug: bool = False) -> None:
        """
        Parameters
        ----------
        frequency : float
            The frequency of the pulses in Hz.
        duration : PreciseDuration
            The duration of the attack after being triggered.
        spoofer_distance_m : float
            The distance between the spoofer and the LiDAR in meters.
        pulse_width : PreciseDuration
            The width of a single pulse.
        perturbation_ns : float
            The maximum random time perturbation to add to each pulse, in nanoseconds.
            The perturbation will be in the range [-perturbation_ns, +perturbation_ns].
        debug : bool, optional
            Whether to print debug information, by default False.
        """
        self.frequency = frequency
        self.duration = duration
        self.distance_m = spoofer_distance_m
        self.pulse_width = pulse_width
        self.perturbation_ns = perturbation_ns
        self.amplitude = amplitude
        self.pulse_period_ns = 1 / self.frequency * 1e9
        self.trigger_time: Optional[PreciseDuration] = None
        self.pulse_perturbations: dict[int, float] = {}

        self._precompute_pulse_shape()

    def set_amplitude(self, amplitude: float):
        self.amplitude = amplitude
        self._precompute_pulse_shape()

    def _precompute_pulse_shape(self):
        # Pre-calculate the Gaussian pulse shape
        sigma = self.pulse_width.in_nanoseconds / (2 * np.sqrt(2 * np.log2(2)))
        pulse_x_range = math.ceil(3 * sigma)
        pulse_x = np.arange(-pulse_x_range, pulse_x_range + 1)
        self.pulse_shape = self.amplitude * np.exp(-(pulse_x ** 2) / (2 * sigma ** 2))

    def trigger(self, config: MeasurementConfig, signal: npt.NDArray[np.float64]):
        if self.trigger_time is not None:
            return

        # Apply delay based on distance
        new_signal = np.zeros_like(signal)
        delay = int(self.distance_m / 0.15)
        if delay < signal.size:
            new_signal[delay:] = signal[:-delay]
        signal = new_signal

        # Find the first rising edge to trigger on
        raises = np.flatnonzero(
            (signal[:-1] < 0.5) & (signal[1:] >= 0.5)
        ) + 1
        if raises.size == 0:
            return
        peak_index = raises[0]
        peak_time = config.start_timestamp + PreciseDuration(nanoseconds=peak_index)

        self.trigger_time = peak_time
        self.pulse_perturbations = {} # トリガー時に摂動をリセット
        print(f"Triggered at {self.trigger_time.in_nanoseconds}ns")

    def get_range_signal(self, start_timestamp: PreciseDuration, duration: PreciseDuration) -> npt.NDArray[np.float64]:
        if self.trigger_time is None:
            return np.zeros(duration.in_nanoseconds)

        attack_start_time = self.trigger_time
        attack_end_time = self.trigger_time + self.duration

        request_start_ns = start_timestamp.in_nanoseconds
        request_end_ns = request_start_ns + duration.in_nanoseconds

        attack_start_ns = attack_start_time.in_nanoseconds
        attack_end_ns = attack_end_time.in_nanoseconds

        # If the request window is completely outside the attack window, return zeros
        if request_end_ns <= attack_start_ns or request_start_ns >= attack_end_ns:
            if request_start_ns >= attack_end_ns:
                self.trigger_time = None  # Reset trigger
            return np.zeros(duration.in_nanoseconds)

        output_signal = np.zeros(duration.in_nanoseconds)

        # Determine the range of pulse indices that could fall within the request window
        pulse_half_width = len(self.pulse_shape) // 2
        
        # Start index considers the earliest possible time a pulse could start affecting the window
        start_pulse_idx = math.floor((request_start_ns - attack_start_ns - pulse_half_width - self.perturbation_ns) / self.pulse_period_ns)
        
        # End index considers the latest possible time a pulse could start and still affect the window
        end_pulse_idx = math.ceil((request_end_ns - attack_start_ns + pulse_half_width + self.perturbation_ns) / self.pulse_period_ns)

        for pulse_idx in range(start_pulse_idx, end_pulse_idx):
            ideal_pulse_time = attack_start_ns + pulse_idx * self.pulse_period_ns
            
            # Add random perturbation
            if pulse_idx not in self.pulse_perturbations:
                self.pulse_perturbations[pulse_idx] = np.random.uniform(-self.perturbation_ns, self.perturbation_ns)
            perturbation = self.pulse_perturbations[pulse_idx]
            perturbed_time = round(ideal_pulse_time + perturbation)

            # Only generate pulses that are within the overall attack duration
            if not (attack_start_ns <= perturbed_time < attack_end_ns):
                continue

            # Calculate the start position of the pulse in the output signal array
            pulse_start_in_output = perturbed_time - pulse_half_width - request_start_ns

            # Determine the slices for copying the pulse shape into the output signal
            out_start = max(0, pulse_start_in_output)
            out_end = min(duration.in_nanoseconds, pulse_start_in_output + len(self.pulse_shape))
            
            shape_start = max(0, -pulse_start_in_output)
            shape_end = shape_start + (out_end - out_start)

            if out_start < out_end:
                output_signal[out_start:out_end] += self.pulse_shape[shape_start:shape_end]

        return output_signal