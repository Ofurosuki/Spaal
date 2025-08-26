import numpy as np
import numpy.typing as npt
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

from spaal2.core.dummy_lidar.echo import Echo

def accumulate_signal_list(signal_list: list[npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
    """
    複数の信号を積算する
    """
    if not signal_list:
        return np.array([])
    return np.sum(signal_list, axis=0)

def detect_echo(
    signal: npt.NDArray[np.float64], 
    max_height: float = 9.0, 
    thd_factor: float = 1.0, 
    use_height_estimation: bool = False, 
    pulse_half_width_ns: int = 10, 
    show_plot: bool = False,
    time_resolution_ns: float = 1.0
) -> list[Echo]:
    # ----------------------- パラメータ計算 -----------------------
    b_one = float(np.mean(signal))
    echo_judgment_threshold1 = b_one + (max_height/2 - b_one * 0.5) * thd_factor
    echo_judgment_threshold2 = b_one + np.sqrt(b_one * (max_height - b_one)) * thd_factor

    # 判定閾値を切り替えるサンプル位置 (100nsを時間に変換)
    echo_judgement_position = int(100 / time_resolution_ns)
    # Ensure position is within signal bounds
    echo_judgement_position = min(echo_judgement_position, len(signal) -1)

    echo_detection_threshold = b_one
    echo_width_threshold = 0
    valley_threshold = 1

    if show_plot:
        plt.figure(figsize=(12, 4))
        plt.cla()
        plt.plot(signal, color='blue')
        plt.hlines(y=echo_judgment_threshold1, xmin=0, xmax=echo_judgement_position, color='red', linestyle='--')
        plt.hlines(y=echo_judgment_threshold2, xmin=echo_judgement_position, xmax=len(signal), color='red', linestyle='--')

    # ----------------------- エコー検出 -----------------------
    peaks_part1 = find_peaks(signal[:echo_judgement_position], height=echo_judgment_threshold1)[0]
    peaks_part2 = find_peaks(signal[echo_judgement_position:], height=echo_judgment_threshold2)[0] + echo_judgement_position
    peaks_global = np.hstack([peaks_part1, peaks_part2])
    
    peaks_binary = np.zeros_like(signal)
    if len(peaks_global) > 0:
        peaks_binary[peaks_global] = signal[peaks_global]

    if show_plot:
        plt.scatter(peaks_global, peaks_binary[peaks_global], color='green')

    raises = np.flatnonzero((signal[:-1] < echo_detection_threshold) & (signal[1:] >= echo_detection_threshold)) + 1
    falls = np.flatnonzero((signal[:-1] >= echo_detection_threshold) & (signal[1:] < echo_detection_threshold)) + 1
    
    if len(raises) == 0 and len(falls) == 0: return []
    if len(raises) == 0 or (len(falls) > 0 and raises[0] > falls[0]): raises = np.insert(raises, 0, 0)
    if len(falls) == 0 or (len(raises) > 0 and raises[-1] > falls[-1]): falls = np.append(falls, len(signal) - 1)
    
    effective_echoes: list[Echo] = []
    for echo_start, echo_end in zip(raises, falls):
        sig_echo = signal[echo_start:echo_end]
        if sig_echo.size == 0: continue

        if show_plot: plt.hlines(y=echo_detection_threshold, xmin=echo_start, xmax=echo_end, color='green', linestyle='--')

        peaks_local = np.flatnonzero(peaks_binary[echo_start:echo_end])
        if len(peaks_local) == 0: continue

        split_positions = []
        for i in range(len(peaks_local)-1):
            if sig_echo[peaks_local[i+1]] - np.min(sig_echo[peaks_local[i]:peaks_local[i+1]]) > valley_threshold:
                split_positions.append(np.argmin(sig_echo[peaks_local[i]:peaks_local[i+1]]) + peaks_local[i])
                if show_plot: plt.axvline(x=split_positions[-1]+echo_start, color='purple', linestyle='--')
        
        candidate_echo_range_list = []
        if not split_positions:
            candidate_echo_range_list.append((0, len(sig_echo)))
        else:
            split_points = [0] + split_positions + [len(sig_echo)]
            for i in range(len(split_points)-1):
                candidate_echo_range_list.append((split_points[i],split_points[i+1]))

        for c_echo_start, c_echo_end in candidate_echo_range_list:
            if c_echo_start >= c_echo_end: continue
            peaks_range = peaks_binary[echo_start+c_echo_start:echo_start+c_echo_end]
            if peaks_range.size == 0: continue
            peak_position = int(np.argmax(peaks_range) + echo_start + c_echo_start)
            peak_height = signal[peak_position]
            width = c_echo_end - c_echo_start
            if width >= echo_width_threshold:
                effective_echoes.append(Echo(peak_position, peak_height, width, signal[echo_start+c_echo_start:echo_start+c_echo_end]))

            if show_plot:
                plt.hlines(y=echo_detection_threshold+0.5, xmin=echo_start+c_echo_start, xmax=echo_start+c_echo_end, color='orange', linestyle='--')
                plt.scatter(peak_position, peak_height + 0.1, color='orange')

    if show_plot: plt.show()

    if use_height_estimation:
        for echo in effective_echoes:
            if echo.peak_height < max_height - 0.1: continue
            thd = max_height - 0.1
            sigma = pulse_half_width_ns / (2 * np.sqrt(2 * np.log2(2)))
            start = np.argmax(echo.signal > thd)
            end = len(echo.signal) - np.argmax(echo.signal[::-1] > thd)
            if end <= start: continue
            if np.sum(echo.signal[start:end] > thd) / (end - start) < 0.8: continue
            saturation_width = end - start
            estimated_height = thd / np.exp( - ((saturation_width / 2) ** 2) / (2 * sigma ** 2))
            echo.peak_height = estimated_height

    return effective_echoes