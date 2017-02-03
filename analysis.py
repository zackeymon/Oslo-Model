import numpy as np


def moving_average(data, temporal_window=25):
    smoothed_data = list(data)
    for i in range(temporal_window, len(data) - temporal_window):
        smoothed_data[i] = sum(data[i - temporal_window: i + temporal_window]) / (2 * temporal_window + 1)
    return smoothed_data


def np_moving_average(data, temporal_window=25):
    window = np.ones(temporal_window) / temporal_window
    return np.convolve(data, window, 'same')[:-temporal_window]
