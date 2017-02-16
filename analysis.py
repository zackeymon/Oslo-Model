import numpy as np
from collections import Counter, OrderedDict
from log_bin import log_bin


def np_moving_average(data, temporal_window=50):
    window = np.ones(temporal_window) / temporal_window
    return np.convolve(data, window, 'valid')


def calculate_height_probability(height_data):
    total_time = len(height_data)
    height_frequencies = sorted(Counter(height_data).items())
    height_probabilities = OrderedDict()
    for (key, value) in height_frequencies:
        height_probabilities[key] = value / total_time
    return height_probabilities


def calculate_avalanche_probability(avalanche_data):
    return calculate_height_probability(avalanche_data)


def make_log_bins(data, bin_scaling=1.5):
    centres, probabilities = log_bin(data, a=bin_scaling, drop_zeros=False)
    return centres, probabilities
