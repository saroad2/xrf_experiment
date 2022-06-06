from typing import List

import numpy as np
from scipy.optimize import curve_fit


def trim_gaussian(x: np.ndarray, y: np.ndarray, start_x: List[int]):
    start_x.sort()
    start_indices = np.where(np.isin(x, start_x))[0]
    start_y = y[start_indices]
    max_index = start_indices[-1] + 1
    while max_index < y.shape[0] and y[max_index] > start_y[-1] / 2:
        max_index += 1
    min_index = start_indices[0] - 1
    while min_index >= 0 and y[min_index] > start_y[0] / 2:
        min_index -= 1
    return x[min_index:max_index], y[min_index:max_index]


def gaussian(x, *args):
    s = 0
    for i in range(len(args) // 3):
        a, mean, std = args[3 * i], args[3 * i + 1], args[3 * i + 2]
        s += a * np.exp(-(((x - mean) / std) ** 2))
    return s


def fit_gaussian(x, y, mean_candidates):
    p0 = []
    for mean_candidate in mean_candidates:
        std_candidate = (x[-1] - x[0]) / (2 * np.sqrt(2 * np.log(2)))
        a_candidate = y[x == mean_candidate][0]
        p0.extend([a_candidate, mean_candidate, std_candidate])
    res, cov = curve_fit(gaussian, x, y, p0=p0)
    return res, cov
