import numpy as np
from scipy.optimize import curve_fit


def gaussian(x, *args):
    s = 0
    for i in range(len(args) // 3):
        a, mean, std = args[3 * i], args[3 * i + 1], args[3 * i + 2]
        s += a * np.exp(-(((x - mean) / std) ** 2))
    return s


def fit_gaussian(x, y, peak_indices):
    p0 = []
    for peak_index in peak_indices:
        a_candidate = y[peak_index]
        mean_candidate = x[peak_index]
        std_candidate = (x[-1] - x[0]) / (2 * np.sqrt(2 * np.log(2)))
        p0.extend([a_candidate, mean_candidate, std_candidate])
    res, cov = curve_fit(gaussian, x, y, p0=p0)
    return res, cov
