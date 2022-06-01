import numpy as np

from xrf.constants import EPSILON


def extract_local_maxima_indices(x: np.ndarray, n: int):
    max_indices = []
    for i in range(n + 1, x.shape[0] - n):
        max_candidate = x[i]
        neighbourhood = np.concatenate([x[i - n : i], x[i + 1 : i + n + 1]])
        neighbourhood_max = neighbourhood.max()
        if max_candidate >= neighbourhood_max >= EPSILON:
            max_indices.append(i)
    return max_indices
