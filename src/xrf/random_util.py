import numpy as np


def random_color():
    return np.random.choice(range(256), size=3) / 256
