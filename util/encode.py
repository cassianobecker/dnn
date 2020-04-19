import numpy as np


def one_hot(x, k):
    return np.array(x[:, None] == np.arange(k)[None, :], dtype=int)
