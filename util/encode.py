import numpy as np


# one_hot = lambda x, k: np.array(x[:, None] == np.arange(k)[None, :], dtype=int)
def one_hot(x, k):
    return np.array(x[:, None] == np.arange(k)[None, :], dtype=int)
