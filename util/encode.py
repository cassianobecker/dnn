import numpy as np


def one_hot(x, k):
    return np.array(x[:, None] == np.arange(k)[None, :], dtype=int)


def one_hot_to_int(x):
    """

    Parameters
    ----------
    x: torch.tensor; (N samples X C classes)

    Returns
    -------
    torch.tensor; (N samples)
    """
    return x.argmax(dim=1)
