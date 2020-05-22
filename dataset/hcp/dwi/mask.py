import numpy as np


def get_mask(fa, md):
    wm_mask = (np.logical_or(fa >= 0.4, (np.logical_and(fa >= 0.15, md >= 0.0011)))).astype(np.int)
    return wm_mask
