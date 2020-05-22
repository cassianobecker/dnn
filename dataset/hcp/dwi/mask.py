import numpy as np


# from https://dipy.org/documentation/1.1.1./examples_built/reconst_csd/#example-reconst-csd
def get_mask(fa, md):
    wm_mask = (np.logical_or(fa >= 0.4, (np.logical_and(fa >= 0.15, md >= 0.0011)))).astype(np.int)
    return wm_mask
