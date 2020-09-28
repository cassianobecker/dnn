import random
import torch
import os
import numpy as np


def seed_everything(seed=1234):
    """
    Sets a random seed for OS, NumPy, PyTorch and CUDA.
    :dwi_params seed: random seed to apply
    :return: None
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
