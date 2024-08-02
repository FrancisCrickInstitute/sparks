import numpy as np
import torch


def normalize(x):
    """
    Normalizes input tensor to [0, 1]
    """

    if isinstance(x, np.ndarray):
        return (x - np.min(x)) / (np.max(x) - np.min(x))
    elif isinstance(x, torch.Tensor):
        return (x - torch.min(x)) / (torch.max(x) - torch.min(x))
    else:
        return NotImplementedError


def smooth(x, window=10):
    return np.convolve(x, np.ones(window) / window, mode='same')
