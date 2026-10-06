import numpy as np


def features(x, cycle):
    """Per sensor last, mean, std, min, max and slope over the window, plus the cycle count."""
    t = np.arange(x.shape[1]) - (x.shape[1] - 1) / 2
    slope = (t[None, :, None] * (x - x.mean(1, keepdims=True))).sum(1) / (t ** 2).sum()
    return np.hstack([x[:, -1], x.mean(1), x.std(1), x.min(1), x.max(1), slope, cycle[:, None]])
