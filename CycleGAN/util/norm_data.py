import numpy as np


def norm_01(x):
    return (x - x.min()) / (x.max() - x.min())


def norm(x):
    """Normalize to 0 mean and 1 std per slide"""
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = (x[i, j] - x[i, j].mean()) / x[i, j].std()
    return x
