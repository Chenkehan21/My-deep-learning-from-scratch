import numpy as np


def mse_loss(x, y):
    return 0.5 * np.sum((x - y)**2)


def cross_entropy_error(x, y):
    delta = 1e-7 # avoid log(0) !
    return -np.sum(x * np.log(y + delta))


def cross_entropy_loss_one_hot(x, y):
    delta = 1e-7
    if x.ndim == 1:
        x = x.reshape(1, x.size)
        y = y.reshape(1, y.size)
    n = x.shape[0]
    return -np.sum(y * np.log(x + delta)) / n

# important!!!
def cross_entropy_loss(x, y):
    delta = 1e-7
    if x.ndim == 1:
        x = x.reshape(1, x.size)
        y = y.reshape(1, y.size)
    n = x.shape[0]
    # according to "one-hot" version, we only need to consider right labels' outputs!
    return -np.sum(np.log(x[np.arange(n), y] + delta)) / n