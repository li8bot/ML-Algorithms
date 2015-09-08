# Author: rushter <me@rushter.com>
import numpy as np


def normalize(X):
    X_norm = np.zeros(X.shape)
    for i in range(X.shape[1]):
        m = np.mean(X[:, i])
        s = np.std(X[:, i])
        X_norm[:, i] = (X_norm[:, i] - m) / s

    return X_norm
