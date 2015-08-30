# Author: rushter <me@rushter.com>

import numpy as np
from mla.error import InputError

class Base(object):
    def __init__(self):

        self.y_required = False
        self.X = None
        self.y = None

    def fit(self, X, y=None):
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if X.size == 0:
            raise InputError('Number of features must be > 0')

        if len(X.shape) > 2:
            raise InputError('Array of features must be 1d or 2d array')

        if len(X.shape) == 1:
            self.n_samples, self.n_features = 1, X.shape
        else:
            self.n_samples, self.n_features = X.shape

        self.X = X

        if self.y_required:
            if y is None:
                raise InputError('Missed required argument y')

            if not isinstance(y, np.ndarray):
                y = np.array(y)

            if len(y.shape) != 1:
                raise InputError('Labels must be 1d array')

            if y.size == 0:
                raise InputError('Number of features must be > 0')

        self.y = y
