# coding: utf-8
# Author: rushter <me@rushter.com>

import numpy as np


class InputError(Exception):
    pass


class Base:
    def __init__(self):
        self.y_required = False

        self.X = None
        self.y = None

    def fit(self, X, y=None):
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if X.size == 0:
            raise InputError('Number of features must be > 0')
        else:
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
