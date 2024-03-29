# Author: rushter <me@rushter.com>

import numpy as np


class Base(object):
    X = None
    y = None
    y_required = False

    def fit(self, X, y=None):
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if X.size == 0:
            raise ValueError('Number of features must be > 0')

        if len(X.shape) > 2:
            raise ValueError('Array of features must be 1d or 2d array')

        if len(X.shape) == 1:
            self.n_samples, self.n_features = 1, X.shape
        else:
            self.n_samples, self.n_features = X.shape

        self.X = X

        if self.y_required:
            if y is None:
                raise ValueError('Missed required argument y')

            if not isinstance(y, np.ndarray):
                y = np.array(y)

            if len(y.shape) != 1:
                raise ValueError('Labels must be 1d array')

            if y.size == 0:
                raise ValueError('Number of features must be > 0')

        self.y = y

    def predict(self, X=None):
        if self.X is not None:
            return self._predict(X)
        else:
            raise ValueError('You must call `fit` before `predict`')

    def _predict(self, X=None):
        pass

    def _train(self):
        pass
