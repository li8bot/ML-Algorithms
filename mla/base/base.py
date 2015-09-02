# Author: rushter <me@rushter.com>

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Base(object):
    def __init__(self):

        self.y_required = False
        self.X = None
        self.y = None

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

    def plot(self, data=None):
        sns.set(style="white")

        if data is None:
            data = self.X

        if len(data[0]) > 2:
            # TODO: Implement TSNE
            from sklearn.manifold import TSNE

            decomposition = TSNE(n_components=2)
            data = decomposition.fit_transform(self.X)

        for i, index in enumerate(self.clusters):
            point = np.array(data[index]).T
            plt.scatter(*point, c=sns.color_palette("hls", self.K + 1)[i])

        for point in self.centroids:
            plt.scatter(*point, marker='x', linewidths=10)

        plt.show()
