# Author: rushter <me@rushter.com>

import random
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np

from mla.base import Base
from mla.metrics.distance import euclidean_distance


class KMeans(Base):
    def __init__(self, K=5, max_iters=100, init='random'):
        super(KMeans, self).__init__()
        self.K = K
        self.max_iters = max_iters
        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []
        self.init = init

    def _initialize_cetroids(self, init):
        """ Seed the initial centroids """

        if init == 'random':
            self.centroids = [self.X[x] for x in
                              random.sample(range(self.n_samples), self.K)]
        elif init == '++':
            self.centroids = [random.choice(self.X)]
            while len(self.centroids) < self.K:
                self.centroids.append(self._choose_next_center())

        else:
            raise ValueError('Unknown type of init parameter')

    def predict(self):
        self._initialize_cetroids(self.init)
        centroids = self.centroids
        for _ in range(self.max_iters):
            self._assign(centroids)
            centroids_old = centroids
            centroids = [self._get_centroid(cluster) for cluster in self.clusters]

            if self._check_convergence(centroids_old, centroids) == 0:
                break

        self.centroids = centroids

        return self._get_predictions()

    def _get_predictions(self):
        predictions = np.empty(self.n_samples)

        for i, cluster in enumerate(self.clusters):
            for index in cluster:
                predictions[index] = i
        return predictions

    def _assign(self, centroids):

        for row in range(self.n_samples):
            for i, cluster in enumerate(self.clusters):
                if row in cluster:
                    self.clusters[i].remove(row)
                    break

            closest = self._closest(row, centroids)
            self.clusters[closest].append(row)

    def _closest(self, fpoint, centroids):
        closest_index = None
        closest_distance = None
        for i, point in enumerate(centroids):
            dist = euclidean_distance(self.X[fpoint], point)
            if closest_index is None or dist < closest_distance:
                closest_index = i
                closest_distance = dist
        return closest_index

    def _get_centroid(self, cluster):
        """ Get values by indices and take the mean """

        return [np.mean(np.take(self.X[:, i], cluster)) for i in range(self.n_features)]

    def _dist_from_centers(self):
        return np.array([min([euclidean_distance(x, c) for c in self.centroids]) for x in self.X])

    def _choose_next_center(self):
        distances = self._dist_from_centers()
        probs = distances / distances.sum()
        cumprobs = probs.cumsum()
        r = random.random()
        ind = np.where(cumprobs >= r)[0][0]
        return self.X[ind]

    def _check_convergence(self, centroids_old, centroids):
        return sum([euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)])

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
