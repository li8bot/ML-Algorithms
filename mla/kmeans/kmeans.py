# coding:utf-8
import random
import math
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time


# TODO: automatic convergence

class KMeans:

    def __init__(self, K, max_iters, X, init='random'):
        self.K = K
        self.X = X
        self.max_iters = max_iters
        self.samples_count, self.features_count = X.shape
        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []
        self._initialize_cetroids(init=init)

    def _dist_from_centers(self):
        return np.array([min([np.linalg.norm(x - c) ** 2 for c in self.centroids]) for x in self.X])

    def _choose_next_center(self):
        d2 = self._dist_from_centers()
        probs = d2 / d2.sum()
        cumprobs = probs.cumsum()
        r = random.random()
        ind = np.where(cumprobs >= r)[0][0]
        return self.X[ind]

    def _initialize_cetroids(self, init):
        # Seed the initial centroids

        if init == 'random':
            self.centroids = [self.X[x] for x in
                              random.sample(range(self.samples_count), self.K)]
        elif init == '++':
            self.centroids = [random.choice(self.X)]
            while len(self.centroids) < self.K:
                self.centroids.append(self._choose_next_center())

    def train(self):
        centroids = self.centroids

        for _ in range(self.max_iters):
            self._assign(centroids)
            centroids = [self._get_centroid(cluster) for cluster in self.clusters]

        self.centroids = centroids

    def _assign(self, centroids):

        for row in range(self.samples_count):
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
            dist = self._distance(self.X[fpoint], point)
            if closest_index is None or dist < closest_distance:
                closest_index = i
                closest_distance = dist
        return closest_index

    def _get_centroid(self, cluster):
        # Get values by indices and take the mean
        return [np.mean(np.take(self.X[:, 0], cluster)),
                np.mean(np.take(self.X[:, 1], cluster))
                ]

    @staticmethod
    def _distance(a, b):
        return math.sqrt(sum((a - b) ** 2))

    def _check_convergence(self):
        pass

    def plot(self):
        sns.set(style="white")
        for i, index in enumerate(self.clusters):
            points = np.array(self.X[index]).T
            plt.scatter(points[0], points[1],
                        c=sns.color_palette("hls", self.K + 1)[i])

        for c in self.centroids:
            plt.scatter(c[0], c[1], marker='x', linewidths=10)

        plt.show()


if __name__ == '__main__':
    # data = pd.read_csv('../datasets/1.csv', header=None, sep=' ')
    data = pd.read_csv('../datasets/2.csv', header=None, sep=',')  # 7
    # data = pd.read_csv('../datasets/3.csv', header=None, sep=',')  # 15
    clusters = 5
    data = data.as_matrix()
    start = time.time()
    k = KMeans(K=clusters, max_iters=50, X=data, init='++')
    k.train()
    print(time.time() - start)
    k.plot()
