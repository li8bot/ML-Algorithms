# coding:utf-8
import random
import math
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time


class KMeans:
    def __init__(self, clusters_count, iters, data):
        self.clusters_count = clusters_count
        self.iters = iters
        self.data = data
        self.dimension = len(data)
        self.clusters = [[] for _ in range(clusters_count)]

    def train(self):

        # Initialize centroids randomly
        centroids = [self.data[x] for x in
                     random.sample(range(self.dimension), self.clusters_count)]

        # Assign labels to the clusters randomly
        for point in range(len(self.data)):
            index = random.randint(0, self.clusters_count) - 1
            self.clusters[index].append(point)

        for _ in range(self.iters):
            centroids = self.classify(centroids)

        self.centroids = centroids
        return self.clusters

    def classify(self, centroids):

        for row in range(len(self.data)):
            for i, cluster in enumerate(self.clusters):
                if row in cluster:
                    self.clusters[i].remove(row)

            closest = self.closest(row, centroids)
            self.clusters[closest].append(row)

        centroids = [self.centroid(cluster) for cluster in
                     self.clusters]
        return centroids

    def closest(self, fpoint, centroids):
        closest_index = None
        closest_distance = None
        for i, point in enumerate(centroids):
            dist = self.distance(self.data[fpoint], point)
            if closest_index is None or dist < closest_distance:
                closest_index = i
                closest_distance = dist
        return closest_index

    def centroid(self, cluster):
        # Get values by indexes and take the mean
        return [np.average(np.take(self.data[:, 0], cluster)),
                np.average(np.take(self.data[:, 1], cluster))
                ]

    @staticmethod
    def distance(a, b):
        return math.sqrt(sum((a - b) ** 2))

    def plot(self):
        sns.set(style="white")
        for i, index in enumerate(self.clusters):
            points = np.array(self.data[index]).T
            plt.scatter(points[0], points[1],
                        c=sns.color_palette("hls", self.clusters_count + 1)[i])

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
    k = KMeans(clusters_count=clusters, iters=50, data=data)
    k.train()
    print(time.time() - start)
    k.plot()
