# coding:utf-8
import random
import math
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class KMeans():
    def __init__(self, clusters_count, iters, data):
        self.clusters_count = clusters_count
        self.iters = iters
        self.clusters = [[] for _ in range(clusters_count)]
        self.data = [tuple(d) for d in data]
        self.dimension = len(data)

    def process(self):
        centroids = [self.data[x] for x in
                     random.sample(range(self.dimension), self.clusters_count)]

        for point in self.data:
            index = random.randint(0, self.clusters_count) - 1
            self.clusters[index].append(point)

        for _ in range(self.iters):
            for point in self.data:

                for i, cluster in enumerate(self.clusters):
                    if point in cluster:
                        self.clusters[i].remove(point)

                closest = self.closest(point, centroids)
                self.clusters[closest].append(point)

                centroids = [self.centroid(cluster) for cluster in
                             self.clusters]

        self.centroids = centroids
        return self.clusters

    def closest(self, fpoint, centroids):
        closest_index = None
        closest_distance = None
        for i, point in enumerate(centroids):
            dist = self.distance(fpoint, point)
            if closest_index is None or dist < closest_distance:
                closest_index = i
                closest_distance = dist
        return closest_index

    def centroid(self, cluster):
        return [np.average(column) for column in np.array(cluster).T]

    def distance(self, a, b):
        return math.sqrt(sum(map(lambda x, y: (x - y) * (x - y), a, b)))


if __name__ == '__main__':
    sns.set(style="white")
    # data = pd.read_csv('../datasets/1.csv', header=None, sep=' ')
    data = pd.read_csv('../datasets/2.csv', header=None, sep=',')  # 7
    # data = pd.read_csv('../datasets/3.csv', header=None, sep=',')  # 15
    clusters = 6

    data = data.as_matrix()
    k = KMeans(clusters_count=clusters, iters=50, data=data)
    data = k.process()

    for i, points in enumerate(data):
        points = np.array(points).T
        plt.scatter(points[0], points[1],
                    c=sns.color_palette("hls", clusters)[i])

    for c in k.centroids:
        plt.scatter(c[0], c[1], marker='x', linewidths=10)

    plt.show()
