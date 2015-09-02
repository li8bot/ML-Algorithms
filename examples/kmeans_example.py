# Author: rushter <me@rushter.com>

from mla.kmeans import KMeans
import pandas as pd
from mla.datasets import load_iris, load_robust
import numpy as np
import timeit


def robust_example(plot=False):
    X, y = load_robust()
    clusters = 5
    k = KMeans(K=clusters, max_iters=50, init='++')
    k.fit(X)
    k.predict()

    if plot:
        k.plot()


def iris_example(plot=False):
    X, y = load_iris()
    clusters = len(np.unique(y))
    k = KMeans(K=clusters, max_iters=50, init='++')
    k.fit(X, y)
    k.predict()
    data = np.zeros([k.n_samples, 2])

    # Dimension reducing
    # Sepal width*length
    data[:, 0] = k.X[:, 0] * k.X[:, 1]

    # Petal width*length
    data[:, 1] = k.X[:, 2] * k.X[:, 3]

    if plot:
        k.plot(data)


if __name__ == '__main__':
    print(timeit.timeit(iris_example, number=1))
    print(timeit.timeit(robust_example, number=1))
    robust_example(plot=True)

    # print(timeit.timeit(run_example_one, number=1))
    # k = run_example_one()
    # k.plot()
