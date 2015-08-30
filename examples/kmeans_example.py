# coding: utf-8
from mla.kmeans import KMeans
import pandas as pd
import timeit


def run_example():
    # data = pd.read_csv('../datasets/1.csv', header=None, sep=' ')
    data = pd.read_csv('../datasets/2.csv', header=None, sep=',')  # 7
    # data = pd.read_csv('../datasets/3.csv', header=None, sep=',')  # 15
    clusters = 5
    data = data.as_matrix()
    k = KMeans(K=clusters, max_iters=50, init='++')
    k.fit(data)
    k.predict()
    return k


if __name__ == '__main__':
    print(timeit.timeit(run_example, number=1))
    k = run_example()
    k.plot()
