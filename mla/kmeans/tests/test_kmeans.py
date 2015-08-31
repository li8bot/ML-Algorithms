# Author: rushter <me@rushter.com>

import pytest
from mla.kmeans import KMeans
import numpy as np

data = np.random.rand(10, 2)


def test_initialization():
    with pytest.raises(ValueError):
        kmeans = KMeans(init='test', K=2)
        kmeans.fit(data)
        kmeans._initialize_cetroids('test')

    kmeans = KMeans(init='random', K=2)
    kmeans.fit(data)
    kmeans._initialize_cetroids('random')
    assert len(kmeans.centroids) == kmeans.K

    kmeans = KMeans(init='++', K=2)
    kmeans.fit(data)
    kmeans._initialize_cetroids('++')
    assert len(kmeans.centroids) == kmeans.K


# def test_predict():
#     kmeans = KMeans(init='random', K=2)
#     kmeans.fit(data)
#     assert len(kmeans.clusters) == kmeans.K
