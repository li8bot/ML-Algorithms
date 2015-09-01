# Author: rushter <me@rushter.com>

from __future__ import division
import pytest
import math
import numpy as np
from mla.metrics.distance import euclidean_distance
from mla.metrics.base import check_data
from mla.metrics.scoring import *


def test_euclidean_distance():
    assert euclidean_distance([1, 2, 3], [3, 2, 1]) == math.sqrt(8)
    assert euclidean_distance([1, 2, 1], [3, 2, 1]) != math.sqrt(8)

    with pytest.raises(ValueError):
        euclidean_distance([1, 2], [3, 2, 1]) != math.sqrt(8)


def test_data_validation():
    with pytest.raises(ValueError):
        check_data([], 1)

    with pytest.raises(ValueError):
        check_data([1, 2, 3], [3, 2])

    a, b = check_data([1, 2, 3], [3, 2, 1])

    assert np.all(a == np.array([1, 2, 3]))
    assert np.all(b == np.array([3, 2, 1]))


def test_classification_error():
    assert classification_error([1, 2, 3, 4], [1, 2, 3, 4]) == 0
    assert classification_error([1, 2, 3, 4], [1, 2, 3, 5]) == 0.25
    assert classification_error([1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0]) == (1.0 / 6)


def test_absolute_error():
    assert absolute_error([1], [1]) == [0]
    assert absolute_error([3], [5]) == [2]
    assert absolute_error([-1], [-4]) == [3]


def test_mean_absolute_error():
    assert mean_absolute_error([1, 2, 3], [1, 2, 3]) == 0
    assert mean_absolute_error([1, 2, 3], [3, 2, 1]) == 4 / 3


def test_squared_error():
    assert squared_error([1], [1]) == [0]
    assert squared_error([3], [1]) == [4]


def test_squared_log_error():
    assert squared_log_error([1], [1]) == [0]
    assert squared_log_error([3], [1]) == [np.log(2) ** 2]
    assert squared_log_error([np.exp(2) - 1], [np.exp(1) - 1]) == [1.0]


def test_mean_squered_error():
    assert mean_squared_log_error([1, 2, 3], [1, 2, 3]) == 0
    assert mean_squared_log_error([1, 2, 3, np.exp(1) - 1], [1, 2, 3, np.exp(2) - 1]) == 0.25


def test_root_mean_squared_log_error():
    assert root_mean_squared_log_error([1, 2, 3], [1, 2, 3]) == 0
    assert root_mean_squared_log_error([1, 2, 3, np.exp(1) - 1], [1, 2, 3, np.exp(2) - 1]) == 0.5


def test_mean_squared_error():
    assert mean_squared_error([1, 2, 3], [1, 2, 3]) == 0
    assert mean_squared_error(range(1, 5), [1, 2, 3, 6]) == 1


def test_root_mean_squared_error():
    assert root_mean_squared_error([1, 2, 3], [1, 2, 3]) == 0
    assert root_mean_squared_error(range(1, 5), [1, 2, 3, 5]) == 0.5


def test_multiclass_logloss():
    np.testing.assert_almost_equal(logloss([1], [1]), 0)
    np.testing.assert_almost_equal(logloss([1, 1], [1, 1]), 0)
    np.testing.assert_almost_equal(logloss([1], [0.5]), -np.log(0.5))
