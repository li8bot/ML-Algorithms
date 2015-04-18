# coding:utf-8

import pytest
from func import stats
import numpy as np
from scipy import stats as st


def test_mode():
    assert stats.mode([0]) == 0

    for _ in xrange(10):
        sequence = np.random.choice(5, np.random.randint(2, 10))
        assert stats.mode(sequence) == st.mode(sequence)[0][0]


def test_check_data():
    assert stats.check_data([1, 2, 3])
    assert not stats.check_data([1, 2, 'b'])
    assert not stats.check_data(object)
    assert not stats.check_data([])


def test_median():
    assert stats.median([1]) == 1
    assert stats.median([1, 1]) == 1
    assert stats.median([1, 1, 2, 4]) == 1.5
    assert stats.median([0, 2, 5, 6, 8, 9, 9]) == 6
    assert stats.median([0, 0, 0, 0, 4, 4, 6, 8]) == 2


def test_mean():
    # Test ints
    assert stats.mean([4, 36, 45, 50, 75]) == 42
    # Test floats
    assert stats.mean([17.25, 19.75, 20.0, 21.5, 21.75, 23.25, 25.125, 27.5]) == 22.015625
    assert stats.mean([0]) == 0


