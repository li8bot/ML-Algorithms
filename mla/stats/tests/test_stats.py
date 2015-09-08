# Author: rushter <me@rushter.com>

import math

import pytest

from mla.stats import base


def test_mode():
    assert base.mode([0]) == 0
    assert base.mode([1, 2, 2, 3, 4, 7, 9]) == 2
    assert base.mode([1, 2, 2, 3, 3, 4, 4]) == 2


def test_check_data():
    assert base.check_data([1, 2, 3])

    with pytest.raises(ValueError):
        base.check_data([1, 2, 'b'])
    with pytest.raises(TypeError):
        base.check_data(object)
    with pytest.raises(TypeError):
        base.check_data([])
    with pytest.raises(TypeError):
        base.check_data(True)


def test_median():
    assert base.median([1]) == 1
    assert base.median([1, 1]) == 1
    assert base.median([1, 1, 2, 4]) == 1.5
    assert base.median([0, 2, 5, 6, 8, 9, 9]) == 6
    assert base.median([0, 0, 0, 0, 4, 4, 6, 8]) == 2


def test_mean():
    assert base.mean([4, 36, 45, 50, 75]) == 42
    assert base.mean([4, 36, 45, 50, 71]) == 41.200000000000003
    assert base.mean(
        [17.25, 19.75, 20.0, 21.5, 21.75, 23.25, 25.125, 27.5]) == 22.015625

    assert base.mean([0]) == 0


def test_arange():
    assert base.arange([4, 6, 9, 3, 7]) == 6
    assert base.arange([3.2, 6.1, 9.1, 32.2, 71.1]) == 67.899999999999991

    assert base.arange([1, 1]) == 0


def test_variance():
    assert base.variance([9, 1, 1, 1, 0, 0, 4, 16]) == 32.571428571428569
    assert base.variance(
        [2.75, 1.75, 1.25, 0.25, 0.5, 1.25, 3.5]) == 1.3720238095238095

    assert base.variance([1, 1]) == 0
    assert not base.variance([0])


def test_stdev():
    assert base.stdev([9, 1, 1, 1, 0, 0, 4, 16]) == math.sqrt(
        32.571428571428569)
    assert base.stdev([2.75, 1.75, 1.25, 0.25, 0.5, 1.25, 3.5]) == math.sqrt(
        1.3720238095238095)


def test_standard_error():
    assert base.standard_error(
        [1, 2, 3, 4, 5, 6, 7, 8]) == 0.86602540378443849
    assert base.standard_error([1.1, 3.4, 5.8, 9.34]) == 1.7609940374686108

    assert base.standard_error([0, 0]) == 0


def test_gmean():
    assert base.gmean([1, 2, 3, 4, 5, 6]) == 2.993795165523909
    assert base.gmean([3.5, 4.1, 5.8, 9.2]) == 5.2603777299731815


def test_hmean():
    assert base.hmean([1, 2, 3, 4, 5, 6]) == 2.4489795918367347
    assert base.hmean([11.1, 71.1, 0.1, 26.1, 5.1, 12.1, 19.1, 62.1, 80.1,
                        7.1]) == 0.9395864189821148
