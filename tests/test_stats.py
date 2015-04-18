# coding:utf-8
from func import stats


def test_mode():
    assert stats.mode([0]) == 0
    assert stats.mode([1, 2, 2, 3, 4, 7, 9]) == 2
    assert stats.mode([1, 2, 2, 3, 3, 4, 4]) == 2


def test_check_data():
    assert stats.check_data([1, 2, 3])
    assert not stats.check_data([1, 2, 'b'])
    assert not stats.check_data(object)
    assert not stats.check_data([])
    assert not stats.check_data(True)


def test_median():
    assert stats.median([1]) == 1
    assert stats.median([1, 1]) == 1
    assert stats.median([1, 1, 2, 4]) == 1.5
    assert stats.median([0, 2, 5, 6, 8, 9, 9]) == 6
    assert stats.median([0, 0, 0, 0, 4, 4, 6, 8]) == 2


def test_mean():
    # Test ints
    assert stats.mean([4, 36, 45, 50, 75]) == 42
    assert stats.mean([4, 36, 45, 50, 71]) == 41.200000000000003

    # Test floats
    assert stats.mean([17.25, 19.75, 20.0, 21.5, 21.75, 23.25, 25.125, 27.5]) == 22.015625

    assert stats.mean([0]) == 0


def test_range():
    # Test ints
    assert stats.range([4, 6, 9, 3, 7]) == 6

    # Test floats
    assert stats.range([3.2, 6.1, 9.1, 32.2, 71.1]) == 67.899999999999991

    assert stats.range([1, 1]) == 0


def test_variance():
    # Test ints
    assert stats.variance([9, 1, 1, 1, 0, 0, 4, 16]) == 32.571428571428569

    # Test floats
    assert stats.variance([2.75, 1.75, 1.25, 0.25, 0.5, 1.25, 3.5]) == 1.3720238095238095

    assert stats.variance([1, 1]) == 0
    assert not stats.variance([0])
