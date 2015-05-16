# coding:utf-8
import math
import operator


def check_data(data):
    if not isinstance(data, list):
        data = list(data)

    if not all(isinstance(item, int) or isinstance(item, float) for item in
               data):
        raise ValueError('sequence can only contain integers or floats')

    if len(data) == 0:
        raise TypeError('function requires at least two data point')

    return data


# Central tendency

def mode(data):
    data = check_data(data)
    m = max([data.count(a) for a in data])

    if m > 1:
        for val in sorted(data):
            if data.count(val) == m:
                return val
    else:
        return min(data)


def median(data):
    data = check_data(data)
    data = sorted(data)

    if len(data) % 2 == 1:
        return data[((len(data) + 1) / 2) - 1]

    if len(data) % 2 == 0:
        return float(sum(data[(len(data) / 2) - 1:(len(data) / 2) + 1])) / 2.0


def mean(data):
    data = check_data(data)
    return sum(data) / float(len(data))


def gmean(data):
    """ Geometric mean """
    return (reduce(operator.mul, data)) ** (1.0 / len(data))


def range(data):
    return max(data) - min(data)


def variance(data, f=None, ddof=1):
    data = check_data(data)

    if len(data) < 2:
        return None

    if f is None:
        f = mean(data)

    return sum((f - x) ** 2 for x in data) / float(len(data) - ddof)


# Standard deviation
def stdev(data, f=None):
    data = check_data(data)
    v = variance(data, f)
    return math.sqrt(v)


# Standard error of the mean
def standard_error(data, f=None):
    data = check_data(data)
    return stdev(data, f) / math.sqrt(len(data))
