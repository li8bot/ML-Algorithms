# Author: rushter <me@rushter.com>

from mla.datasets import *
from mla.regression.linear import LinearRegression


def boston_example():
    X, y = load_boston()
    model = LinearRegression()
    model.fit(X, y)
    model.predict()


if __name__ == '__main__':
    boston_example()
