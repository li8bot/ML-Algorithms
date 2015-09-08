# Author: rushter <me@rushter.com>
from sklearn.cross_validation import train_test_split
from mla.datasets import *
from mla.metrics.scoring import *
from mla.linear.regression import LinearRegression
import logging

logging.basicConfig(level=logging.DEBUG)


def boston_example():
    X, y = load_boston()
    # X, y = load_lin()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1111)

    # # L = LinearRegressionSK()
    # L = SGDRegressor(n_iter=10000, learning_rate=0.0001)
    #
    # L.fit(X_train, y_train)
    # predictions = L.predict(X_test)
    # print(predictions)
    # print(root_mean_squared_log_error(y_test, predictions))

    model = LinearRegression(alpha=0.000001, max_iters=10000, gradient_type=1)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(root_mean_squared_log_error(y_test, predictions))


if __name__ == '__main__':
    boston_example()
