# Author: rushter <me@rushter.com>
import math

from mla.base import Base
from mla.metrics.scoring import squared_error
import numpy as np
import logging
import seaborn as sns
import matplotlib.pyplot as plt


class LinearRegression(Base):
    def __init__(self, max_iters=1000, alpha=0.0001, tolerance=0.0001, gradient_type=0):
        """

        :param alpha: learning rate
        """

        self.gradient_type = gradient_type
        self.tolerance = tolerance
        self.alpha = alpha
        self.max_iters = max_iters
        self.y_required = True
        self.errors = []
        self.theta = []

    def fit(self, X, y=None):
        super(LinearRegression, self).fit(X, y)

        # Weights
        self.theta = np.zeros(shape=(self.n_features + 1, 1))

        # Add an intercept columsn
        self.X = self._add_intercept(self.X)

        self._train()

    def _add_intercept(self, X):
        b = np.ones([X.shape[0], 1])
        return np.concatenate([b, X], axis=1)

    def _train(self):
        self.theta, self.errors = self._gradient_descent(self.X, self.y)
        logging.info(' Theta: %s' % self.theta.flatten())

    def _predict(self, X=None):
        X = self._add_intercept(X)
        return [max(0, x) for x in self._get_predictions(X)]

    def _get_predictions(self, X=None, theta=None):

        if X is None:
            X = self.X

        if theta is None:
            theta = self.theta

        return X.dot(theta).sum(axis=1)

    def _compute_cost(self, X, y, theta):
        prediction = X.dot(theta)
        error = squared_error(y, prediction).sum()
        return error

    def _gradient_descent(self, X, y):
        theta = self.theta
        errors = [self._compute_cost(X, y, theta)]

        if self.gradient_type == 1:
            gradient = self._step_batch_gradient
        elif self.gradient_type == 2:
            gradient = self._step_w_gradient
        else:
            gradient = self._step_gradient

        for i in range(1, self.max_iters + 1):

            theta = gradient(X, y, theta, self.alpha)

            errors.append(self._compute_cost(X, y, theta))
            logging.info('Iteration %s, error %s' % (i, errors[i]))

            # TODO: abs?
            error_diff = (errors[i - 1] - errors[i])

            if error_diff < self.tolerance:
                logging.info('Convergence has reached.')
                break
        return theta, errors

    def _step_gradient(self, X, y, theta, alpha):
        for i in range(self.n_samples):
            prediction_difference = np.dot(X[i, :], theta)[0] - y[i]
            gradient = (2 / self.n_samples) * X[i, :] * prediction_difference
            gradient = np.reshape(gradient, (gradient.size, 1))
            theta = theta - (alpha * gradient)
        return theta

    def _step_batch_gradient(self, X, y, theta, alpha):
        gradient = 0
        for i in range(self.n_samples):
            prediction_difference = np.dot(X[i, :], theta) - y[i]
            gradient += (2 / self.n_samples) * X[i, :] * prediction_difference

        gradient = np.reshape(gradient, (len(gradient), 1))
        theta = theta - (alpha * gradient)
        return theta

    def _step_w_gradient(self, X, y, theta, alpha):

        prediction = self._get_predictions()
        m = X.size
        for it in range(len(theta)):
            temp = X[:, it].copy()
            errors = (prediction - y) * temp
            theta[it] -= alpha * (2 / m) * errors.sum()
        return theta

    def plot(self):
        sns.set(style="white")
        plt.plot([math.log(x) for x in self.errors])
        plt.ylabel('Error rate')
        plt.xlabel('Iteration')
        plt.show()
