# Author: rushter <me@rushter.com>

import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
from scipy.optimize import fmin_bfgs, fmin_ncg
from scipy.special import expit
from sklearn.metrics import accuracy_score


class LinearRegression():
    def __init__(self, df, shuffle=True):
        if shuffle:
            df = df.reindex(np.random.permutation(df.index))

        self.x, self.y = np.array(df[df.columns[0:-1]]), np.array(
            df.iloc[:, -1])
        self.m, self.n = self.x.shape

        # Sigmoid implementation from scipy
        self.sigmoid = expit

    def predict(self):
        theta_initial = np.zeros((1, self.n))
        theta = fmin_bfgs(self.cost, theta_initial, fprime=self.grad,
                          args=(self.x, self.y))

        # Accuracy for training set
        predicted = map(lambda x: 1 if x >= 0.5 else 0,
                        self.sigmoid(np.dot(self.x, theta)))
        print(accuracy_score(self.y, predicted))

    def plot(self):
        plt.plot(self.x, self.y, 'o')
        plt.show()

    def cost(self, theta, x, y):
        h = self.sigmoid(np.dot(x, theta))
        cost_pos = np.dot(-y.T, np.log(h))
        cost_neg = np.dot((1 - y).T, np.log(1 - h))
        return (cost_pos - cost_neg) / self.m

    def grad(self, theta, x, y):
        prediction = self.sigmoid(np.dot(x, theta))
        return 1.0 / self.m * np.dot(x.T, prediction - y)


if __name__ == '__main__':
    df = pd.read_csv('../datasets/7.csv', header=None, sep=',')
    df = df.dropna()
    model = LinearRegression(df=df)
    model.predict()
    # model.plot()
