# coding:utf-8
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
from scipy.special import expit

class LinearRegression():
    def __init__(self, data):
        self.data = data
        self.sigmoid = expit

    def plot(self):
        plt.scatter(df[0], df[1])
        plt.show()

    def process(self):
        pass

    def plot(self):
        fx = sp.linspace(0, self.x[-1], 1000)

        plt.plot(fx, self.func(fx), linewidth=4)
        plt.plot(self.x, self.y, 'o')
        plt.legend(["error=%i" % self.error()], loc="upper right")
        plt.show()

    def error(self):
        return np.linalg.norm(self.func(self.x)-self.y)/np.sqrt(len(self.y))


if __name__ == '__main__':
    df = pd.read_csv('../datasets/7.csv', header=None, sep=',')
    df = df.dropna()
    model = LinearRegression(data=df)
    model.process()

