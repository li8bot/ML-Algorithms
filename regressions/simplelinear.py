# coding:utf-8
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np



class LinearRegression():
    def __init__(self, data, degree=3):
        self.x, self.y = data
        self.deegree = degree

    def process(self):
        # Least squares polynomial fit
        self.func = sp.poly1d(sp.polyfit(self.x, self.y, self.deegree))
        self.plot()

    def plot(self):
        fx = sp.linspace(0, self.x[-1], 1000)
        plt.plot(fx, self.func(fx), linewidth=4)
        plt.plot(self.x, self.y, 'o')
        # plt.legend(["d=%i" % self.deegree], loc="upper left")
        plt.legend(["error=%i" % self.error()], loc="upper right")
        plt.show()

    def error(self):
        return np.linalg.norm(self.func(self.x)-self.y)/np.sqrt(len(self.y))


if __name__ == '__main__':
    df = pd.read_csv('../datasets/4.csv', header=None, sep=',')
    df = df.dropna()
    model = LinearRegression(data=[list(df[0]), list(df[1])], degree=1)
    model.process()

