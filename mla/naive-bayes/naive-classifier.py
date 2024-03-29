# Author: rushter <me@rushter.com>

from functools import reduce
import math
import operator
import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


class Naive:
    def __init__(self, X, y):

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X.values, y.values,
                                                                                test_size=0.25
                                                                                )

        self.classes = y.unique()
        self.features = X.shape[1]
        self.X_train = self.split_by_class(self.X_train, self.y_train)

    def predict(self):
        predictions = []
        for v in self.X_test:
            class_probabilities = {}
            for i in self.classes:
                prob = sum([math.log(self.normpdf(v[j],
                                                  self.summaries[i][j][0],
                                                  self.summaries[i][j][1]))
                            for j in range(self.features)])
                class_probabilities[i] = prob
            cls = max(class_probabilities.items(), key=operator.itemgetter(1))[0]
            predictions.append(cls)
        self.predictions = predictions

    def train(self):
        self.summaries = {}
        for i in self.classes:
            self.summaries[i] = {}
            for j in range(self.features):
                self.summaries[i][j] = self.mean_std(self.X_train[i][:, j])

    def mean_std(self, x):
        return np.mean(x), np.std(x)

    def normpdf(self, x, mean, sd):
        var = float(sd) ** 2
        denom = (2 * math.pi * var) ** .5
        num = math.exp(-(float(x) - float(mean)) ** 2 / (2 * var))
        return num / denom

    def split_by_class(self, X, y):
        splits = {}
        for i in self.classes:
            splits[i] = (X[y == i])
        return splits

    def validate(self):
        print(accuracy_score(self.y_test, self.predictions))

    def process(self):
        self.train()
        self.predict()
        self.validate()


if __name__ == '__main__':
    df = pd.read_csv('../datasets/pima-indians.csv')
    y = df['class']
    df.drop('class', 1, inplace=True)
    X = df
    for i in range(10):
        n = Naive(X, y)
        n.process()
