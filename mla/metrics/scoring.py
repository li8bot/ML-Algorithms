# Author: rushter <me@rushter.com>

from __future__ import division
import numpy as np
from mla.metrics.base import validate_input

#TODO: get rid of multiple calls @validate_input

@validate_input
def absolute_error(actual, predicted):
    return np.abs(actual - predicted)


@validate_input
def classification_error(actual, predicted):
    return sum(actual != predicted) / len(actual)


@validate_input
def mean_absolute_error(actual, predicted):
    return np.mean(absolute_error(actual, predicted))


@validate_input
def squared_error(actual, predicted):
    return (actual - predicted) ** 2


@validate_input
def squared_log_error(actual, predicted):
    return (np.log(np.array(actual) + 1) - np.log(np.array(predicted) + 1)) ** 2


@validate_input
def mean_squared_log_error(actual, predicted):
    return np.mean(squared_log_error(actual, predicted))


@validate_input
def mean_squared_error(actual, predicted):
    return np.mean(squared_error(actual, predicted))


@validate_input
def root_mean_squared_error(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))


@validate_input
def root_mean_squared_log_error(actual, predicted):
    return np.sqrt(mean_squared_log_error(actual, predicted))


@validate_input
def logloss(actual, predicted):
    epsilon = 1e-15

    pred = np.array([max(epsilon, pred) for pred in predicted])
    pred = np.array([min(1 - epsilon, pred) for pred in predicted])

    ll = sum(actual * np.log(pred) + (1-actual) * np.log(1-pred))
    ll = ll * -1.0 / len(actual)
    return ll
