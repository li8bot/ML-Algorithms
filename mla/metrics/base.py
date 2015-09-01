# Author: rushter <me@rushter.com>

import numpy as np


def validate_input(function):
    def wrapper(a, b):
        if isinstance(a, list):
            a = np.array(a)

        if isinstance(b, list):
            b = np.array(b)

        if type(a) != type(b):
            raise ValueError('Type mismatch: %s and %s' % (type(a), type(b)))

        if a.size != b.size:
            raise ValueError('Arrays must be equal in length.')
        return function(a, b)

    return wrapper
