# Author: rushter <me@rushter.com>

import pytest
import numpy as np
from mla.base import Base

def test_fit():
    b = Base()

    b.fit([1], [2])
    b.fit([1])

    # For X
    with pytest.raises(ValueError):
        b.fit([], [1])

    with pytest.raises(ValueError):
        X = np.ndarray(shape=(2, 2, 2), dtype=float, order='F')
        b.fit(X)

    # For y
    b.y_required = True

    with pytest.raises(ValueError):
        b.fit([[1], [1]], [[1], [1]])

    with pytest.raises(ValueError):
        b.fit([1], [])

    with pytest.raises(ValueError):
        b.fit([1], )
