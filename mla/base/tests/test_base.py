# coding: utf-8
# Author: rushter <me@rushter.com>

import pytest
from mla.base import Base, InputError


def test_fit():
    b = Base()

    b.fit([1], [2])
    b.fit([1])

    # For X
    with pytest.raises(InputError):
        b.fit([], [1])
    # For y
    b.y_required = True

    with pytest.raises(InputError):
        b.fit([[1], [1]], [[1], [1]])

    with pytest.raises(InputError):
        b.fit([1], [])

    with pytest.raises(InputError):
        b.fit([1],)
