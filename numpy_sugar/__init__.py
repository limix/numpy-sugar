"""
*******************
NumPy Sugar package
*******************

Missing NumPy functionalities.

"""
from __future__ import absolute_import

from . import epsilon, linalg, ma, random
from ._array import cartesian, is_all_equal, is_all_finite, is_crescent, unique
from ._testit import test

__version__ = "1.2.8"

__all__ = [
    "__version__",
    "test",
    "epsilon",
    "linalg",
    "ma",
    "random",
    "cartesian",
    "is_all_equal",
    "is_all_finite",
    "is_crescent",
    "unique",
]
