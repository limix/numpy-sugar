"""
*******************
NumPy Sugar package
*******************

Missing NumPy functionalities.

"""
from __future__ import absolute_import as _

from . import epsilon, linalg, ma, random
from ._array import cartesian, is_all_equal, is_all_finite, is_crescent, unique
from .testit import test

__name__ = "numpy-sugar"
__version__ = "1.2.5"
__author__ = "Danilo Horta"
__author_email__ = "horta@ebi.ac.uk"

__all__ = [
    "__name__", "__version__", "__author__", "__author_email__", "test",
    "epsilon", "linalg", "ma", "random", "cartesian", "is_all_equal",
    "is_all_finite", "is_crescent", "unique"
]
