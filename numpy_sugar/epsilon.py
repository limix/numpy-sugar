# pylint: disable=W0105

from numpy import finfo as _finfo
from numpy import sqrt as _sqrt

"""Same as ``numpy.finfo(float).eps``."""
tiny = _finfo(float).eps
"""Same as ``numpy.sqrt(numpy.finfo(float).eps)``."""
small = _sqrt(_finfo(float).eps)
"""Same as ``numpy.sqrt(numpy.sqrt(numpy.finfo(float).eps))``."""
large = _sqrt(_sqrt(_finfo(float).eps))
