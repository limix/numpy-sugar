from numpy import finfo as _finfo, sqrt as _sqrt

"""Same as ``numpy.finfo(float).tiny``."""
super_tiny = _finfo(float).tiny
"""Same as ``numpy.finfo(float).eps``."""
tiny = _finfo(float).eps
"""Same as ``numpy.sqrt(numpy.finfo(float).eps)``."""
small = _sqrt(_finfo(float).eps)
"""Same as ``numpy.sqrt(numpy.sqrt(numpy.finfo(float).eps))``."""
large = _sqrt(_sqrt(_finfo(float).eps))
