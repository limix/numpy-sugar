"""
*****
Array
*****

.. automodule:: numpy_sugar.array

  .. autofunction:: is_all_equal
  .. autofunction:: is_all_finite
  .. autofunction:: is_crescent

"""

from numpy import sum as _sum
from numpy import asarray, isfinite, mgrid, prod, rollaxis

try:
    from numba import jit
    _NUMBA = True
except ImportError:
    _NUMBA = False


def is_crescent(arr):
    """Verify if the array values are in non-decreasing order."""
    arr = asarray(arr)
    return _is_crescent(arr)


def is_all_equal(arr):
    """Verify if the array values are all equal."""
    arr = asarray(arr)
    return _is_all_equal(arr)


def is_all_finite(arr):
    """Verify if the array values are all finite.

    NaN is not finite, as well as Inf."""
    return isfinite(_sum(arr))


def _is_crescent(arr):
    i = 0
    while i < arr.shape[0] - 1:
        if arr[i] > arr[i + 1]:
            return False
        i += 1
    return True


if _NUMBA:
    _is_crescent = jit(_is_crescent, nogil=True, nopython=True)


def _is_all_equal(arr):
    arr = arr.ravel()
    v = arr[0]
    i = 1
    while i < arr.shape[0]:
        if arr[i] != v:
            return False
        i += 1
    return True


if _NUMBA:
    _is_all_equal = jit(_is_all_equal, nogil=True, nopython=True)


def cartesian(shape):
    """Cartesian indexing.

    Returns a sequence of n-tuples indexing each element of a hypothetical
    matrix of the given shape.

    Example:

    .. testcode::

        from numpy_sugar import cartesian
        print(cartesian((2, 3)))

    .. testoutput::

        [[0 0]
         [0 1]
         [0 2]
         [1 0]
         [1 1]
         [1 2]]


    Reference: http://stackoverflow.com/a/27286794
    """
    n = len(shape)
    idx = [slice(0, s) for s in shape]
    g = rollaxis(mgrid[idx], 0, n + 1)
    return g.reshape((prod(shape), n))
