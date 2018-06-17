r"""
*****
Array
*****

.. autofunction:: is_all_equal
.. autofunction:: is_all_finite
.. autofunction:: is_crescent

"""

from numpy import sum as _sum
from numpy import unique as _unique
from numpy import asarray, isfinite, mgrid, prod, rollaxis

try:
    from numba import jit

    _NUMBA = True
except ImportError:
    _NUMBA = False


def is_crescent(arr):
    r"""Check if the array values are in non-decreasing order.

    Args:
        arr (array_like): sequence of values.

    Returns:
        bool: ``True`` for non-decreasing order.
    """
    arr = asarray(arr)
    return _is_crescent(arr)


def is_all_equal(arr):
    r"""Check if the array values are all equal.

    Args:
        arr (array_like): sequence of values.

    Returns:
        bool: ``True`` if values are all equal.
    """
    arr = asarray(arr)
    return _is_all_equal(arr)


def is_all_finite(arr):
    r"""Check if the array values are all finite.

    Args:
        arr (array_like): sequence of values.

    Returns:
        bool: ``True`` if values are all finite.
    """
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
    r"""Cartesian indexing.

    Returns a sequence of n-tuples indexing each element of a hypothetical
    matrix of the given shape.

    Args:
        shape (tuple): tuple of dimensions.

    Returns:
        array_like: indices.

    Example
    -------

    .. doctest::

        >>> from numpy_sugar import cartesian
        >>> print(cartesian((2, 3)))
        [[0 0]
         [0 1]
         [0 2]
         [1 0]
         [1 1]
         [1 2]]

    Reference:

    [1] http://stackoverflow.com/a/27286794
    """
    n = len(shape)
    idx = [slice(0, s) for s in shape]
    g = rollaxis(mgrid[idx], 0, n + 1)
    return g.reshape((prod(shape), n))


def unique(ar):
    r"""Find the unique elements of an array.

    It uses ``dask.array.unique`` if necessary.

    Args:
        ar (array_like): Input array.

    Returns:
        array_like: the sorted unique elements.
    """

    import dask.array as da

    if isinstance(ar, da.core.Array):
        return da.unique(ar)

    return _unique(ar)
