from numpy import asarray
from numpy import isfinite
from numpy import sum

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
    return isfinite(sum(arr))


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
