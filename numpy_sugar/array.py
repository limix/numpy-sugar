from numpy import asarray
from numpy import isfinite
from numpy import sum
from numpy import all
from numba import jit


def isint_alike(arr):
    return all(arr == asarray(arr, int))


@jit
def _iscrescent(arr):
    i = 0
    while i < arr.shape[0] - 1:
        if arr[i] > arr[i + 1]:
            return False
        i += 1
    return True


def iscrescent(arr):
    arr = asarray(arr)
    return _iscrescent(arr)


@jit(nogil=True, nopython=True)
def _issingleton(arr):
    arr = arr.ravel()
    v = arr[0]
    i = 1
    while i < arr.shape[0]:
        if arr[i] != v:
            return False
        i += 1
    return True


def issingleton(arr):
    arr = asarray(arr)
    return _issingleton(arr)


def is_all_finite(arr):
    return isfinite(sum(arr))
