from scipy.linalg import lu_solve as sp_lu_solve

from numpy import arange
from numpy import prod
from numpy import sign
from numpy import log
from numpy import sum
from numpy import abs


def lu_slogdet(LU):
    """Natural logarithm of a LU decomposition."""
    adet = sum(log(abs(LU[0].diagonal())))

    s = prod(sign(LU[0].diagonal()))

    nrows_exchange = LU[1].size - \
        sum(LU[1] == arange(LU[1].size, dtype='int32'))

    odd = nrows_exchange % 2 == 1
    if odd:
        s *= -1.0

    return (s, adet)


def lu_solve(LU, x):
    """Compute ``numpy.dot(numpy.linalg.inv(LU), x)`` for LU decomposition."""
    return sp_lu_solve(LU, x, check_finite=False)
