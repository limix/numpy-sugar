from numpy import abs as _abs
from numpy import sum as _sum
from numpy import arange, asarray, log, prod, sign


def lu_slogdet(LU):
    r"""Natural logarithm of a LU decomposition.

    Args:
        LU (tuple): LU decomposition.

    Returns:
        tuple: sign and log-determinant.
    """
    LU = (asarray(LU[0], float), asarray(LU[1], float))
    adet = _sum(log(_abs(LU[0].diagonal())))

    s = prod(sign(LU[0].diagonal()))

    nrows_exchange = LU[1].size - _sum(LU[1] == arange(LU[1].size, dtype="int32"))

    odd = nrows_exchange % 2 == 1
    if odd:
        s *= -1.0

    return (s, adet)


def lu_solve(LU, b):
    r"""Solve for LU decomposition.

    Solve the linear equations :math:`\mathrm A \mathbf x = \mathbf b`,
    given the LU factorization of :math:`\mathrm A`.

    Args:
        LU (array_like): LU decomposition.
        b (array_like): Right-hand side.

    Returns:
        :class:`numpy.ndarray`: The solution to the system
        :math:`\mathrm A \mathbf x = \mathbf b`.

    See Also
    --------
    scipy.linalg.lu_factor : LU decomposition.
    scipy.linalg.lu_solve : Solve linear equations given LU factorization.
    """
    from scipy.linalg import lu_solve as sp_lu_solve

    LU = (asarray(LU[0], float), asarray(LU[1], float))
    b = asarray(b, float)
    return sp_lu_solve(LU, b, check_finite=False)
