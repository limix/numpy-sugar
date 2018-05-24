from numpy import abs as npy_abs
from numpy import all as npy_all
from numpy import (array, asarray, dot, errstate, finfo, isfinite, nan_to_num,
                   sqrt, zeros)
from numpy.linalg import LinAlgError, lstsq
from numpy.linalg import solve as npy_solve

from .. import epsilon

_epsilon = sqrt(finfo(float).eps)


def _norm(x0, x1):
    m = max(abs(x0), abs(x1))
    with errstate(invalid='ignore'):
        a = (x0 / m) * (x0 / m)
        b = (x1 / m) * (x1 / m)
        return nan_to_num(m * sqrt(a + b))


def hsolve(A, y):
    r"""Solver for the linear equations of two variables and equations only.

    It uses Householder reductions to solve ``Ax = y`` in a robust manner.

    Parameters
    ----------
    A : array_like
        Coefficient matrix.
    y : array_like
        Ordinate values.

    Returns
    -------
    :class:`numpy.ndarray`  Solution ``x``.
    """

    n = _norm(A[0, 0], A[1, 0])
    u0 = A[0, 0] - n
    u1 = A[1, 0]
    nu = _norm(u0, u1)

    with errstate(invalid='ignore', divide='ignore'):
        v0 = nan_to_num(u0 / nu)
        v1 = nan_to_num(u1 / nu)

    B00 = 1 - 2 * v0 * v0
    B01 = 0 - 2 * v0 * v1
    B11 = 1 - 2 * v1 * v1

    D00 = B00 * A[0, 0] + B01 * A[1, 0]
    D01 = B00 * A[0, 1] + B01 * A[1, 1]
    D11 = B01 * A[0, 1] + B11 * A[1, 1]

    b0 = y[0] - 2 * y[0] * v0 * v0 - 2 * y[1] * v0 * v1
    b1 = y[1] - 2 * y[0] * v1 * v0 - 2 * y[1] * v1 * v1

    n = _norm(D00, D01)
    u0 = D00 - n
    u1 = D01
    nu = _norm(u0, u1)

    with errstate(invalid='ignore', divide='ignore'):
        v0 = nan_to_num(u0 / nu)
        v1 = nan_to_num(u1 / nu)

    E00 = 1 - 2 * v0 * v0
    E01 = 0 - 2 * v0 * v1
    E11 = 1 - 2 * v1 * v1

    F00 = E00 * D00 + E01 * D01
    F01 = E01 * D11
    F11 = E11 * D11

    F11 = (npy_abs(F11) > epsilon.small) * F11

    with errstate(divide='ignore', invalid='ignore'):
        Fi00 = nan_to_num(F00 / F00 / F00)
        Fi11 = nan_to_num(F11 / F11 / F11)
        Fi10 = nan_to_num(-(F01 / F00) * Fi11)

    c0 = Fi00 * b0
    c1 = Fi10 * b0 + Fi11 * b1

    x0 = E00 * c0 + E01 * c1
    x1 = E01 * c0 + E11 * c1

    return array([x0, x1])


def solve(A, b):
    r"""Solve for the linear equations :math:`\mathrm A \mathbf x = \mathbf b`.

    Args:
        A (array_like): Coefficient matrix.
        b (array_like): Ordinate values.

    Returns:
        :class:`numpy.ndarray`: Solution ``x``.
    """
    A = asarray(A, float)
    b = asarray(b, float)
    if A.shape[0] == 1:

        with errstate(divide='ignore'):
            A_ = array([[1. / A[0, 0]]])

        if not isfinite(A_[0, 0]):
            raise LinAlgError("Division error.")

        return dot(A_, b)
    elif A.shape[0] == 2:
        a = A[0, 0]
        b_ = A[0, 1]
        c = A[1, 0]
        d = A[1, 1]
        A_ = array([[d, -b_], [-c, a]])

        with errstate(divide='ignore'):
            A_ /= a * d - b_ * c

        if not npy_all(isfinite(A_)):
            raise LinAlgError("Division error.")

        return dot(A_, b)
    return _solve(A, b)


def rsolve(A, b, epsilon=_epsilon):
    r"""Robust solve for the linear equations.

    Args:
        A (array_like): Coefficient matrix.
        b (array_like): Ordinate values.

    Returns:
        :class:`numpy.ndarray`: Solution ``x``.
    """
    x = lstsq(A, b, rcond=epsilon)
    r = sum(x[3] > epsilon)
    if r == 0:
        return zeros(A.shape[1])
    return x[0]


def _solve(A, b):
    return npy_solve(A, b)
