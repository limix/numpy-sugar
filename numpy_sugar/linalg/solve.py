from numpy import array, asarray, dot, finfo, sqrt, zeros
from numpy.linalg import solve as npy_solve
from numpy.linalg import lstsq

_epsilon = sqrt(finfo(float).eps)


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
        A_ = array([[1. / A[0, 0]]])
        return dot(A_, b)
    elif A.shape[0] == 2:
        a = A[0, 0]
        b_ = A[0, 1]
        c = A[1, 0]
        d = A[1, 1]
        A_ = array([[d, -b_], [-c, a]])
        A_ /= a * d - b_ * c
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
