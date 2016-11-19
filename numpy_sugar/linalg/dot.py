from numpy import dot
from numpy import empty
from numpy import einsum
from numpy import copy
from numpy import multiply
from numpy import newaxis


def dotd(A, B, out=None):
    """Retrieves the diagonal of ``numpy.dot(A, B.T)``.

    If `A` is :math:`n\times p` and `B` is :math:`p\times n`, it is done in
    :math:`O(pn)`.

    Args:
        A (array_like): Left matrix.
        B (array_like): Right matrix.
        out (array_like): copy result to.
    """
    if A.ndim == 1 and B.ndim == 1:
        if out is None:
            return dot(A, B)
        return dot(A, B, out)
    if out is None:
        out = empty((A.shape[0], ), float)
    return einsum('ij,ji->i', A, B, out=out)


def ddot(L, R, left=True, out=None):
    """Multiply a matrix by a diagonal one.

    Args:
        L (array_like): Left matrix or vector.
        R (array_like): Right matrix or vector.
        left (bool): ``True`` for vector `L`, ``False`` for vector `R`.

    Returns:
        Resulting matrix.
    """
    if left:
        if out is None:
            out = copy(R)
        return multiply(L[:, newaxis], R, out=out)
    else:
        if out is None:
            out = copy(L)
        return multiply(out, R, out=out)
