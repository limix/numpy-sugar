from numpy import asarray, copy, copyto, einsum
from numpy import sum as _sum


def trace2(A, B):
    r"""Trace of :math:`\mathrm A \mathrm B`.

    Args:
        A (array_like): Left-hand side.
        B (array_like): Right-hand side.

    Returns:
        float: Trace of :math:`\mathrm A \mathrm B`.
    """
    A = asarray(A, float)
    B = asarray(B, float)

    layout_error = "Wrong matrix layout."

    if not (len(A.shape) == 2 and len(B.shape) == 2):
        raise ValueError(layout_error)

    if not (A.shape[1] == B.shape[0] and A.shape[0] == B.shape[1]):
        raise ValueError(layout_error)

    return _sum(A.T * B)


def sum2diag(A, D, out=None):
    r"""Add values ``D`` to the diagonal of matrix ``A``.

    Args:
        A (array_like): Left-hand side.
        D (array_like or float): Values to add.
        out (:class:`numpy.ndarray`, optional): copy result to.

    Returns:
        :class:`numpy.ndarray`: Resulting matrix.
    """
    A = asarray(A, float)
    D = asarray(D, float)
    if out is None:
        out = copy(A)
    else:
        copyto(out, A)
    einsum("ii->i", out)[:] += D
    return out
