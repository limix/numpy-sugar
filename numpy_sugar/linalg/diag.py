from numpy import sum
from numpy import copy
from numpy import copyto
from numpy import einsum


def trace2(A, B):
    r"""Trace of :math::`\mathrm A \mathrm B^\intercal`.

    Args:
        A (array_like): Left-hand side.
        B (array_like): Right-hand side.

    Returns:
        float: Trace of :math::`\mathrm A \mathrm B^\intercal`.
    """
    assert len(A.shape) == 2 and len(B.shape) == 2
    assert A.shape[1] == B.shape[0] and A.shape[0] == B.shape[1]
    return sum(A.T * B)


def sum2diag(A, D, out=None):
    """Add to the diagonal of a matrix.

    Args:
        A (array_like): Matrix to have its diagonal elements changed.
        D (array_like or scalar): Add those values to the diagonal of `A`.
        out (array_like): copy result to.

    Returns:
        ``A + numpy.diag(D)``.
    """
    if out is None:
        out = copy(A)
    else:
        copyto(out, A)
    einsum('ii->i', out)[:] += D
    return out
