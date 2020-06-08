from numpy.ma import asarray, dot, empty, sum


def dotd(A, B):
    r"""Diagonal of :math:`\mathrm A\mathrm B^\intercal`.

    If ``A`` is :math:`n\times p` and ``B`` is :math:`p\times n`, it is done in
    :math:`O(pn)`.

    Args:
        A (array_like): Left matrix.
        B (array_like): Right matrix.

    Returns:
        :class:`numpy.ndarray`: Resulting diagonal.
    """
    A = asarray(A, float)
    B = asarray(B, float)
    if A.ndim == 1 and B.ndim == 1:
        return dot(A, B)

    out = empty((A.shape[0],), float)
    out[:] = sum(A * B.T, axis=1)

    return out
