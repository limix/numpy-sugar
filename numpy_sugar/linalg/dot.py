from numpy import asarray, copy, dot, einsum, empty, multiply, newaxis


def dotd(A, B, out=None):
    r"""Diagonal of :math:`\mathrm A\mathrm B^\intercal`.

    If ``A`` is :math:`n\times p` and ``B`` is :math:`p\times n`, it is done in
    :math:`O(pn)`.

    Args:
        A (array_like): Left matrix.
        B (array_like): Right matrix.
        out (:class:`numpy.ndarray`, optional): copy result to.

    Returns:
        :class:`numpy.ndarray`: Resulting diagonal.
    """
    A = asarray(A, float)
    B = asarray(B, float)
    if A.ndim == 1 and B.ndim == 1:
        if out is None:
            return dot(A, B)
        return dot(A, B, out)
    if out is None:
        out = empty((A.shape[0], ), float)
    return einsum('ij,ji->i', A, B, out=out)


def ddot(L, R, left=True, out=None):
    r"""Dot product of a matrix and a diagonal one.

    Args:
        L (array_like): Left matrix.
        R (array_like): Right matrix.
        left (bool): ``True`` if ``L`` is the diagonal matrix;
                     ``False`` otherwise.
        out (:class:`numpy.ndarray`, optional): copy result to.

    Returns:
        :class:`numpy.ndarray`: Resulting matrix.
    """
    L = asarray(L, float)
    R = asarray(R, float)
    if left:
        if out is None:
            out = copy(R)
        return multiply(L[:, newaxis], R, out=out)
    else:
        if out is None:
            out = copy(L)
        return multiply(out, R, out=out)


def cdot(L, out=None):
    r"""Product of a Cholesky matrix with itself transposed.

    Args:
        L (array_like): Cholesky matrix.
        out (:class:`numpy.ndarray`, optional): copy result to.

    Returns:
        :class:`numpy.ndarray`: :math:`\mathrm L\mathrm L^\intercal`.
    """
    L = asarray(L, float)
    assert L.ndim == 2
    assert L.shape[0] == L.shape[1]
    if out is None:
        out = empty((L.shape[0], L.shape[1]), float)
    return einsum('ij,ji->ij', L, L, out=out)
