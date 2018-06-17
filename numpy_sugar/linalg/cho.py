from numpy import asarray, empty


def cho_solve(L, b):
    r"""Solve for Cholesky decomposition.

    Solve the linear equations :math:`\mathrm A \mathbf x = \mathbf b`,
    given the Cholesky factorization of :math:`\mathrm A`.

    Args:
        L (array_like): Lower triangular matrix.
        b (array_like): Right-hand side.

    Returns:
        :class:`numpy.ndarray`: The solution to the system
                                :math:`\mathrm A \mathbf x = \mathbf b`.

    See Also
    --------
    numpy.linalg.cholesky : Cholesky decomposition.
    scipy.linalg.cho_solve : Solve linear equations given Cholesky
                             factorization.
    """
    from scipy.linalg import cho_solve as sp_cho_solve

    L = asarray(L, float)
    b = asarray(b, float)
    if L.size == 0:
        if b.size != 0:
            raise ValueError("Dimension mismatch between L and b.")
        return empty(b.shape)
    return sp_cho_solve((L, True), b, check_finite=False)
