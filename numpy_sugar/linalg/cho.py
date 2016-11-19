from scipy.linalg import cho_solve as sp_cho_solve


def cho_solve(L, b):
    """Solver for Cholesky decomposition.

    Solve the linear equations A b = b, given the Cholesky factorization of A.

    Args:
        L (array_like): Lower triangular matrix.
        b (array_like): Right-hand side.

    Returns:
        :class:`numpy.ndarray`: The solution to the system A x = b.

    See Also
    --------
    numpy.linalg.cholesky : Cholesky decomposition.
    scipy.linalg.cho_solve : Solve linear equations given Cholesky.
    """
    return sp_cho_solve((L, True), b, check_finite=False)
