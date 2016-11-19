from scipy.linalg import cho_solve as sp_cho_solve


def cho_solve(L, x):
    """Solver for Cholesky decomposition.

    Args:
        L (array_like): Lower triangular matrix.

    Returns:
        ``solve(dot(L, L.T), x)`` in practice.
    """
    return sp_cho_solve((L, True), x, check_finite=False)
