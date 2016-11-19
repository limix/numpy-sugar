from scipy.linalg import cho_solve as sp_cho_solve


def cho_solve(L, b):
    """Solver for Cholesky decomposition.

    Solve the linear equations A b = b, given the Cholesky factorization of A.

    Args:
        L (array_like): Lower triangular matrix.
        b (array_like): Lower triangular matrix.

    Returns:
        array_like: The solution to the system A x = b.
    """
    return sp_cho_solve((L, True), b, check_finite=False)
