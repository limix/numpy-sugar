from scipy.linalg import solve_triangular


def stl(A, b):
    r"""Shortcut to ``solve_triangular(A, b, lower=True, check_finite=False)``.

    Solve linear systems :math:`\mathrm A \mathbf x = \mathbf b` when
    :math:`\mathrm A` is a lower-triangular matrix.

    Args:
        A (array_like): A lower-triangular matrix.
        b (array_like): Ordinate values.

    Returns:
        :class:`numpy.ndarray`: Solution ``x``.

    See Also
    --------
    scipy.linalg.solve_triangular
    """
    return solve_triangular(A, b, lower=True, check_finite=False)
