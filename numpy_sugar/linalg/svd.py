from numpy import asarray, finfo, sqrt


def economic_svd(G, epsilon=sqrt(finfo(float).eps)):
    r"""Economic Singular Value Decomposition.

    Args:
        G (array_like): Matrix to be factorized.
        epsilon (float): Threshold on the square root of the eigen values.
                         Default is ``sqrt(finfo(float).eps)``.

    Returns:
        :class:`numpy.ndarray`: Unitary matrix.
        :class:`numpy.ndarray`: Singular values.
        :class:`numpy.ndarray`: Unitary matrix.

    See Also
    --------
    numpy.linalg.svd : Cholesky decomposition.
    scipy.linalg.svd : Cholesky decomposition.
    """
    from scipy.linalg import svd

    G = asarray(G, float)
    (U, S, V) = svd(G, full_matrices=False, check_finite=False)
    ok = S >= epsilon
    S = S[ok]
    U = U[:, ok]
    V = V[ok, :]
    return (U, S, V)
