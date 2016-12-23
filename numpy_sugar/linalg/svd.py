from numpy import asarray
from numpy.linalg import svd
from .. import _epsilon as epsilon


def economic_svd(G, epsilon=epsilon.small):
    r"""Economic Singular Value Decomposition.

    Args:
        G (array_like): Matrix to be factorized.
        epsilon (float): Threshold on the square root of the eigen values.
                         Default is :obj:`numpy_sugar.epsilon.small`.

    Returns:
        :class:`numpy.ndarray`: Unitary matrix.
        :class:`numpy.ndarray`: Singular values.
        :class:`numpy.ndarray`: Unitary matrix.

    See Also
    --------
    numpy.linalg.svd : Cholesky decomposition.
    """
    G = asarray(G, float)
    (U, S, V) = svd(G, full_matrices=False)
    ok = S >= epsilon
    S = S[ok]
    U = U[:, ok]
    V = V[:, ok]
    return (U, S, V)
