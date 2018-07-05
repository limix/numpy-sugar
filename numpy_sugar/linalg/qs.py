from numpy import asarray, finfo, logical_not, sqrt
from numpy.linalg import eigh, svd


def economic_qs(K, epsilon=sqrt(finfo(float).eps)):
    r"""Economic eigen decomposition for symmetric matrices.

    A symmetric matrix ``K`` can be decomposed in
    :math:`\mathrm Q_0 \mathrm S_0 \mathrm Q_0^\intercal + \mathrm Q_1\
    \mathrm S_1 \mathrm Q_1^ \intercal`, where :math:`\mathrm S_1` is a zero
    matrix with size determined by ``K``'s rank deficiency.

    Args:
        K (array_like): Symmetric matrix.
        epsilon (float): Eigen value threshold. Default is
                         ``sqrt(finfo(float).eps)``.

    Returns:
        tuple: ``((Q0, Q1), S0)``.
    """

    (S, Q) = eigh(K)

    nok = max(Q[0][0].min(), Q[0][0].max(), key=abs) < epsilon
    nok = nok and max(K.min(), K.max(), key=abs) >= epsilon
    if nok:
        from scipy.linalg import eigh as sp_eigh

        (S, Q) = sp_eigh(K)

    ok = S >= epsilon
    nok = logical_not(ok)
    S0 = S[ok]
    Q0 = Q[:, ok]
    Q1 = Q[:, nok]
    return ((Q0, Q1), S0)


def economic_qs_linear(G):
    r"""Economic eigen decomposition for symmetric matrices ``dot(G, G.T)``.

    It is theoretically equivalent to ``economic_qs(dot(G, G.T))``.
    Refer to :func:`numpy_sugar.economic_qs` for further information.

    Args:
        G (array_like): Matrix.

    Returns:
        tuple: ``((Q0, Q1), S0)``.
    """
    import dask.array as da

    if not isinstance(G, da.Array):
        G = asarray(G, float)

    if G.shape[0] > G.shape[1]:
        (Q, Ssq, _) = svd(G, full_matrices=True)
        S0 = Ssq ** 2
        rank = len(S0)
        Q0, Q1 = Q[:, :rank], Q[:, rank:]
        return ((Q0, Q1), S0)

    return economic_qs(G.dot(G.T))
