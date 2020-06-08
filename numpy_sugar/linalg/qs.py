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

    nok = abs(max(Q[0].min(), Q[0].max(), key=abs)) < epsilon
    nok = nok and abs(max(K.min(), K.max(), key=abs)) >= epsilon
    if nok:
        from scipy.linalg import eigh as sp_eigh

        (S, Q) = sp_eigh(K)

    ok = S >= epsilon
    nok = logical_not(ok)
    S0 = S[ok]
    Q0 = Q[:, ok]
    Q1 = Q[:, nok]
    return ((Q0, Q1), S0)


def economic_qs_linear(G, return_q1=True):
    """
    Economic eigen decomposition for a symmetric matrix 𝙺=𝙶𝙶ᵀ.

    Let us define ::

        𝙺 = [𝚀₀  𝚀₁] [𝚂₀  𝟎] [𝚀₀ᵀ]
                     [ 𝟎  𝟎] [𝚀₁ᵀ]

    where the eigenvectors are the columns of [𝚀₀  𝚀₁] and the positive
    eigenvalues are the diagonal elements of 𝚂₀.

    Args:
        G (array_like): Matrix.
        return_q1 (bool): Return 𝚀₁ matrix. Defaults to ``True``.

    Returns:
        tuple: ((𝚀₀, 𝚀₁), 𝚂₀).
    """
    import dask.array as da

    if not isinstance(G, da.Array):
        G = asarray(G, float)

    if not return_q1:
        return _economic_qs_linear_noq1(G)

    if G.shape[0] > G.shape[1]:
        (Q, Ssq, _) = svd(G, full_matrices=True)
        S0 = Ssq ** 2
        rank = len(S0)
        Q0, Q1 = Q[:, :rank], Q[:, rank:]
        return ((Q0, Q1), S0)

    return economic_qs(G.dot(G.T))


def _economic_qs_linear_noq1(G):
    if G.shape[0] > G.shape[1]:
        (Q0, Ssq, _) = svd(G, full_matrices=False)
        S0 = Ssq ** 2
        return ((Q0,), S0)

    QS = economic_qs(G.dot(G.T))
    return ((QS[0][0],), QS[1])
