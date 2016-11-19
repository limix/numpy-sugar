from numpy.linalg import svd
from .. import _epsilon as epsilon


def economic_svd(G, epsilon=epsilon.small):
    (U, S, V) = svd(G, full_matrices=False)
    ok = S >= epsilon
    S = S[ok]
    U = U[:, ok]
    V = V[:, ok]
    return (U, S, V)
