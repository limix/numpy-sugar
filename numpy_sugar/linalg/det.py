from numpy.linalg import eigvalsh
from numpy import log, sqrt, finfo
from numpy import sum as npsum

epsilon = sqrt(finfo(float).eps)

def plogdet(K):
    r"""Log of the pseudo-determinant.

    It assumes that ``K`` is a positive semi-definite matrix.

    Args:
        K (array_like): matrix.

    Returns:
        float: log of the pseudo-determinant.
    """
    egvals = eigvalsh(K)
    return npsum(log(egvals[egvals > epsilon]))
