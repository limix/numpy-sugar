from numpy import empty_like
from numpy import diag_indices_from
from numpy.linalg import cholesky
from numpy.linalg import LinAlgError
from .. import _epsilon as epsilon


def check_definite_positiveness(A):
    B = empty_like(A)
    B[:] = A
    B[diag_indices_from(B)] += epsilon.small
    try:
        cholesky(B)
    except LinAlgError:
        return False
    return True


def check_symmetry(A):
    return abs(A - A.T).max() < epsilon.small
