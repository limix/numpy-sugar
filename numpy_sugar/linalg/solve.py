from numpy.linalg import solve as npy_solve
from numpy import array
from numpy import dot


def _solve(A, B):
    return npy_solve(A, B)


def solve(A, B):
    if A.shape[0] == 1:
        A_ = array([[1. / A[0, 0]]])
        return dot(A_, B)
    elif A.shape[0] == 2:
        a = A[0, 0]
        b = A[0, 1]
        c = A[1, 0]
        d = A[1, 1]
        A_ = array([[d, -b], [-c, a]])
        A_ /= a * d - b * c
        return dot(A_, B)
    return _solve(A, B)
