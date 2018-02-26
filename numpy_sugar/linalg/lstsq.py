from numpy import asarray, dot, newaxis, squeeze
from numpy.core import double, finfo
from numpy.linalg import lstsq as npy_lstsq


def lstsq(A, b):
    r"""Return the least-squares solution to a linear matrix equation.

    Args:
        A (array_like): Coefficient matrix.
        b (array_like): Ordinate values.

    Returns:
        :class:`numpy.ndarray`: Least-squares solution.
    """
    A = asarray(A, float)
    b = asarray(b, float)

    if A.ndim == 1:
        A = A[:, newaxis]

    if A.shape[1] == 1:
        return dot(A.T, b) / squeeze(dot(A.T, A))

    rcond = finfo(double).eps * max(*A.shape)
    return npy_lstsq(A, b, rcond=rcond)[0]
