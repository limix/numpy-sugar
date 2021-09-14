from numpy import asanyarray, diag_indices_from, empty_like, finfo, sqrt
from numpy.linalg import eigvals


def check_definite_positiveness(A):
    """Check if ``A`` is a definite positive matrix.

    Args:
        A (array_like): Matrix.

    Returns:
        bool: ``True`` if ``A`` is definite positive; ``False`` otherwise.
    """
    return all(eigvals(A) > 0)


def check_semidefinite_positiveness(A):
    """Check if ``A`` is a semi-definite positive matrix.

    Args:
        A (array_like): Matrix.

    Returns:
        bool: ``True`` if ``A`` is definite positive; ``False`` otherwise.
    """
    B = empty_like(A)
    B[:] = A
    B[diag_indices_from(B)] += sqrt(finfo(float).eps)
    return check_definite_positiveness(B)


def check_symmetry(A):
    """Check if ``A`` is a symmetric matrix.

    Args:
        A (array_like): Matrix.

    Returns:
        bool: ``True`` if ``A`` is symmetric; ``False`` otherwise.
    """
    A = asanyarray(A)
    if A.ndim != 2:
        raise ValueError("Checks symmetry only for bi-dimensional arrays.")

    if A.shape[0] != A.shape[1]:
        return False

    return abs(A - A.T).max() < sqrt(finfo(float).eps)
