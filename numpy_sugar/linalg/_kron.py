def kron_dot(A, B, C, out=None):
    r"""Kronecker product followed by dot product.

    Let :math:`\mathrm A`, :math:`\mathrm B`, and :math:`\mathrm C` be matrices of
    dimensions :math:`p\times p`, :math:`n\times d`, and :math:`d\times p`.

    It computes

    .. math::

        \text{unvec}((\mathrm A\otimes\mathrm B)\text{vec}(\mathrm C))
        \in n\times p,

    which is equivalent to :math:`\mathrm B\mathrm C\mathrm A^{\intercal}`.

    Parameters
    ----------
    A : array_like
        Matrix A.
    B : array_like
        Matrix B.
    C : array_like
        Matrix C.
    out : :class:`numpy.ndarray`, optional
        Copy result to. Defaults to ``None``.

    Returns
    -------
    :class:`numpy.ndarray`
        unvec((A ⊗ B) vec(C))
    """
    from numpy import asarray, dot, zeros

    A = asarray(A)
    B = asarray(B)
    C = asarray(C)

    if out is None:
        out = zeros((B.shape[0], A.shape[0]))
    dot(B, dot(C, A.T), out=out)
    return out
