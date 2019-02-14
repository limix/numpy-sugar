from numpy import integer, issubdtype, asarray, isnan, apply_along_axis
from scipy.stats import rankdata


def nanrankdata(a, axis=-1, inplace=False):
    """ Rank data for arrays contaning NaN values.

    Parameters
    ----------
    X : array_like
        Array of values.
    axis : int, optional
        Axis value. Defaults to `1`.
    inplace : bool, optional
        Defaults to `False`.


    Returns
    -------
    array_like
        Ranked array.

    Examples
    --------

    .. doctest::

        >>> from numpy_sugar import nanrankdata
        >>> from numpy import arange
        >>>
        >>> X = arange(15).reshape((5, 3)).astype(float)
        >>> print(nanrankdata(X))
        [[1. 1. 1.]
         [2. 2. 2.]
         [3. 3. 3.]
         [4. 4. 4.]
         [5. 5. 5.]]
    """
    if hasattr(a, "dtype") and issubdtype(a.dtype, integer):
        raise ValueError("Integer type is not supported.")

    if isinstance(a, (tuple, list)):
        if inplace:
            raise ValueError("Can't use `inplace=True` for {}.".format(type(a)))
        a = asarray(a, float)

    orig_shape = a.shape
    if a.ndim == 1:
        a = a.reshape(orig_shape + (1,))

    if not inplace:
        a = a.copy()

    def rank1d(x):
        idx = ~isnan(x)
        x[idx] = rankdata(x[idx])
        return x

    a = a.swapaxes(1, axis)
    a = apply_along_axis(rank1d, 0, a)
    a = a.swapaxes(1, axis)

    return a.reshape(orig_shape)


def _rank(func1d, a, axis):
    import numpy as np

    a = np.array(a, copy=False)
    if axis is None:
        a = a.ravel()
        axis = 0
    if a.size == 0:
        y = a.astype(np.float64, copy=True)
    else:
        y = np.apply_along_axis(func1d, axis, a)
        if a.dtype != np.float64:
            y = y.astype(np.float64)
    return y


def _nanrankdata_1d(a):
    import numpy as np
    from scipy.stats import rankdata

    y = np.empty(a.shape, dtype=np.float64)
    y.fill(np.nan)
    idx = ~np.isnan(a)
    y[idx] = rankdata(a[idx])

    return y
