import numpy as np

from . import _special_ffi

from ._numba import (float64, int64)
from ._numba import vectorize
from ._numba import jit
from ._numba import HAS_NUMBA

# try:
#     from numba import cffi_support as _cffi_support
#     from numba import (float64, int64)
#     from numba import vectorize
#     from numba import jit
#     _cffi_support.register_module(_special_ffi)
#     _NUMBA = True
# except ImportError:
#     def vectorize(*args, **kwargs):
#         return decorator(func):
#             return func
#
#     def jit(*args, **kwargs):
#         return decorator(func):
#             return func
#     _NUMBA = False

_chi2_sf = _special_ffi.lib.nsugar_chi2_sf
_lgamma = _special_ffi.lib.nsugar_lgamma
_normal_pdf = _special_ffi.lib.nsugar_normal_pdf
_normal_cdf = _special_ffi.lib.nsugar_normal_cdf
_normal_icdf = _special_ffi.lib.nsugar_normal_icdf
_normal_sf = _special_ffi.lib.nsugar_normal_sf
_normal_isf = _special_ffi.lib.nsugar_normal_isf
_normal_logpdf = _special_ffi.lib.nsugar_normal_logpdf
_normal_logcdf = _special_ffi.lib.nsugar_normal_logcdf
_normal_logsf = _special_ffi.lib.nsugar_normal_logsf
_beta_isf = _special_ffi.lib.nsugar_beta_isf
_logaddexp = _special_ffi.lib.nsugar_logaddexp
_logaddexps = _special_ffi.lib.nsugar_logaddexps
_logaddexpss = _special_ffi.lib.nsugar_logaddexpss
_logbinom = _special_ffi.lib.nsugar_logbinom


def _if_numba(deco):
    if HAS_NUMBA:

        def decorator(func):
            return deco(func)
    else:

        def decorator(func):
            return func

    return decorator


@_if_numba(vectorize([float64(int64, float64)], nopython=True))
def chi2_sf(k, x):
    r"""Chi-squared distribution [1] survival function.

    Args:
        k (array_like): Degrees of fredom.
        x (array_like): Evaluation point.

    Returns:
        :class:`numpy.ndarray`: Survival function of the :math:`\chi_k^2`
                                distribution.

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Chi-squared_distribution
    """
    return _chi2_sf(k, x)


@_if_numba(vectorize([float64(float64)], nopython=True))
def lgamma(x):
    r"""Natural logarithm of the Gamma function [1].

    Args:
        x (array_like): evaluation point.

    Returns:
        :class:`numpy.ndarray`: Log-gamma of ``x``.

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Gamma_function
    """
    return _lgamma(x)


@_if_numba(vectorize([float64(float64)], nopython=True))
def normal_pdf(x):
    """P.d.f. of the Normal distribution.

    Args:
        x (array_like): evaluation point.
    """
    return _normal_pdf(x)


@_if_numba(vectorize([float64(float64)], nopython=True))
def normal_cdf(x):
    """C.d.f. of the Normal distribution.

    Args:
        x (array_like): evaluation point.
    """
    return _normal_cdf(x)


@_if_numba(vectorize([float64(float64)], nopython=True))
def normal_icdf(x):
    """Inverse of the c.d.f. of the Normal distribution.

    Args:
        x (array_like): evaluation point.
    """
    return _normal_icdf(x)


@_if_numba(vectorize([float64(float64)], nopython=True))
def normal_sf(x):
    """Survival function of the Normal distribution.

    Args:
        x (array_like): evaluation point.
    """
    return _normal_sf(x)


@_if_numba(vectorize([float64(float64)], nopython=True))
def normal_isf(x):
    """Inverse of the survival function of the Normal distribution.

    Args:
        x (array_like): evaluation point.
    """
    return _normal_isf(x)


@_if_numba(vectorize([float64(float64)], nopython=True))
def normal_logpdf(x):
    """Natural logarithm of the p.d.f. of the Normal distribution.

    Args:
        x (array_like): evaluation point.
    """
    return _normal_logpdf(x)


@_if_numba(vectorize([float64(float64)], nopython=True))
def normal_logcdf(x):
    """Natural logarithm of the c.d.f. of the Normal distribution.

    Args:
        x (array_like): evaluation point.
    """
    return _normal_logcdf(x)


@_if_numba(vectorize([float64(float64)], nopython=True))
def normal_logsf(x):
    """Natural logarithm of the survival function of the Normal distribution.

    Args:
        x (array_like): evaluation point.
    """
    return _normal_logsf(x)


@_if_numba(
    jit([
        float64(float64, float64, float64),
        float64[:](float64, float64, float64[:])
    ]))
def beta_isf(a, b, x):
    """Inverse of the Beta survival function.

    Args:
        a (array_like): parameter `a`.
        b (array_like): parameter `b`.
        x (array_like): evaluation point.
    """
    if np.isscalar(x):
        return _beta_isf(a, b, x)
    r = np.empty(x.size)
    for i in range(x.size):
        r[i] = _beta_isf(a, b, x[i])
    return r


@_if_numba(vectorize([float64(float64, float64)], nopython=True))
def logbinom(a, b):
    r"""Natural logarithm of Binomial coefficient: :math:`\log{a \choose b}`.
    """
    return _logbinom(a, b)


@_if_numba(vectorize([float64(float64, float64)], nopython=True))
def logaddexp(x, y):
    """Numerically-stable ``numpy.log(numpy.exp(x)+numpy.exp(y))``.

    Args:
        x (array_like): First array.
        y (array_like): Second array.

    Returns:
        ``numpy.log(numpy.exp(x)+numpy.exp(y))``.
    """
    return _logaddexp(x, y)


@_if_numba(
    vectorize([float64(float64, float64, float64, float64)], nopython=True))
def logaddexps(x, y, sx, sy):
    """Numerically-stable ``numpy.log(sx*numpy.exp(x)+sy*numpy.exp(y))``."""
    return _logaddexps(x, y, sx, sy)


def logaddexpss(x, y, sx, sy, r, sign):
    """Numerically-stable ``numpy.log(sx*numpy.exp(x)+sy*numpy.exp(y))``.

    Suppose you are interested in computing::

        sx[i]*exp(x[i]) + sy[i]*exp(y[i])

    where ``sx[i]`` and ``sy[i]`` are either -1 or +1. Often a direct
    calculation of the above is numerically innacurate. Instead, let::

        sign[i]*exp(r[i]) = sx[i]*exp(x[i]) + sy[i]*exp(y[i])

    This function accurately computes ``r[i]`` and ``sign[i]``, where
    ``sign[i]`` is either -1 or +1.
    """
    ptr = _special_ffi.ffi.cast("double*", sign.ctypes.data)
    for i in range(len(x)):
        r[i] = _logaddexpss(x[i], y[i], sx[i], sy[i], ptr + i)


@_if_numba(jit((float64[:], ), nogil=True, nopython=True))
def logsumexp(x):
    """Numerically-stable ``numpy.log(sum(numpy.exp(x)))``.

    Args:
        x (array_like): We want the sum of their exponentiated values but in a
                        numerically-stable way.

    Returns:
        ``numpy.log(sum(numpy.exp(x)))``.
    """
    c = x[0]
    for i in range(1, x.shape[0]):
        c = np.logaddexp(c, x[i])
    return c


@_if_numba(
    jit(float64(float64[:], float64[:]), nopython=True, nogil=True))
def r_squared(x, y):
    """ Coefficient of determination between ``x`` and ``y``.

    It equals to ``scipy.stats.pearsonr(x, y)[0]**2``.

    Args:
        x (array_like): First array.
        y (array_like): Second array.

    Returns:
        Squared Pearson correlation.
    """
    n = x.shape[0]

    x -= x.mean()
    y -= y.mean()

    ssxm = np.dot(x, x) / (n - 1.)
    ssxym = np.dot(x, y) / (n - 1.)
    ssym = np.dot(y, y) / (n - 1.)

    r_num = ssxym
    r_den = np.sqrt(ssxm * ssym)

    if r_den == 0.0:
        r = 0.0
    else:
        r = r_num / r_den
        # test for numerical error propagation
        if r > 1.0:
            r = 1.0
        elif r < -1.0:
            r = -1.0
    return r * r
