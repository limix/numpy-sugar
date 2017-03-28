"""
*****************
Special functions
*****************

Beta
^^^^

.. automodule:: numpy_sugar.special

  .. autofunction:: beta_isf(a, b, x)

Chi-squared
^^^^^^^^^^^

.. automodule:: numpy_sugar.special

  .. autofunction:: chi2_sf(k, x)

Gamma
^^^^^

.. automodule:: numpy_sugar.special

  .. autofunction:: lgamma(x)

Log sum
^^^^^^^

.. automodule:: numpy_sugar.special

  .. autofunction:: logbinom(a, b)
  .. autofunction:: logaddexp(x, y)
  .. autofunction:: logaddexps(x, y, sx, sy)
  .. autofunction:: logaddexpss(x, y, sx, sy, r, sign)
  .. autofunction:: logsumexp(x)

Normal
^^^^^^

.. automodule:: numpy_sugar.special

  .. autofunction:: normal_pdf(x)
  .. autofunction:: normal_cdf(x)
  .. autofunction:: normal_icdf(x)
  .. autofunction:: normal_sf(x)
  .. autofunction:: normal_isf(x)
  .. autofunction:: normal_logpdf(x)
  .. autofunction:: normal_logcdf(x)
  .. autofunction:: normal_logsf(x)

Pearson
^^^^^^^

.. automodule:: numpy_sugar.special

  .. autofunction:: r_squared(x, y)

"""
from __future__ import absolute_import as _

from .special import (chi2_sf, lgamma, normal_pdf, normal_cdf, normal_icdf,
                      normal_sf, normal_isf, normal_logpdf, normal_logcdf,
                      normal_logsf, beta_isf, logbinom, logaddexp, logaddexps,
                      logaddexpss, logsumexp, r_squared)

__all__ = [
    "chi2_sf", "lgamma", "normal_pdf", "normal_cdf", "normal_icdf",
    "normal_sf", "normal_isf", "normal_logpdf", "normal_logcdf",
    "normal_logsf", "beta_isf", "logbinom", "logaddexp", "logaddexps",
    "logaddexpss", "logsumexp", "r_squared"
]
