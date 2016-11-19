.. toctree::
  :caption: Table of Contents
  :name: mastertoc
  :maxdepth: 3

  index

===========================
Numpy-sugar's documentation
===========================

You can get the source and open issues on `Github.`_

.. _Github.: https://github.com/Horta/numpy-sugar

*******
Install
*******

The recommended way of installing it is via `conda`_::

  conda install -c conda-forge numpy-sugar

An alternative way would be via pip::

  pip install numpy-sugar

.. _conda: http://conda.pydata.org/docs/index.html

*****
Array
*****

.. automodule:: numpy_sugar.array

  .. autofunction:: is_all_equal
  .. autofunction:: is_all_finite
  .. autofunction:: is_crescent

**************
Linear algebra
**************

--------------
Decompositions
--------------

.. automodule:: numpy_sugar.linalg

  .. autofunction:: economic_qs
  .. autofunction:: economic_qs_linear
  .. autofunction:: economic_svd

-----------
Dot and sum
-----------

.. automodule:: numpy_sugar.linalg

  .. autofunction:: sum2diag
  .. autofunction:: ddot

----------
Properties
----------

.. automodule:: numpy_sugar.linalg

  .. autofunction:: check_definite_positiveness
  .. autofunction:: check_symmetry

--------
Reducers
--------

.. automodule:: numpy_sugar.linalg

  .. autofunction:: lu_slogdet
  .. autofunction:: trace2
  .. autofunction:: dotd

-------
Solvers
-------

.. automodule:: numpy_sugar.linalg

  .. autofunction:: cho_solve
  .. autofunction:: lu_solve
  .. autofunction:: stl


*****************
Special functions
*****************

----
Beta
----

.. automodule:: numpy_sugar.special

  .. autofunction:: beta_isf(a, b, x)

-----------
Chi-squared
-----------

.. automodule:: numpy_sugar.special

  .. autofunction:: chi2_sf(k, x)

-----
Gamma
-----

.. automodule:: numpy_sugar.special

  .. autofunction:: lgamma(x)

-------
Log sum
-------

.. automodule:: numpy_sugar.special

  .. autofunction:: logbinom(a, b)
  .. autofunction:: logaddexp(x, y)
  .. autofunction:: logaddexps(x, y, sx, sy)
  .. autofunction:: logaddexpss(x, y, sx, sy, r, sign)
  .. autofunction:: logsumexp(x)

------
Normal
------

.. automodule:: numpy_sugar.special

  .. autofunction:: normal_pdf(x)
  .. autofunction:: normal_cdf(x)
  .. autofunction:: normal_icdf(x)
  .. autofunction:: normal_sf(x)
  .. autofunction:: normal_isf(x)
  .. autofunction:: normal_logpdf(x)
  .. autofunction:: normal_logcdf(x)
  .. autofunction:: normal_logsf(x)

-------
Pearson
-------

.. automodule:: numpy_sugar.special

  .. autofunction:: r_squared(x, y)
