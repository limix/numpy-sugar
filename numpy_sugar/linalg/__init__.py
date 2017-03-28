"""
**************
Linear algebra
**************

Decomposition
^^^^^^^^^^^^^

.. automodule:: numpy_sugar.linalg

  .. autofunction:: economic_qs
  .. autofunction:: economic_qs_linear
  .. autofunction:: economic_svd

Dot and sum
^^^^^^^^^^^

.. automodule:: numpy_sugar.linalg

  .. autofunction:: sum2diag
  .. autofunction:: ddot

Properties
^^^^^^^^^^

.. automodule:: numpy_sugar.linalg

  .. autofunction:: check_definite_positiveness
  .. autofunction:: check_symmetry

Reducers
^^^^^^^^

.. automodule:: numpy_sugar.linalg

  .. autofunction:: lu_slogdet
  .. autofunction:: trace2
  .. autofunction:: dotd

Solvers
^^^^^^^

.. automodule:: numpy_sugar.linalg

  .. autofunction:: cho_solve
  .. autofunction:: lu_solve
  .. autofunction:: stl

"""

from .cho import cho_solve
from .diag import sum2diag, trace2
from .dot import ddot, dotd
from .lu import lu_slogdet, lu_solve
from .property import check_definite_positiveness, check_symmetry
from .qs import economic_qs, economic_qs_linear
from .solve import solve
from .svd import economic_svd
from .tri import stl
