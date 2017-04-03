r"""
**************
Linear algebra
**************

Decomposition
^^^^^^^^^^^^^

.. autofunction:: economic_qs
.. autofunction:: economic_qs_linear
.. autofunction:: economic_svd

Determinant
^^^^^^^^^^^

.. autofunction:: plogdet

Dot and sum
^^^^^^^^^^^

.. autofunction:: sum2diag
.. autofunction:: ddot

Properties
^^^^^^^^^^

.. autofunction:: check_definite_positiveness
.. autofunction:: check_symmetry

Reducers
^^^^^^^^

.. autofunction:: lu_slogdet
.. autofunction:: trace2
.. autofunction:: dotd

Solvers
^^^^^^^

.. autofunction:: cho_solve
.. autofunction:: lu_solve
.. autofunction:: stl
.. autofunction:: lstsq

"""

from .cho import cho_solve
from .det import plogdet
from .diag import sum2diag, trace2
from .dot import ddot, dotd
from .lu import lu_slogdet, lu_solve
from .property import check_definite_positiveness, check_symmetry
from .qs import economic_qs, economic_qs_linear
from .solve import solve
from .svd import economic_svd
from .tri import stl
from .lstsq import lstsq

__all__ = ['cho_solve', 'sum2diag', 'trace2', 'ddot', 'dotd', 'lu_slogdet',
           'lu_solve', 'check_definite_positiveness', 'check_symmetry',
           'economic_qs', 'economic_qs_linear', 'solve', 'economic_svd', 'stl',
           'lstsq', 'plogdet']
