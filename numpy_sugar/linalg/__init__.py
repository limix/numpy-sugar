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
.. autofunction:: cdot

Properties
^^^^^^^^^^

.. autofunction:: check_semidefinite_positiveness
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
.. autofunction:: solve
.. autofunction:: hsolve
.. autofunction:: rsolve
.. autofunction:: stl
.. autofunction:: lstsq

"""

from .cho import cho_solve
from .det import plogdet
from .diag import sum2diag, trace2
from .dot import ddot, dotd, cdot
from .lstsq import lstsq
from .lu import lu_slogdet, lu_solve
from .property import check_semidefinite_positiveness
from .property import check_definite_positiveness, check_symmetry
from .qs import economic_qs, economic_qs_linear
from .solve import rsolve, solve, hsolve
from .svd import economic_svd
from .tri import stl

__all__ = [
    'cho_solve', 'sum2diag', 'trace2', 'ddot', 'dotd', 'lu_slogdet',
    'lu_solve', 'check_definite_positiveness', 'check_symmetry', 'economic_qs',
    'economic_qs_linear', 'solve', 'rsolve', 'economic_svd', 'stl', 'lstsq',
    'plogdet', 'check_semidefinite_positiveness', 'cdot', 'hsolve'
]
