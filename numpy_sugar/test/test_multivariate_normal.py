from __future__ import division

from numpy import dot
from numpy.random import RandomState
from numpy.testing import assert_allclose

from numpy_sugar.random import multivariate_normal


def test_multivariate_normal():
    rs = RandomState(3)
    m = rs.randn(10)
    G = rs.randn(10, 12)
    K = dot(G, G.T)
    assert_allclose(
        multivariate_normal(m, K, rs), [
            7.67701071054, 1.36892007698, -6.43324011594, -7.34236124724,
            -1.27151910971, 3.19037731625, 0.90263960298, -1.85267387027,
            -0.735702280982, 0.948785481537
        ])
