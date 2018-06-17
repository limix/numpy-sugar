from numpy import inf, nan
from numpy.testing import assert_allclose, assert_equal, assert_
import dask.array as da

from numpy_sugar import cartesian, is_all_finite, is_crescent, is_all_equal
from numpy_sugar import unique


def test_is_crescent():
    a = [1, 2, 3]
    b = [1, 2, 2]
    c = [3, 2, 1]
    d = [3, 3, 1]
    e = [1, 3, 2]

    assert_(is_crescent(a))
    assert_(is_crescent(b))
    assert_(not is_crescent(c))
    assert_(not is_crescent(d))
    assert_(not is_crescent(e))


def test_is_all_equal():
    assert_(is_all_equal([1, 1, 1]))
    assert_(not is_all_equal([1, 1, 2]))


def test_is_all_finite():
    assert_equal(is_all_finite([1, -1, 2393.]), True)
    assert_equal(is_all_finite([1, -1, nan, 2393.]), False)
    assert_equal(is_all_finite([1, -1, inf, 2393.]), False)


def test_cartesian():
    assert_allclose(
        cartesian((2, 3)), [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]
    )


def test_unique():
    a = [1, 2, 2]
    assert_allclose(unique(a), [1, 2])

    a = da.asarray(a)
    assert_allclose(unique(a), [1, 2])
