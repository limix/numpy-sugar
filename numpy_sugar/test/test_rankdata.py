from numpy import array, nan
from numpy.random import RandomState
from numpy.testing import assert_allclose
from numpy_sugar import nanrankdata


def test_nanrankdata():
    Xorig = RandomState(0).randn(3, 2)
    Xorig[0, 0] = nan
    R = array([[nan, 2.0], [1.0, 3.0], [2.0, 1.0]])
    Rt = array([[nan, 1.0], [1.0, 2.0], [2.0, 1.0]])

    X = Xorig.copy()
    assert_allclose(nanrankdata(X), R)
    assert_allclose(X, Xorig)

    X = Xorig.copy()
    assert_allclose(nanrankdata(X, inplace=True), R)
    assert_allclose(X, R)

    X = Xorig.copy()
    assert_allclose(nanrankdata(X, axis=0), Rt)
    assert_allclose(X, Xorig)

    X = Xorig.copy()
    assert_allclose(nanrankdata(X, axis=0, inplace=True), Rt)
    assert_allclose(X, Rt)
