from numpy import inf, nan
from numpy.testing import assert_allclose, assert_equal

from numpy_sugar import cartesian, is_all_finite


def test_is_all_finite():
    assert_equal(is_all_finite([1, -1, 2393.]), True)
    assert_equal(is_all_finite([1, -1, nan, 2393.]), False)
    assert_equal(is_all_finite([1, -1, inf, 2393.]), False)


def test_cartesian():
    assert_allclose(
        cartesian((2, 3)), [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]])


if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
