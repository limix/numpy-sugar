from numpy import (nan, inf)
from numpy.testing import assert_equal

from limix_math import is_all_finite


def test_is_all_finite():
    assert_equal(is_all_finite([1, -1, 2393.]), True)
    assert_equal(is_all_finite([1, -1, nan, 2393.]), False)
    assert_equal(is_all_finite([1, -1, inf, 2393.]), False)

if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
