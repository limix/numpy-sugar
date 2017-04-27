from numpy import exp
from numpy import log
from numpy import ones
from numpy import array
from numpy import empty
from numpy import sign as get_sign
from numpy.random import RandomState
from numpy.testing import assert_allclose

def test_chi2_sf():
    from numpy_sugar.special import chi2_sf
    assert_allclose(chi2_sf(1, 1), 0.317310507863)
    assert_allclose(chi2_sf(1, [1, 3]), [0.317310507863, 0.0832645166636])


def test_norm_logpdf():
    from numpy_sugar.special import normal_logpdf

    assert_allclose(normal_logpdf(1.3), -1.7639385332)

    random = RandomState(426)

    x = random.randn(10)

    r = array([
        -0.94668757, -1.4313161, -2.89441909, -2.38091302, -1.62194931,
        -1.52167417, -1.37205765, -1.19692429, -0.96231757, -0.9599998
    ])

    assert_allclose(normal_logpdf(x), r)

def test_norm_logcdf():
    from numpy_sugar.special import normal_logcdf

    assert_allclose(normal_logcdf(1.3), -0.101811802668)

    random = RandomState(426)
    x = random.randn(10)

    r = array([
        -0.89923898, -0.16924368, -3.7540666, -0.04461775, -0.12540335,
        -0.14631613, -1.76868427, -1.4786554, -0.95667042, -0.48975041
    ])

    assert_allclose(normal_logcdf(x), r)


def test_beta_isf():
    from numpy_sugar.special import beta_isf

    random = RandomState(426)
    x = random.rand(10)

    assert_allclose(beta_isf(0.5, 2.0, x[0]), 0.007348224049952436)

    r = array([
        0.00734822, 0.17205478, 0.09223004, 0.15997467, 0.04793589, 0.04370639,
        0.27443191, 0.33717068, 0.01231064, 0.43142393
    ])

    assert_allclose(beta_isf(0.5, 2.0, x), r, rtol=1e-6)


def test_r_squared():
    from numpy_sugar.special import r_squared

    random = RandomState(12345678)
    x = random.rand(10)
    y = random.rand(10)
    ideal = 0.080402268539028335

    assert_allclose(ideal, r_squared(x, y))


def test_logaddexpss():
    from numpy_sugar.special import logaddexpss
    random = RandomState(26)
    x = 10 * (random.rand(10) - 0.5)
    y = 10 * (random.rand(10) - 0.5)

    sx = random.rand(10)
    sx[sx < 0.5] = -1
    sx[sx >= 0.5] = +1

    sy = random.rand(10)
    sy[sy < 0.5] = -1
    sy[sy >= 0.5] = +1

    r = empty(10)
    sign = empty(10)
    logaddexpss(x, y, sx, sy, r, sign)

    v = sx * exp(x) + sy * exp(y)

    assert_allclose(r, log(abs(v)))
    assert_allclose(sign * get_sign(v), ones(10))


if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
