from numpy.ma import dot, masked_array
from numpy.random import RandomState
from numpy.testing import assert_allclose
from numpy_sugar.ma import dotd


def test_dotd_masked():
    random = RandomState(958)
    A = random.randn(3, 2)
    B = random.randn(2, 3)

    A = masked_array(A, mask=[[0, 0], [1, 0], [0, 0]])
    B = masked_array(B, mask=[[1, 0], [0, 0], [0, 1]])

    want = [-0.0047906181652413345, -0.30497335055401054, 0.10254886261295763]
    assert_allclose(dot(A, B).diagonal(), want)
    assert_allclose(dotd(A, B), want)

    a = random.randn(2)
    b = random.randn(2)

    a = masked_array(a, mask=[0, 1])
    b = masked_array(b, mask=[0, 1])

    assert_allclose(dotd(a, b), -0.06600230202137543)
