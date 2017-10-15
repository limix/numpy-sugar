import pytest

from numpy import diag, dot, empty, zeros, array
from numpy.linalg import lstsq as npy_lstsq
from numpy.linalg import solve as npy_solve
from numpy.linalg import cholesky, LinAlgError
from numpy.random import RandomState
from numpy.testing import assert_allclose

from numpy_sugar.linalg import (ddot, dotd, economic_qs, lstsq, rsolve, solve,
                                trace2)


def test_economic_qs():
    random = RandomState(633)
    A = random.randn(10, 10)
    Q, S = economic_qs(A)

    e = []
    e.append([-0.30665477, -0.06935249, -0.19790895, -0.31966245, -0.2041274])
    e.append([-0.41417631, 0.70463554, -0.029418, 0.23839354, -0.01000668])
    e.append([-0.15609931, 0.30659134, 0.12898542, 0.21192988, 0.40325725])
    e.append([0.23709357, 0.03994193, -0.12559863, -0.08280338, 0.07192297])
    e.append([0.03497126, -0.2059239, 0.13106679, -0.28509727, 0.42246837])
    e.append([0.01812449, 0.22538233, -0.8011112, -0.24884526, 0.09372236])
    e.append([0.21498028, 0.45112184, 0.49290517, -0.46891478, -0.06614824])
    e.append([-0.26556389, -0.12489177, 0.09633139, 0.183802, -0.6557185])
    e.append([0.2334407, 0.24061674, -0.0327931, -0.42773492, -0.37992895])
    e.append([-0.69358038, -0.1813236, 0.12363693, -0.45758927, 0.15649533])

    assert_allclose(Q[0], e, rtol=1e-5)

    e = []
    e.append([0.2602789, -0.16562995, 0.3652314, 0.69671087, -0.06445665])
    e.append([-0.08773482, -0.01183062, 0.17659833, -0.12014128, -0.46977809])
    e.append([-0.28447028, 0.05696058, 0.0320373, 0.37750777, 0.6555612])
    e.append([-0.52350264, -0.24840377, -0.50625997, 0.4225672, -0.37916396])
    e.append([-0.31660658, 0.56505551, 0.39638672, 0.04048807, -0.31803257])
    e.append([0.1010443, 0.37774975, -0.21705131, -0.10319173, 0.16038291])
    e.append([0.34690429, 0.24425194, -0.2942541, 0.12823675, 0.00535483])
    e.append([-0.28705797, 0.54312199, -0.19616314, 0.13677848, 0.07922182])
    e.append([-0.48731195, -0.23394529, 0.38236667, -0.24101999, 0.25046401])
    e.append([-0.13751969, -0.19006396, -0.32029686, -0.27120733, 0.07565524])

    assert_allclose(Q[1], e, rtol=1e-5)

    e = [0.96267995, 1.51363689, 2.17446661, 2.73659799, 5.83305263]

    assert_allclose(S, e, rtol=1e-5)


def test_trace2():
    random = RandomState(38493)
    A = random.randn(10, 2)
    B = random.randn(2, 10)

    assert_allclose(A.dot(B).trace(), trace2(A, B))


def test_dotd():
    random = RandomState(958)
    A = random.randn(10, 2)
    B = random.randn(2, 10)

    r = A.dot(B).diagonal()
    assert_allclose(r, dotd(A, B))
    r1 = empty(len(r))
    assert_allclose(dotd(A, B, out=r1), r)


def test_ddot():
    random = RandomState(633)
    A = random.randn(10, 10)
    B = random.randn(10)

    AdB = A.dot(diag(B))
    assert_allclose(AdB, ddot(A, B, left=False))
    assert_allclose(AdB, ddot(A, B, left=False, out=A))

    B = random.randn(10, 10)
    A = random.randn(10)

    AdB = diag(A).dot(B)
    assert_allclose(AdB, ddot(A, B, left=True))
    assert_allclose(AdB, ddot(A, B, left=True, out=B))


def test_solve():
    random = RandomState(0)
    A = random.randn(1, 1)
    b = random.randn(1)

    assert_allclose(solve(A, b), npy_solve(A, b))

    A = random.randn(2, 2)
    b = random.randn(2)

    assert_allclose(solve(A, b), npy_solve(A, b))

    A = random.randn(3, 3)
    b = random.randn(3)

    assert_allclose(solve(A, b), npy_solve(A, b))


def test_solve_raise():
    A = array([[2.05036632, 2.05036632], [2.05036632, 2.05036632]])

    b = array([0.11260227, 0.11260227])

    with pytest.raises(LinAlgError):
        solve(A, b)

    A = array([[0.0]])
    b = array([1.0])
    with pytest.raises(LinAlgError):
        solve(A, b)


def test_rsolve():
    random = RandomState(0)
    A = random.randn(1, 1)
    b = random.randn(1)

    assert_allclose(solve(A, b), npy_solve(A, b))

    A = random.randn(2, 2)
    b = random.randn(2)

    assert_allclose(solve(A, b), npy_solve(A, b))

    A = random.randn(3, 3)
    b = random.randn(3)

    assert_allclose(rsolve(A, b), npy_solve(A, b))

    A[:] = 1e-10
    assert_allclose(rsolve(A, b), zeros(A.shape[1]))


def test_cdot():
    random = RandomState(0)
    A = random.randn(3, 3)
    A = dot(A, A.T)
    L = cholesky(A)
    assert_allclose(L, [[2.05668046, 0., 0.], [1.82034632, 2.48007792, 0.],
                        [0.73633938, 0.24469357, 0.57806618]])


def test_lstsq():
    random = RandomState(0)
    A = random.randn(4, 2)
    b = random.randn(4)
    assert_allclose(lstsq(A, b), npy_lstsq(A, b)[0])

    A = random.randn(4, 1)
    b = random.randn(4)
    assert_allclose(lstsq(A, b), npy_lstsq(A, b)[0])
