import pytest
from numpy import all as npy_all
from numpy import argsort, array, diag, dot, empty, isfinite, kron, ones, sqrt, zeros
from numpy.linalg import LinAlgError, cholesky
from numpy.linalg import lstsq as npy_lstsq
from numpy.linalg import norm, slogdet
from numpy.linalg import solve as npy_solve
from numpy.random import RandomState
from numpy.testing import assert_, assert_allclose, assert_equal
from scipy.linalg import lu_factor

from numpy_sugar.linalg import (
    cdot,
    check_definite_positiveness,
    check_semidefinite_positiveness,
    check_symmetry,
    cho_solve,
    ddot,
    dotd,
    economic_qs,
    economic_qs_linear,
    economic_svd,
    hsolve,
    kron_dot,
    lstsq,
    lu_slogdet,
    lu_solve,
    plogdet,
    rsolve,
    solve,
    stl,
    sum2diag,
    trace2,
)


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


def test_economic_qs_linear():
    random = RandomState(2951)

    G = random.randn(3, 5)
    QS0 = economic_qs_linear(G)
    QS1 = economic_qs(dot(G, G.T))
    QS2 = economic_qs_linear(G, return_q1=False)
    assert_allclose(QS0[0][0], QS1[0][0])
    assert_allclose(QS2[0][0], QS1[0][0])
    assert_equal(len(QS2[0]), 1)
    assert_allclose(QS0[0][1], QS1[0][1])
    assert_allclose(QS0[1], QS1[1])
    assert_allclose(QS2[1], QS1[1])

    G = G.T.copy()
    QS0 = economic_qs_linear(G)
    QS1 = economic_qs(dot(G, G.T))
    idx = argsort(-1 * QS1[1])
    QS1 = ((QS1[0][0][:, idx], QS1[0][1]), QS1[1][idx])
    QS2 = economic_qs_linear(G, return_q1=False)

    assert_allclose(QS0[0][0], QS1[0][0])
    assert_allclose(QS2[0][0], QS1[0][0])
    assert_equal(len(QS2[0]), 1)
    assert_allclose(QS0[1], QS1[1])
    assert_allclose(QS2[1], QS1[1])


def test_trace2():
    random = RandomState(38493)
    A = random.randn(10, 2)
    B = random.randn(2, 10)

    assert_allclose(A.dot(B).trace(), trace2(A, B))

    with pytest.raises(ValueError):
        trace2(A[:, 0], B)

    with pytest.raises(ValueError):
        trace2(A, B.T)


def test_dotd():
    random = RandomState(958)
    A = random.randn(10, 2)
    B = random.randn(2, 10)

    r = A.dot(B).diagonal()
    assert_allclose(r, dotd(A, B))
    r1 = empty(len(r))
    assert_allclose(dotd(A, B, out=r1), r)

    a = random.randn(2)
    b = random.randn(2)
    c = array(0.0)

    assert_allclose(dotd(a, b), -1.05959423672)
    dotd(a, b, out=c)
    assert_allclose(c, -1.05959423672)


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

    with pytest.raises(ValueError):
        ddot(A, A)


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

    A = zeros((0, 0))
    b = zeros((0,))
    assert_(rsolve(A, b).ndim == 1)
    assert_(rsolve(A, b).shape[0] == 0)

    A = zeros((0, 1))
    b = zeros((0,))
    assert_(rsolve(A, b).ndim == 1)
    assert_(rsolve(A, b).shape[0] == 1)


def test_lstsq():
    random = RandomState(0)
    A = random.randn(4, 2)
    b = random.randn(4)
    assert_allclose(lstsq(A, b), npy_lstsq(A, b, rcond=-1)[0])

    A = random.randn(4, 1)
    b = random.randn(4)
    assert_allclose(lstsq(A, b), npy_lstsq(A, b, rcond=-1)[0])


def test_stl():
    random = RandomState(0)
    A = random.randn(2, 2)
    b = random.randn(2)
    assert_allclose(stl(A, b), [1.05867493, -0.89850031])


def test_cho_solve():
    random = RandomState(0)
    L = random.randn(2, 2)
    b = random.randn(2)
    assert_allclose(cho_solve(L, b), [0.82259811, -0.40095633])


def test_sum2diag():
    random = RandomState(0)
    A = random.randn(2, 2)
    b = random.randn(2)

    C = A.copy()
    C[0, 0] = C[0, 0] + b[0]
    C[1, 1] = C[1, 1] + b[1]

    assert_allclose(sum2diag(A, b), C)

    want = array([[2.76405235, 0.40015721], [0.97873798, 3.2408932]])
    assert_allclose(sum2diag(A, 1), want)

    D = empty((2, 2))
    sum2diag(A, b, out=D)
    assert_allclose(C, D)


def test_plogdet():
    K = array([[2.76405235, 0.40015721], [0.97873798, 3.2408932]])
    K = dot(K, K.T)

    assert_allclose(plogdet(K), 4.29568333649)


def test_lu_slogdet():
    K = array([[2.76405235, 0.40015721], [0.97873798, 3.2408932]])
    K = dot(K, K.T)

    LU = lu_factor(K)
    assert_allclose(lu_slogdet(LU), slogdet(K))

    random = RandomState(6)
    K = random.randn(3, 3)
    K = dot(K, K.T)

    LU = lu_factor(K)
    assert_allclose(lu_slogdet(LU), slogdet(K))


def test_lu_solve():
    random = RandomState(6)
    A = random.randn(3, 3)
    A = dot(A, A.T)
    y = random.randn(3)

    LU = lu_factor(A)
    assert_allclose(lu_solve(LU, y), [-0.14503211, 0.43277517, -0.22340499])


def test_economic_svd():
    random = RandomState(6)
    A = random.randn(3, 2)
    A = dot(A, A.T)

    S = [
        [-0.21740668, 0.56064537],
        [0.21405445, -0.77127452],
        [-0.95232086, -0.30135094],
    ]
    V = [7.65340901, 0.84916508]
    D = [[-0.21740668, 0.21405445, -0.95232086], [0.56064537, -0.77127452, -0.30135094]]
    SVD = economic_svd(A)

    assert_allclose(SVD[0], S)
    assert_allclose(SVD[1], V)
    assert_allclose(SVD[2], D)


def test_check_definite_positiveness():
    random = RandomState(6)
    A = random.randn(3, 3)
    A = dot(A, A.T)
    A = sum2diag(A, 1e-4)
    assert_(check_definite_positiveness(A))
    assert_(not check_definite_positiveness(zeros((4, 4))))


def test_check_semidefinite_positiveness():
    random = RandomState(6)
    A = random.randn(3, 2)
    A = dot(A, A.T)
    assert_(check_semidefinite_positiveness(A))
    B = -1e-10 * ones((4, 4))

    assert_(check_semidefinite_positiveness(B))

    B = -1e-1 * ones((4, 4))
    assert_(not check_semidefinite_positiveness(B))


def test_check_symmetry():
    random = RandomState(6)
    A = random.randn(3)
    with pytest.raises(ValueError):
        check_symmetry(A)

    A = random.randn(3, 2)
    assert_(not check_symmetry(A))

    A = random.randn(3, 3)
    assert_(not check_symmetry(A))

    A = dot(A, A.T)
    assert_(check_symmetry(A))


def test_cdot():
    random = RandomState(0)
    A = random.randn(3, 3)
    A = dot(A, A.T)
    L = cholesky(A)
    assert_allclose(cdot(L), A)

    a = random.randn(3)
    with pytest.raises(ValueError):
        cdot(a)

    a = random.randn(3, 2)
    with pytest.raises(ValueError):
        cdot(a)


def test_hsolve():

    y = [-0.3, 2.1]

    A = []
    random = RandomState(0)

    a = random.randn(2, 2)
    A.append(a.dot(a.T))

    a = random.randn(2, 2)
    A.append(a.dot(a.T))

    a = random.randn(2, 2)
    A.append(a.dot(a.T))

    a = random.randn(2, 2)
    A.append(a.dot(a.T))

    a = random.randn(2, 2)
    A.append(a.dot(a.T))

    a = random.randn(2, 2)
    A.append(a.dot(a.T))

    a = random.randn(2, 2)
    A.append(a.dot(a.T))

    A += [
        zeros((2, 2)),
        array([[1, sqrt(3)], [sqrt(3), 3]]),
        array([[1.0, 0.5], [0.5, 0.25]]),
        1e-7 * ones((2, 2)),
        array([[0.5, 1.0], [0.25, 0.5]]),
        array([[1.0, 0.5], [0.5, 2.0]]),
        array([[0.5, 1.0], [2.0, 0.5]]),
        array([[0.5, 2.0], [2.0, 1.5]]),
        array([[0.5, 2.0], [2.0, 0.0]]),
        array([[0, 2.0], [2.0, 1.5]]),
        array([[0, 2.0], [2.0, 0.0]]),
        array([[0, -2.0], [-2.0, 0.0]]),
        1e-12 * ones((2, 2)),
        1e-15 * ones((2, 2)),
        1e-16 * ones((2, 2)),
        1e-17 * ones((2, 2)),
        1e-20 * ones((2, 2)),
        1e-23 * ones((2, 2)),
        1e-24 * ones((2, 2)),
        1e-25 * ones((2, 2)),
        1e-27 * ones((2, 2)),
        1e-30 * ones((2, 2)),
        1e-50 * ones((2, 2)),
        1e-90 * ones((2, 2)),
        1e-300 * ones((2, 2)),
        array([[1e-300, 0.1], [0.1, 1e-10]]),
        zeros((2, 2)),
        array([[1.24683824e00, 1.10215051e-01], [1.10215051e-01, 1.00000000e04]]),
        array([[1.24683824e00, 1.10215051e-01], [1.10215051e-01, -1.00000000e04]]),
        array([[1.24683824e00, -1.10215051e-01], [-1.10215051e-01, -1.00000000e04]]),
    ]

    A = A + [-a for a in A]

    for a in A:
        x0 = hsolve(a, y)
        x1 = npy_lstsq(a, y, rcond=-1)[0]
        e0 = norm(dot(a, x0) - y)
        e1 = norm(dot(a, x1) - y)

        assert_allclose(e0, e1, atol=1e-7)
        assert_(npy_all(isfinite(x0)))
        assert_allclose(x0, x1)


def test_economic_svd_zero_rank():
    A = zeros((3, 2))
    SVD = economic_svd(A)
    assert_(SVD[0].shape == (3, 0))
    assert_(SVD[1].shape == (0,))
    assert_(SVD[2].shape == (0, 2))


def test_kron_dot():
    random = RandomState(0)
    A = random.randn(2, 2)
    B = random.randn(5, 3)
    C = random.randn(3, 2)
    out = dot(kron(A, B), _vec(C)).reshape((5, 2), order="F")
    assert_allclose(kron_dot(A, B, C), out)
    out2 = empty((5, 2))
    assert_allclose(out, out2)


def _vec(X):
    return X.reshape(X.shape[0] * X.shape[1], order="F")
