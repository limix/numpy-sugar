from numpy_sugar.linalg import economic_qs
from numpy_sugar.linalg import trace2
from numpy_sugar.linalg import dotd
from numpy_sugar.linalg import ddot
from numpy_sugar.linalg import solve
from numpy.linalg import solve as npy_solve
from numpy.linalg import cholesky
from numpy import empty
from numpy import diag
from numpy.random import RandomState
from numpy.testing import assert_allclose


def test_economic_qs():
    random = RandomState(633)
    A = random.randn(10, 10)
    Q, S = economic_qs(A)

    e = [[-0.30665477, -0.06935249, -0.19790895, -0.31966245, -0.2041274],
         [-0.41417631, 0.70463554, -0.029418, 0.23839354, -0.01000668],
         [-0.15609931, 0.30659134, 0.12898542, 0.21192988, 0.40325725],
         [0.23709357, 0.03994193, -0.12559863, -0.08280338, 0.07192297],
         [0.03497126, -0.2059239, 0.13106679, -0.28509727, 0.42246837],
         [0.01812449, 0.22538233, -0.8011112, -0.24884526, 0.09372236],
         [0.21498028, 0.45112184, 0.49290517, -0.46891478, -0.06614824],
         [-0.26556389, -0.12489177, 0.09633139, 0.183802, -0.6557185],
         [0.2334407, 0.24061674, -0.0327931, -0.42773492, -0.37992895],
         [-0.69358038, -0.1813236, 0.12363693, -0.45758927, 0.15649533]]

    assert_allclose(Q[0], e, rtol=1e-5)

    e = [[0.2602789, -0.16562995, 0.3652314, 0.69671087, -0.06445665],
         [-0.08773482, -0.01183062, 0.17659833, -0.12014128, -0.46977809],
         [-0.28447028, 0.05696058, 0.0320373, 0.37750777, 0.6555612],
         [-0.52350264, -0.24840377, -0.50625997, 0.4225672, -0.37916396],
         [-0.31660658, 0.56505551, 0.39638672, 0.04048807, -0.31803257],
         [0.1010443, 0.37774975, -0.21705131, -0.10319173, 0.16038291],
         [0.34690429, 0.24425194, -0.2942541, 0.12823675, 0.00535483],
         [-0.28705797, 0.54312199, -0.19616314, 0.13677848, 0.07922182],
         [-0.48731195, -0.23394529, 0.38236667, -0.24101999, 0.25046401],
         [-0.13751969, -0.19006396, -0.32029686, -0.27120733, 0.07565524]]

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
    assert_allclose(r, dotd(A, B, out=r1))


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

<<<<<<< HEAD
=======
def test_cdot():
    random = RandomState(0)
    A = random.randn(3, 3)
    L = cholesky(A)
    print(K)

>>>>>>> 745dd52a96c571bffa6493d31e848b699c972473

if __name__ == '__main__':
    from pytest import main
    main(['-s'])
