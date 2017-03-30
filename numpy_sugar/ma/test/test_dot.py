from numpy.ma import dot, masked_array
from numpy.random import RandomState
from numpy_sugar.ma import dotd


def test_dotd_masked():
    random = RandomState(958)
    A = random.randn(3, 2)
    B = random.randn(2, 3)

    A = masked_array(A, mask=[[0, 0], [1, 0], [0, 0]])
    B = masked_array(B, mask=[[1, 0], [0, 0], [0, 1]])

    print(dot(A, B).diagonal())
    print(dotd(A, B))
