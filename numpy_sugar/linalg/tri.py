from scipy.linalg import solve_triangular


def stl(a, b):
    return solve_triangular(a, b, lower=True, check_finite=False)


def stu(a, b):
    return solve_triangular(a, b, lower=False, check_finite=False)


def tri_solve(L, x):
    return solve_triangular(L, x, lower=True, check_finite=False)
