import numpy as np
import pytest

from nlinprog.solvers import steepest_descent


class OptimizationTestFunction():
    def __init__(self, f: callable, x_min: np.ndarray, x_start: np.ndarray):
        self.f = f
        self.x_min = x_min
        self.x_start = x_start


sphere_func_2d = OptimizationTestFunction(
    f = lambda x: np.sum(x**2),
    x_min=np.zeros(2),
    x_start=1*np.ones(2)
)

bazaraa_ex = OptimizationTestFunction(
    f = lambda x: (x[0] - 2)**4 + (x[0] - 2*x[1])**2,
    x_min=np.array([2., 1.]),
    x_start=np.array([0.0, 3.0])
)

func = bazaraa_ex
x_res = steepest_descent(func.f, func.x_start)

@pytest.mark.parametrize("solver", [steepest_descent])
@pytest.mark.parametrize("func", [sphere_func_2d, bazaraa_ex])
@pytest.mark.parametrize("atol", [1e-4])
def test_sovlers_atol(solver: callable, func: OptimizationTestFunction, atol):
    x_res = solver(func.f, func.x_start, atol=atol)
    assert np.allclose(x_res, func.x_min, atol=atol)