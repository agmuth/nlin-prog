import numpy as np
import pytest

from nlinprog.solvers import steepest_descent
from nlinprog.numerical_differentiation import central_difference
from nlinprog.line_search import armijo_backtracking_line_search, wolfe_zoom_line_search


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



@pytest.mark.parametrize("solver", [steepest_descent])
@pytest.mark.parametrize("func", [sphere_func_2d, bazaraa_ex])
@pytest.mark.parametrize("line_search", [armijo_backtracking_line_search, wolfe_zoom_line_search])
@pytest.mark.parametrize("atol", [1e-4])
def test_sovlers_atol(solver: callable, func: OptimizationTestFunction, line_search, atol):
    x_res = solver(func.f, func.x_start, line_search, atol=atol)
    grad = central_difference(func.f)
    assert np.allclose(
        np.linalg.norm(grad(x_res)), 
        0.0,
        atol=atol
    )

