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
    f=lambda x: np.sum(x**2),
    x_min=np.zeros(2),
    x_start=1*np.ones(2)
)

bazaraa_ex = OptimizationTestFunction(
    f=lambda x: (x[0] - 2)**4 + (x[0] - 2*x[1])**2,
    x_min=np.array([2., 1.]),
    x_start=np.array([0.0, 3.0])
)

booth_func = OptimizationTestFunction(
    f=lambda x: (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2,
    x_min=np.array([1., 3.]),
    x_start=np.array([-8., -8.])
)

matyas_func = OptimizationTestFunction(
    f=lambda x: 0.26*np.sum(x*x) - 0.48*np.prod(x),
    x_min=np.array([0., 0.]),
    x_start=np.array([2., -8.])
)

bulkin_no_6_func = OptimizationTestFunction(
    f=lambda x: 100*np.sqrt(np.abs(x[1] * 0.01*x[0]**2) + 0.01*np.abs(x[0]+10)),
    x_min=np.array([-10., 1.]),
    x_start=np.array([-12., 3.])
)

rosebrock_10d_func = OptimizationTestFunction(
    f=lambda x: 100*np.sum((x[1:] - x[:-1]**2)**2 + (1 - x[:-1]**2)),
    x_min=np.ones(10),
    x_start=-5*np.ones(1)
)

goldstien_price_func = OptimizationTestFunction(
    f=lambda x: (
        (
            1 
            + (x.sum() + 1)**2 
            * (19 - 14*x[0] + 3*x[0]**2 - 14*x[1] + 6*x.prod() + 3*x[1]**2) 
        )
        * (
            30
            + (2*x[0] - 3*x[1])**2
            * (18 - 32*x[0] + 12*x[0]**2 + 48*x[1] - 36*x.prod() + 27*x[1]**2) 
        ) 
    ),
    x_min=np.array([0., -1.]),
    x_start=np.array([-2., 2.])
)


@pytest.mark.parametrize("solver", [steepest_descent])
@pytest.mark.parametrize("func", [sphere_func_2d, bazaraa_ex, booth_func, matyas_func, rosebrock_10d_func, goldstien_price_func])
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

