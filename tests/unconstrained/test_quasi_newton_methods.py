import numpy as np
import pytest

from nlinprog.unconstrained.quasi_newton import QuasiNewtonMethod
from tests.unconstrained.unconstrained_test_functions import UNCONSTRAINED_OPTIMIZATION_TEST_FUNCTIONS, UnconstrainedOptimizationTestFunction


@pytest.mark.parametrize("func", UNCONSTRAINED_OPTIMIZATION_TEST_FUNCTIONS[:-1])
@pytest.mark.parametrize("line_search_method", ["wolfe", "armijo"])
@pytest.mark.parametrize("inverse_hessian_method", ["exact", "bfgs", "dfp", "broyden"])
@pytest.mark.parametrize("atol", [1e-4])
def test_newton_sovlers_grad_atol(
        func: UnconstrainedOptimizationTestFunction, 
        line_search_method,
        inverse_hessian_method, 
        atol,
    ):
    solver = QuasiNewtonMethod(f=func.f, line_search_method=line_search_method, inverse_hessian_method=inverse_hessian_method)
    res = solver.solve(x_0=func.x_start, grad_atol=atol)

    # check that L2 norm of grad is same order of magnitude as atol - some methods do not garuantee that grad is monotonic decreasing. 
    assert np.allclose(np.linalg.norm(res.x - func.x_min), 0, atol=0.1)

