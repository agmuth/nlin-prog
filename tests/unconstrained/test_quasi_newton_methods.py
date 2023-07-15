import numpy as np
import pytest

from nlinprog.unconstrained.quasi_newton import QuasiNewtonMethod
from tests.unconstrained.unconstrained_test_functions import (
    UNCONSTRAINED_OPTIMIZATION_TEST_FUNCTIONS,
    UnconstrainedOptimizationTestFunction)


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
    solver = QuasiNewtonMethod(
        f=func.f,
        line_search_method=line_search_method,
        inverse_hessian_method=inverse_hessian_method,
    )
    res = solver.solve(x_0=func.x_start, atol=atol)
    assert np.abs(func.f(func.x_min) - res.func) < np.sqrt(atol)
