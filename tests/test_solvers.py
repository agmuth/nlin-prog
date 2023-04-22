import numpy as np
import pytest

from nlinprog.solvers import newtons_method, conjugate_gradient_method
from tests.test_functions import UNCONSTRAINED_OPTIMIZATION_TEST_FUNCTIONS, UncontrainedOptimizationTestFunction


@pytest.mark.parametrize("func", UNCONSTRAINED_OPTIMIZATION_TEST_FUNCTIONS[:-1])
@pytest.mark.parametrize("line_search_method", ["wolfe", "armijo"])
@pytest.mark.parametrize("inverse_hessian_method", ["newton", "bfgs", "dfp", "broyden"])
@pytest.mark.parametrize("atol", [1e-4])
def test_newton_sovlers_grad_atol(
        func: UncontrainedOptimizationTestFunction, 
        line_search_method,
        inverse_hessian_method, 
        atol,
    ):
    res = newtons_method(
        f=func.f,
        x0=func.x_start,
        line_search_method=line_search_method,
        inverse_hessian_method=inverse_hessian_method,
        atol=atol
    )
    assert np.allclose(np.linalg.norm(res["grad"]), 0.0, atol=atol) and res["converged"]

