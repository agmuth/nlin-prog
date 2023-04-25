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
    # check that L2 norm of grad is same order of magnitude as atol - some methods do not garuantee that grad is monotonic decreasing. 
    assert np.allclose(np.logaddexp(np.linalg.norm(res["grad"]), atol), 0.0, atol=10) and res["converged"]

@pytest.mark.parametrize("func", UNCONSTRAINED_OPTIMIZATION_TEST_FUNCTIONS[:-1])
@pytest.mark.parametrize("line_search_method", ["wolfe", "armijo"])
@pytest.mark.parametrize("conjugate_gradient_direction_method", ["pr", "fr"])
@pytest.mark.parametrize("atol", [1e-4])
def test_conjugate_gradient_sovlers_grad_atol(
        func: UncontrainedOptimizationTestFunction, 
        line_search_method,
        conjugate_gradient_direction_method, 
        atol,
    ):
    res = conjugate_gradient_method(
        f=func.f,
        x0=func.x_start,
        line_search_method=line_search_method,
        conjugate_gradient_direction_method=conjugate_gradient_direction_method,
        atol=atol
    )
    # check that L2 norm of grad is same order of magnitude as atol - some methods do not garuantee that grad is monotonic decreasing. 
    assert np.allclose(np.logaddexp(np.linalg.norm(res["grad"]), atol), 0.0, atol=10) and res["converged"]
    
    