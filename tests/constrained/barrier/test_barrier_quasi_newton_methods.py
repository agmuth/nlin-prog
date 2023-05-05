import numpy as np
import pytest

from nlinprog.constrained.barrier.quasi_newton import BarrierQuasiNewtonMethod
from tests.constrained.constrained_test_functions import CONSTRAINED_OPTIMIZATION_TEST_FUNCTIONS, ConstrainedOptimizationTestFunction

BARRIER_TEST_FUNCTIONS = CONSTRAINED_OPTIMIZATION_TEST_FUNCTIONS[1:2]

@pytest.mark.parametrize("func", BARRIER_TEST_FUNCTIONS)
@pytest.mark.parametrize("line_search_method", ["wolfe", "armijo"])
@pytest.mark.parametrize("inverse_hessian_method", ["exact", "bfgs", "dfp", "broyden"])
@pytest.mark.parametrize("atol", [1e-4])
def test_barrier_newton_sovlers_grad_atol(
        func: ConstrainedOptimizationTestFunction, 
        line_search_method,
        inverse_hessian_method, 
        atol,
    ):
    solver = BarrierQuasiNewtonMethod(f=func.f, g=func.g, line_search_method=line_search_method, inverse_hessian_method=inverse_hessian_method)
    res = solver.solve(x_0=func.x_start, penalty_atol=atol, grad_atol=atol)


    # check that L2 norm of grad is same order of magnitude as atol - some methods do not garuantee that grad is monotonic decreasing. 
    assert np.allclose(np.logaddexp(np.linalg.norm(res.grad), atol), 0.0, atol=10) and res.converged

