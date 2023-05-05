import numpy as np
import pytest

from nlinprog.constrained.penalty.conjugate_gradient import PenalizedConjugateGradientMethod
from tests.constrained.constrained_test_functions import CONSTRAINED_OPTIMIZATION_TEST_FUNCTIONS, ConstrainedOptimizationTestFunction


@pytest.mark.parametrize("func", CONSTRAINED_OPTIMIZATION_TEST_FUNCTIONS)
@pytest.mark.parametrize("line_search_method", ["armijo", "wolfe"])
@pytest.mark.parametrize("conjugate_gradient_direction_method", ["polak-ribiere", "fletcher-reeves"])
@pytest.mark.parametrize("atol", [1e-8])
def test_penalize_conjugate_gradient_sovlers_grad_atol(
        func: ConstrainedOptimizationTestFunction, 
        line_search_method,
        conjugate_gradient_direction_method, 
        atol,
    ):
    solver = PenalizedConjugateGradientMethod(f=func.f, g=func.g, h=func.h, line_search_method=line_search_method, conjugate_gradient_direction_method=conjugate_gradient_direction_method)
    res = solver.solve(x_0=func.x_start, grad_atol=atol)

    # check that L2 norm of grad is same order of magnitude as atol - some methods do not garuantee that grad is monotonic decreasing. 
    assert np.allclose(np.logaddexp(np.linalg.norm(res.grad), atol), 0.0, atol=10) and res.converged
    
    