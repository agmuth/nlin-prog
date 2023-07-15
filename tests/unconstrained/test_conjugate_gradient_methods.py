import numpy as np
import pytest

from nlinprog.unconstrained.conjugate_gradient import ConjugateGradientMethod
from tests.unconstrained.unconstrained_test_functions import (
    UNCONSTRAINED_OPTIMIZATION_TEST_FUNCTIONS,
    UnconstrainedOptimizationTestFunction)


@pytest.mark.parametrize("func", UNCONSTRAINED_OPTIMIZATION_TEST_FUNCTIONS[:-1])
@pytest.mark.parametrize("line_search_method", ["armijo", "wolfe"])
@pytest.mark.parametrize(
    "conjugate_gradient_direction_method", ["polak-ribiere", "fletcher-reeves"]
)
@pytest.mark.parametrize("atol", [1e-4])
def test_conjugate_gradient_sovlers_grad_atol(
    func: UnconstrainedOptimizationTestFunction,
    line_search_method,
    conjugate_gradient_direction_method,
    atol,
):
    solver = ConjugateGradientMethod(
        f=func.f,
        line_search_method=line_search_method,
        conjugate_gradient_direction_method=conjugate_gradient_direction_method,
    )
    res = solver.solve(x_0=func.x_start, atol=atol)
    assert np.abs(func.f(func.x_min) - res.func) < np.sqrt(atol)
