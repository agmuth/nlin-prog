import numpy as np
import pytest

from nlinprog.constrained.penalty.conjugate_gradient import \
    PenalizedConjugateGradientMethod
from tests.constrained.constrained_test_functions import (
    CONSTRAINED_OPTIMIZATION_TEST_FUNCTIONS,
    ConstrainedOptimizationTestFunction)

PENALIZED_TEST_FUNCTIONS = [
    func
    for func in CONSTRAINED_OPTIMIZATION_TEST_FUNCTIONS
    if not func.min_not_on_boundary or func.h
]


@pytest.mark.parametrize("func", PENALIZED_TEST_FUNCTIONS)
@pytest.mark.parametrize("line_search_method", ["wolfe"])
@pytest.mark.parametrize(
    "conjugate_gradient_direction_method", ["polak-ribiere", "fletcher-reeves"]
)
@pytest.mark.parametrize("atol", [1e-4])
def test_penalize_conjugate_gradient_sovlers_grad_atol(
    func: ConstrainedOptimizationTestFunction,
    line_search_method,
    conjugate_gradient_direction_method,
    atol,
):
    solver = PenalizedConjugateGradientMethod(
        f=func.f,
        g=func.g,
        h=func.h,
        line_search_method=line_search_method,
        conjugate_gradient_direction_method=conjugate_gradient_direction_method,
    )
    res = solver.solve(
        x_0=func.x_start, atol1=atol, atol2=atol**2, rtol1=atol, rtol2=atol**2
    )
    assert np.allclose(np.abs(res.func - func.f(func.x_min)), 0, atol=atol)
