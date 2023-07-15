import numpy as np
import pytest

from nlinprog.unconstrained.quasi_newton import (BroydenInverseHessian,
                                                 QuasiNewtonMethod)
from tests.unconstrained.unconstrained_test_functions import (
    UNCONSTRAINED_OPTIMIZATION_TEST_FUNCTIONS,
    UnconstrainedOptimizationTestFunction)


@pytest.mark.parametrize("func", UNCONSTRAINED_OPTIMIZATION_TEST_FUNCTIONS)
@pytest.mark.parametrize("line_search_method", ["armijo", "wolfe"])
@pytest.mark.parametrize("maxiters", [1, 2, 5, 10])
@pytest.mark.parametrize("atol", [1e-4])
def test_broyden_bfgs_agreement(
    func: UnconstrainedOptimizationTestFunction, line_search_method, maxiters, atol
):
    bfgs_solver = QuasiNewtonMethod(
        f=func.f, line_search_method=line_search_method, inverse_hessian_method="bfgs"
    )
    broyden_solver = QuasiNewtonMethod(
        f=func.f,
        line_search_method=line_search_method,
        inverse_hessian_method=BroydenInverseHessian(phi=1.0),
    )

    bfgs_res = bfgs_solver.solve(x_0=func.x_start, maxiters=maxiters, atol=atol)
    broyden_res = broyden_solver.solve(x_0=func.x_start, maxiters=maxiters, atol=atol)

    assert np.allclose(bfgs_res.x, broyden_res.x, atol=1e-4)


@pytest.mark.parametrize("func", UNCONSTRAINED_OPTIMIZATION_TEST_FUNCTIONS[:1])
@pytest.mark.parametrize("line_search_method", ["armijo", "wolfe"])
@pytest.mark.parametrize("maxiters", [1, 2, 5, 10])
@pytest.mark.parametrize("atol", [1e-4])
def test_broyden_dfp_agreement(
    func: UnconstrainedOptimizationTestFunction, line_search_method, maxiters, atol
):
    dfp_solver = QuasiNewtonMethod(
        f=func.f, line_search_method=line_search_method, inverse_hessian_method="dfp"
    )
    broyden_solver = QuasiNewtonMethod(
        f=func.f,
        line_search_method=line_search_method,
        inverse_hessian_method=BroydenInverseHessian(phi=0.0),
    )

    dfp_res = dfp_solver.solve(x_0=func.x_start, maxiters=maxiters, atol=atol)
    broyden_res = broyden_solver.solve(x_0=func.x_start, maxiters=maxiters, atol=atol)

    assert np.allclose(dfp_res.x, broyden_res.x, atol=1e-4)
