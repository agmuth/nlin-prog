import numpy as np
import pytest

from nlinprog.solvers import newtons_method
from tests.test_functions import UNCONSTRAINED_OPTIMIZATION_TEST_FUNCTIONS, UncontrainedOptimizationTestFunction


@pytest.mark.parametrize("func", UNCONSTRAINED_OPTIMIZATION_TEST_FUNCTIONS)
@pytest.mark.parametrize("line_search_method", ["armijo", "wolfe"])
@pytest.mark.parametrize("maxiters", [1, 2, 5, 10])
@pytest.mark.parametrize("atol", [1e-4])
def test_broyden_bfgs_agreement(
        func: UncontrainedOptimizationTestFunction, 
        line_search_method, 
        maxiters,
        atol
    ):

    res_bfgs = newtons_method(
        f=func.f,
        x0=func.x_start,
        line_search_method=line_search_method,
        inverse_hessian_method="bfgs",
        maxiters=maxiters,
        atol=atol
    )

    res_broyden = newtons_method(
        f=func.f,
        x0=func.x_start,
        line_search_method=line_search_method,
        inverse_hessian_method="bfgs",
        maxiters=maxiters,
        atol=atol,
        phi=1.0
    )

    assert np.allclose(res_bfgs["x"], res_broyden["x"], atol=atol)


@pytest.mark.parametrize("func", UNCONSTRAINED_OPTIMIZATION_TEST_FUNCTIONS[:1])
@pytest.mark.parametrize("line_search_method", ["armijo", "wolfe"])
@pytest.mark.parametrize("maxiters", [1, 2, 5, 10])
@pytest.mark.parametrize("atol", [1e-4])
def test_broyden_dfp_agreement(
        func: UncontrainedOptimizationTestFunction, 
        line_search_method, 
        maxiters,
        atol
    ):

    res_dfp = newtons_method(
        f=func.f,
        x0=func.x_start,
        line_search_method=line_search_method,
        inverse_hessian_method="dfp",
        maxiters=maxiters,
        atol=atol
    )

    res_broyden = newtons_method(
        f=func.f,
        x0=func.x_start,
        line_search_method=line_search_method,
        inverse_hessian_method="bfgs",
        maxiters=maxiters,
        atol=atol,
        phi=0.0
    )

    assert np.allclose(res_dfp["x"], res_broyden["x"], atol=atol)

