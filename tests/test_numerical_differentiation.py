import numpy as np
import pytest

from nlinprog.numerical_differentiation import (backward_difference,
                                                central_difference,
                                                forward_difference, hessian)

ATOL = 1e-4


class NumericalDerivativeTestFunction:
    def __init__(
        self, f: callable, x: np.ndarray, f_jacobian: np.ndarray, f_hessian: np.ndarray
    ) -> None:
        self.f = f
        self.x = x
        self.f_jacobian = f_jacobian
        self.f_hessian = f_hessian


sphere_test_func = NumericalDerivativeTestFunction(
    f=lambda x: np.sum(x * x),
    x=np.array([1, -1]),
    f_jacobian=np.array([2, -2]),
    f_hessian=np.array([[2.0, 0.0], [0.0, 2.0]]),
)


@pytest.mark.parametrize(
    "jacobian", [forward_difference, backward_difference, central_difference]
)
@pytest.mark.parametrize("func", [sphere_test_func])
def test_numerical_jacobian(jacobian: callable, func: NumericalDerivativeTestFunction):
    assert np.allclose(jacobian(func.f)(func.x), func.f_jacobian, atol=ATOL)


@pytest.mark.parametrize("func", [sphere_test_func])
def test_hessian(func: NumericalDerivativeTestFunction):
    assert np.allclose(hessian(func.f)(func.x), func.f_hessian, atol=ATOL)
