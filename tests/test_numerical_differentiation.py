import numpy as np
import pytest 

from nlinprog.numerical_differentiation import (
    forward_difference,
    central_difference,
    backward_difference,
    jacobian
)


RTOL = 1e-3


class NumericalDerivativeTestFunction():
    def __init__(self, f:callable, x: np.ndarray, p: np.ndarray, f_prime_res: float, f_jacobian: np.ndarray) -> None:
        self.f = f
        self.x = x
        self.p = p
        self.f_prime_res = f_prime_res
        self.f_jacobian = f_jacobian
        

sphere_test_func = NumericalDerivativeTestFunction(
    f=lambda x: np.sum(x*x),
    x=np.array([[1, -1]]).T,
    p=np.array([[-1, 1]]).T,
    f_prime_res=-4.,
    f_jacobian=np.array([[2, -2]]).T,
)


@pytest.mark.parametrize("calc_derivative", [forward_difference, central_difference, backward_difference])
@pytest.mark.parametrize("func", [sphere_test_func])
def test_numerical_first_derivative(calc_derivative: callable, func: NumericalDerivativeTestFunction):
    assert np.allclose(
        calc_derivative(func.f)(func.x, func.p), 
        func.f_prime_res, 
        rtol=RTOL
    )


@pytest.mark.parametrize("func", [sphere_test_func])
def test_jacobian(func: NumericalDerivativeTestFunction):
    assert np.allclose(
        jacobian(func.f)(func.x), 
        func.f_jacobian,
        rtol=RTOL
    )