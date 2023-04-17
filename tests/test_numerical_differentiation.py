import numpy as np
import pytest 

from nlinprog.numerical_differentiation import (
    forward_difference,
    central_difference,
    backward_difference,
)


RTOL = 1e-3


class NumericalDerivativeTestFunction():
    def __init__(self, f:callable, x: np.ndarray, p: np.ndarray, f_prime_res: float) -> None:
        self.f = f,
        self.x = x,
        self.p = p,
        self.f_prime_res = f_prime_res
        

sphere_test_func = NumericalDerivativeTestFunction(
    f=lambda x: np.dot(x, x),
    x=np.array([1, -1]),
    p=np.array([-1, 1]),
    f_prime_res=-4.
)


@pytest.mark.parametrize("calc_derivative", [forward_difference, central_difference, backward_difference])
@pytest.mark.parametrize("test_func", [sphere_test_func])
def test_numerical_first_derivative(calc_derivative: callable, test_func: NumericalDerivativeTestFunction):
    f_prime = calc_derivative(test_func.f[0])
    assert np.allclose(
        f_prime(test_func.x[0], test_func.p[0]), # not sure why being passed in as tuples
        test_func.f_prime_res, 
        rtol=RTOL
    )


