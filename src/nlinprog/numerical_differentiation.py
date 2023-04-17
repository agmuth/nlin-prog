import numpy as np
from typing import Optional

def forward_difference(f: callable, h: Optional[float]=1e-4) -> callable:
    h_inv = h**-1
    def f_prime(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return h_inv * (f(x + h*p) - f(x))
    return f_prime


def central_difference(f: callable, h: Optional[float]=1e-4) -> callable:
    h_inv = h**-1
    def f_prime(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return 0.5*h_inv * (f(x + h*p) - f(x - h*p))
    return f_prime


def backward_difference(f: callable, h: Optional[float]=1e-4) -> callable:
    h_inv = h**-1
    def f_prime(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return h_inv * (f(x) - f(x - h*p))
    return f_prime