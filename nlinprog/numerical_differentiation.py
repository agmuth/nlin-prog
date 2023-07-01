import numpy as np
from typing import Optional


def forward_difference(f: callable, h: Optional[float]=1e-8) -> callable:
    h_inv = h**-1
    def f_prime(x: np.ndarray) -> np.ndarray:
        f_of_x = f(x)
        return h_inv * np.apply_along_axis(arr=np.eye(x.shape[0]), axis=0, func1d=lambda e: f(x + h*e) - f_of_x)
    return f_prime


def central_difference(f: callable, h: Optional[float]=1e-8) -> callable:
    h_inv = h**-1
    def f_prime(x: np.ndarray) -> np.ndarray:
        return 0.5*h_inv*np.apply_along_axis(arr=np.eye(x.shape[0]), axis=0, func1d=lambda e: f(x + h*e) - f(x - h*e))
    return f_prime


def backward_difference(f: callable, h: Optional[float]=1e-8) -> callable:
    h_inv = h**-1
    def f_prime(x: np.ndarray) -> np.ndarray:
        f_of_x = f(x)
        return h_inv * np.apply_along_axis(arr=np.eye(x.shape[0]), axis=0, func1d=lambda e: f_of_x - f(x - h*e))
    return f_prime


def hessian(f: callable, h: Optional[float]=1e-4) -> callable:
    hessian = backward_difference(forward_difference(f, h), h)
    return hessian

