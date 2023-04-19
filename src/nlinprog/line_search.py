import numpy as np
from typing import Optional
from nlinprog.numerical_differentiation import central_difference


def sufficient_decrease_condition(
    phi_of_alpha: np.ndarray,
    phi_of_zero: np.ndarray,
    phi_prime_of_zero: np.ndarray,
    alpha: float,
    c1: float
) -> bool:
    return np.all(phi_of_alpha <= phi_of_zero + c1 * phi_prime_of_zero * alpha)


def curvature_condition(phi_prime_of_alpha, phi_prime_of_zero, c2) -> bool:
    return np.all(np.abs(phi_prime_of_alpha) <= c2*np.abs(phi_prime_of_zero))


def armijo_backtracking_line_search(f: callable, x: np.ndarray, d: np.ndarray, alpha: Optional[float]=1.0, epsilon: Optional[float]=0.2, eta: Optional[float]=2.0) -> float:
    f_grad = central_difference(f)
    phi = lambda alpha: f(x + alpha*d)
    phi_prime = lambda alpha: f_grad(x + alpha*d).T @ d

    zero = 0.0
    phi_of_zero = phi(zero)
    phi_prime_of_zero = phi_prime(zero)

    _armijo_condition = lambda alpha: sufficient_decrease_condition(phi(alpha), phi_of_zero, phi_prime_of_zero, alpha, epsilon*eta)

    while _armijo_condition(alpha): # alpha does not exceed bound -> increase until alpha violates bound and then take penultimate alpha
        alpha *= eta

    eta_inv = 1/eta
    while not _armijo_condition(alpha): # alpha exceeds bound -> decrese alpha until it meets bound
        alpha *= eta_inv

    return alpha


def wolfe_zoom_line_search(f: callable, x: np.ndarray, d: np.ndarray, alpha_max: Optional[float]=1.0, c1: Optional[float]=0.2, c2: Optional[float]=2.0) -> float:
    alpha_i_minus_one = 0
    alpha_i = 0.5*(alpha_i_minus_one + alpha_max) #TODO: change to interpolation
    i = 0

    f_grad = central_difference(f)
    phi = lambda alpha: f(x + alpha*d)
    phi_prime = lambda alpha: f_grad(x + alpha*d).T @ d

    zero = 0.0
    phi_of_zero = phi(zero)
    phi_prime_of_zero = phi_prime(zero)

    _sufficient_decrease_condition = lambda alpha: sufficient_decrease_condition(phi(alpha), phi_of_zero, phi_prime_of_zero, alpha, c1)
    _curvature_condition_negative_c2 = lambda alpha: curvature_condition(phi(alpha), phi_prime_of_zero, alpha, -c2)

    while True:
        i += 1
        phi_of_alpha_i = phi(alpha_i)
        if _sufficient_decrease_condition(alpha_i) or (phi(alpha_i) > phi(alpha_i_minus_one) and i > 1):
            return zoom(phi, phi_prime, alpha_i_minus_one, alpha_i)
        
        phi_prime_of_alpha_i = phi_prime(alpha_i)
        if np.all(np.abs(phi_prime_of_alpha_i) <= -c2*phi_prime_of_zero):
            return alpha_i
        
        if phi_of_alpha_i >= 0:
            return zoom(phi, phi_prime, alpha_i, alpha_i_minus_one)
        
        alpha_i_minus_one = alpha_i
        alpha_i = 0.5*(alpha_i + alpha_max) #TODO: change to interpolation


def zoom(phi: callable, phi_prime: callable, alpha_low: float, alpha_high: float, epsilon: Optional[float]=0.2) -> float:
    alpha_high = alpha_high
    alpha_low = alpha_low
    alpha = 0.5*(alpha_high + alpha_low)
    phi_of_alpha = phi(alpha)

    zero = 0.0
    phi_of_zero = phi(zero)
    phi_prime_of_zero = phi_prime(zero)

    armijo_condition = lambda alpha: np.all(phi(alpha) <= phi_of_zero + epsilon * phi_prime_of_zero * alpha)

    while True:
        if not armijo_condition(alpha) or phi_of_alpha > phi(alpha_low):
            alpha_high = alpha
        else:
            phi_prime_of_alpha = phi_prime(alpha)
            if np.all(np.abs(phi_prime_of_alpha) <= -(1-epsilon)*phi_prime_of_zero):
                return alpha
            if phi_prime_of_alpha*(alpha_high - alpha_low):
                alpha_high = alpha_low
            alpha_low = alpha


