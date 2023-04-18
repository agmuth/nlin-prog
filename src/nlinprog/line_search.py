import numpy as np
from typing import Optional
from nlinprog.numerical_differentiation import central_difference


def armijo_line_search(f: callable, x: np.ndarray, d: np.ndarray, alpha: Optional[float]=1.0, epsilon: Optional[float]=0.2, eta: Optional[float]=2.0) -> float:
    phi = lambda alpha: f(x + alpha*d)
    phi_prime = central_difference(phi)

    zero = np.zeros(1)
    phi_of_zero = phi(zero)
    phi_prime_of_zero = phi_prime(zero)

    armijo_condition = lambda alpha: np.all(phi(alpha) <= phi_of_zero + epsilon * phi_prime_of_zero * alpha)

    if armijo_condition(alpha):
        # alpha does not exceed bound -> increase until alpha violates bound and then take penultimate alpha
        while armijo_condition(alpha):
            # increase alpha until upper bound is violated
            alpha *= eta
        alpha /= eta  # take penultimate alpha
    else:
        # alpha exceeds bound -> decrese alpha until it meets bound
        while not armijo_condition(alpha):
            alpha /= eta

    return alpha


def wolfe_line_search(f: callable, x: np.ndarray, d: np.ndarray, alpha_max: Optional[float]=1.0, epsilon: Optional[float]=0.2, eta: Optional[float]=2.0) -> float:
    alpha_i_minus_one = 0
    alpha_i = 0.5*(alpha_i_minus_one + alpha_max)
    i = 0

    phi = lambda alpha: f(x + alpha*d)
    phi_prime = central_difference(phi)

    zero = np.zeros(1)
    phi_of_zero = phi(zero)
    phi_prime_of_zero = phi_prime(zero)

    armijo_condition = lambda alpha: np.all(phi(alpha) <= phi_of_zero + epsilon * phi_prime_of_zero * alpha)

    while True:
        i += 1
        phi_of_alpha_i = phi(alpha_i)
        if armijo_condition(alpha_i) or (phi(alpha_i) > phi(alpha_i_minus_one) and i > 1):
            return zoom(phi, phi_prime, alpha_i_minus_one, alpha_i)
        
        phi_prime_of_alpha_i = phi_prime(alpha_i)
        if np.all(np.abs(phi_prime_of_alpha_i) <= -(1-epsilon)*phi_prime_of_zero):
            return alpha_i
        
        if phi_of_alpha_i >= 0:
            return zoom(phi, phi_prime, alpha_i, alpha_i_minus_one)
        
        alpha_i_minus_one = alpha_i
        alpha_i = 0.5*(alpha_i + alpha_max)


def zoom(phi: callable, phi_prime: callable, alpha_low: float, alpha_high: float, epsilon: Optional[float]=0.2) -> float:
    alpha_high = alpha_high*np.ones(1)
    alpha_low = alpha_low*np.ones(1)
    alpha = 0.5*(alpha_high + alpha_low)
    phi_of_alpha = phi(alpha)

    zero = np.zeros(1)
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


