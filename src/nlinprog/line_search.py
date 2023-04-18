import numpy as np
from typing import Optional
from nlinprog.numerical_differentiation import central_difference


def armijo_line_search(f: callable, x: np.ndarray, d: np.ndarray, alpha: Optional[float]=1.0, epsilon: Optional[float]=0.2, eta: Optional[float]=2.0) -> float:
    phi = lambda alpha: f(x + alpha*d)
    phi_prime = central_difference(phi)

    zero = np.zeros(1)
    phi_of_zero = phi(zero)
    phi_prime_of_zero = phi_prime(zero)

    armijo_bound = lambda alpha: np.all(phi(alpha) <= phi_of_zero + epsilon * phi_prime_of_zero * alpha)

    if armijo_bound(alpha):
        # alpha does not exceed bound -> increase until alpha violates bound and then take penultimate alpha
        while armijo_bound(alpha):
            # increase alpha until upper bound is violated
            alpha *= eta
        alpha /= eta  # take penultimate alpha
    else:
        # alpha exceeds bound -> decrese alpha until it meets bound
        while not armijo_bound(alpha):
            alpha /= eta

    return alpha

