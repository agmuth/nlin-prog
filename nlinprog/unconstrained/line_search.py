import numpy as np
from typing import Optional
from nlinprog.numerical_differentiation import central_difference

from abc import ABC, abstractclassmethod
from types import MappingProxyType


def sufficient_decrease_condition(phi_of_alpha: np.ndarray, phi_of_zero: np.ndarray, phi_prime_of_zero: np.ndarray, alpha: float, c1: float) -> bool:
    return np.all(phi_of_alpha <= phi_of_zero + c1 * phi_prime_of_zero * alpha)

class LineSearch(ABC):
    @abstractclassmethod
    def __call__(self, x_k, d_k) -> float:
        pass


class ArmijoBacktraackingLineSearch(LineSearch):
    def __init__(self, alpha: Optional[float]=2.0, epsilon: Optional[float]=0.2, eta: Optional[float]=2.0,):
        self.alpha = alpha
        self.epsilon = epsilon
        self.eta = eta
        self.eta_inv = 1/self.eta

    def __call__(self, f: callable, x_k: np.ndarray, d_k: np.ndarray) -> float:
        grad_f = central_difference(f)
        alpha = np.array(self.alpha)
        phi = lambda alpha: f(x_k + alpha*d_k)
        phi_prime = lambda alpha: grad_f(x_k + alpha*d_k).T @ d_k

        zero = 0.0
        phi_of_zero = phi(zero)
        phi_prime_of_zero = phi_prime(zero)

        _armijo_condition = lambda alpha: sufficient_decrease_condition(phi(alpha), phi_of_zero, phi_prime_of_zero, alpha, self.epsilon*self.eta)

        while _armijo_condition(alpha): # alpha does not exceed bound -> increase until alpha violates bound and then take penultimate alpha
            alpha *= self.eta

        while not _armijo_condition(alpha): # alpha exceeds bound -> decrese alpha until it meets bound
            alpha *= self.eta_inv

        return alpha


class WolfeZoomLineSearch(LineSearch):
    def __init__(self, alpha_max: float=2.0, c1: float=0.2, c2: float=2.0):
        self.alpha_max = alpha_max
        self.c1 = c1
        self.c2 = c2

    def __call__(self, f: callable, x_k: np.ndarray, d_k: np.ndarray) -> float:
        # algorithm 3.5 Numerical Optimization

        grad_f = central_difference(f)
        phi = lambda alpha: f(x_k + alpha*d_k)
        phi_prime = lambda alpha: grad_f(x_k + alpha*d_k).T @ d_k

        zero = 0.0
        phi_of_zero = phi(zero)
        phi_prime_of_zero = phi_prime(zero)
        _sufficient_decrease_condition = lambda alpha: sufficient_decrease_condition(phi(alpha), phi_of_zero, phi_prime_of_zero, alpha, self.c1)
        
        alpha_i = 0.0
        i = 0
    
        while True:
            i += 1
            alpha_i_minus_one = alpha_i
            phi_of_alpha_i_minus_one = phi(alpha_i_minus_one) 
            alpha_i = 0.5*(self.alpha_max + alpha_i_minus_one)  # take midpoint

            phi_of_alpha_i = phi(alpha_i)
            if not _sufficient_decrease_condition(alpha_i) or (phi_of_alpha_i > phi_of_alpha_i_minus_one and i > 1):
                # zoom(alpha_i-1, alpha_i)
                return self._zoom(phi, phi_prime, alpha_i_minus_one, alpha_i)
            
            phi_prime_of_alpha_i = phi_prime(alpha_i)
            if np.all(np.abs(phi_prime_of_alpha_i) <= -1*self.c2*phi_prime_of_zero):
                return alpha_i
            
            if phi_prime_of_alpha_i >= 0:
                # zoom(alpha_i, alpha_i-1)
                return self._zoom(phi, phi_prime, alpha_i, alpha_i_minus_one)
            
            if i > 10: return alpha_i # early termination TODO: make better
            
    def _zoom(self, phi: callable, phi_prime: callable, alpha_low: float, alpha_high: float) -> float:
        zero = 0.0
        phi_of_zero = phi(zero)
        phi_prime_of_zero = phi_prime(zero)
        _sufficient_decrease_condition = lambda alpha: sufficient_decrease_condition(phi(alpha), phi_of_zero, phi_prime_of_zero, alpha, self.c1)
        
        while True:
        #TODO: handle early termination
            alpha = 0.5*(alpha_high + alpha_low) # interpolate via bisection #TODO: circle back and make smarter choice
            phi_of_alpha = phi(alpha)

            if not _sufficient_decrease_condition(alpha) or phi_of_alpha > phi(alpha_low):
                alpha_high = alpha
            else:
                phi_prime_of_alpha = phi_prime(alpha)
                if np.all(np.abs(phi_prime_of_alpha) <= -self.c2*phi_prime_of_zero):
                    return alpha
                if phi_prime_of_alpha*(alpha_high - alpha_low):
                    alpha_high = alpha_low
                alpha_low = alpha

            if abs(alpha_high-alpha_low) < 1e-3: # CG-method seems to produce search directions that increase function -> think its numerical issues with calculating derivatives
                return alpha



LINE_SEARCH_MAPPING = MappingProxyType(
    {
        "armijo": ArmijoBacktraackingLineSearch,
        "wolfe": WolfeZoomLineSearch
    }
)







