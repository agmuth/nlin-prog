import numpy as np
from typing import Optional
from nlinprog.numerical_differentiation import central_difference

from abc import ABC, abstractclassmethod
from types import MappingProxyType


class LineSearch(ABC):
    @abstractclassmethod
    def __call__(self, x_k, d_k) -> float:
        pass
    
class SufficientDecreaseCondition:
    """Sufficient Decrease Condition
    ref: pg.33 Numerical Optimization Nocedal + Wright
    """
    def __init__(self, phi: callable, c1: float):
        self.phi = phi
        self.phi_prime = central_difference(phi)
        self.c1 = c1
        zeros = np.zeros(1)
        self.phi_of_zero = self.phi(zeros)
        self.phi_prime_of_zero = self.phi_prime(zeros)
    
    def __call__(self, alpha: float) -> bool:
        return np.all(self.phi(alpha) <= self.phi_of_zero + self.c1 * alpha * self.phi_prime_of_zero)


class CurvatureCondition:
    """Curvature Condition
    ref: pg. 33 Numerical Optimization Nocedal + Wright
    """
    def __init__(self, phi: callable, c2: float):
        self.phi_prime = central_difference(phi)
        self.c2 = c2
        zeros = np.zeros(1)
        self.phi_prime_of_zero = self.phi_prime(zeros)
    
    def __call__(self, alpha: float) -> bool:
        return np.all(self.phi_prime(alpha*np.ones(1)) >= self.c2 * self.phi_prime_of_zero)



class ArmijoBacktraackingLineSearch(LineSearch):
    def __init__(self, alpha: Optional[float]=2.0, epsilon: Optional[float]=0.2, eta: Optional[float]=2.0, *args, **kwargs):
        """Armijo backtracking line search.
        ref: pg.231 Linear and Nonlinear programming Luenberger + Ye
        ref: pg. 37 Numerical Optimization Nocedal + Wright

        Parameters
        ----------
        alpha : Optional[float], optional
            Initial step length each time functino is called, by default 2.0
        epsilon : Optional[float], optional
            Epislon to use in Armijo's sufficient decrease condition, by default 0.2
        eta : Optional[float], optional
            Amount by which to scale alpha up/down if current step length does/does not exceed Armijo condition , by default 2.0
        """
        self.alpha = alpha
        self.epsilon = epsilon
        self.eta = eta
        self.eta_inv = 1/self.eta

    def __call__(self, f: callable, x_k: np.ndarray, d_k: np.ndarray) -> float:
        alpha = np.array(self.alpha)
        phi = lambda alpha: f(x_k + alpha*d_k)
        
        armijo_condition = SufficientDecreaseCondition(phi=phi, c1=self.epsilon*self.eta)

        while not armijo_condition(alpha): # alpha exceeds bound -> decrese alpha until it meets bound
            alpha *= self.eta_inv

        return alpha


class WolfeZoomLineSearch(LineSearch):
    def __init__(self, alpha_max: float=2.0, c1: float=1e-4, c2: float=0.9, *args, **kwargs):
        """Wolfe zoom line search. 
        ref: pg. 61 Numerical Optimization Nocedal + Wright

        Parameters
        ----------
        alpha_max : float, optional
            Maximum posible step size, by default 2.0
        c1 : float, optional
            Coefficient used in sufficient decrease condition, by default 0.2
        c2 : float, optional
            Coefficient used in curvature condition, by default 2.0
        """
        self.alpha_max = alpha_max
        self.c1 = c1
        self.c2 = c2
        self._early_termination_width = 1e-4
        self._early_termination_iters = 10

    def __call__(self, f: callable, x_k: np.ndarray, d_k: np.ndarray) -> float:
        # algorithm 3.5 Numerical Optimization Nocedal + Wright
        grad_f = central_difference(f)
        phi = lambda alpha: f(x_k + alpha*d_k)
        phi_prime = lambda alpha: grad_f(x_k + alpha*d_k).T @ d_k

        zero = 0.0
        phi_prime_of_zero = phi_prime(zero)
        sufficient_decrease_condition = SufficientDecreaseCondition(phi=phi, c1=self.c1)
        
        alpha_i = 0.0
        i = 0
    
        while True:
            i += 1
            alpha_i_minus_one = alpha_i
            phi_of_alpha_i_minus_one = phi(alpha_i_minus_one) 
            alpha_i = 0.5*(self.alpha_max + alpha_i_minus_one)  # take midpoint

            phi_of_alpha_i = phi(alpha_i)
            if (
                 not sufficient_decrease_condition(alpha_i) 
                 or (phi_of_alpha_i > phi_of_alpha_i_minus_one and i > 1)
            ):
                # violates sufficent decrease cond.
                # zoom(alpha_i-1, alpha_i)
                return self._zoom(phi, phi_prime, alpha_i_minus_one, alpha_i)
            
            phi_prime_of_alpha_i = phi_prime(alpha_i)
            if np.all(np.abs(phi_prime_of_alpha_i) <= -1*self.c2*phi_prime_of_zero):
                return alpha_i
            
            if phi_prime_of_alpha_i >= 0:
                # zoom(alpha_i, alpha_i-1)
                return self._zoom(phi, phi_prime, alpha_i, alpha_i_minus_one)
            
            if i > self._early_termination_iters: return alpha_i # early termination TODO: make better
            
    def _zoom(self, phi: callable, phi_prime: callable, alpha_low: float, alpha_high: float) -> float:
        # algorithm 3.6 Numerical Optimization Nocedal + Wright
        zero = 0.0
        phi_of_zero = phi(zero)
        phi_prime_of_zero = phi_prime(zero)
        
        while True:
            alpha = 0.5*(alpha_high + alpha_low)
            phi_of_alpha = phi(alpha)
            if (
                (phi_of_alpha > phi_of_zero + self.c1*alpha*phi_prime_of_zero)
                or (phi_of_alpha > phi(alpha_low))
            ):
                alpha_high = alpha
            else:
                phi_prime_of_alpha = phi_prime(alpha)
                if abs(phi_prime_of_alpha) <= -self.c2*phi_prime_of_zero: 
                    return alpha 
                if phi_prime_of_alpha*(alpha_high - alpha_low) > 0:
                    alpha_high = alpha_low
                alpha_low = alpha
                    
            if abs(alpha_high-alpha_low) < self._early_termination_width: 
                return alpha
        



LINE_SEARCH_MAPPING = MappingProxyType(
    {
        "armijo": ArmijoBacktraackingLineSearch,
        "wolfe": WolfeZoomLineSearch
    }
)







