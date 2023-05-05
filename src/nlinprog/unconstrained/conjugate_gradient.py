import numpy as np
from typing import Optional, Union
from abc import ABC, abstractclassmethod
from types import MappingProxyType

from nlinprog.numerical_differentiation import central_difference as jacobian
from nlinprog.unconstrained.line_search import LINE_SEARCH_MAPPING, LineSearch
from nlinprog.utils import SimpleConvergenceTest, build_result_object


class CalcConjugateGradientDirection(ABC):
    @abstractclassmethod
    def __call__(self):
        pass


class FletcherReevesConjugateGradientDirection(CalcConjugateGradientDirection):
    def __call__(self, d_k: np.ndarray, g_k: np.ndarray, g_k_plus_one: np.ndarray) -> np.ndarray:
        beta_k = g_k_plus_one.T @ g_k_plus_one / (g_k.T @ g_k)
        return -1*g_k_plus_one + beta_k*d_k
    

class PolakRibiereConjugateGradientDirection(CalcConjugateGradientDirection):
    def __call__(self, d_k: np.ndarray, g_k: np.ndarray, g_k_plus_one: np.ndarray) -> np.ndarray:
        beta_k = (g_k_plus_one - g_k).T @ g_k_plus_one / (g_k.T @ g_k)
        return -1*g_k_plus_one + beta_k*d_k


CALC_CONJUGATE_GRADIENT_DIRECTION_MAPPING = MappingProxyType(
    {
        "fletcher-reeves": FletcherReevesConjugateGradientDirection,
        "polak-ribiere": PolakRibiereConjugateGradientDirection
    }
)


class ConjugateGradientMethod():
    def __init__(self, f: callable, line_search_method: Union[LineSearch, str]="wolfe", conjugate_gradient_direction_method: Union[CalcConjugateGradientDirection, str]="fletcher-reeves"):
        self.f = f
        self.line_search = LINE_SEARCH_MAPPING[line_search_method]() if isinstance(line_search_method, str) else line_search_method
        self.calc_conjugate_direction = CALC_CONJUGATE_GRADIENT_DIRECTION_MAPPING[conjugate_gradient_direction_method]() if isinstance(conjugate_gradient_direction_method, str) else conjugate_gradient_direction_method
        

    def solve(self, x_0: np.ndarray, maxiters: int=200, grad_atol: Optional[float]=1e-4):
        # init params for algorithm
        x_k = np.array(x_0).astype(np.float64)
        grad = jacobian(self.f)
        g_k = grad(x_k)
        n = x_k.shape[0]

        # set up convergece tracking
        grad_atol_convergence = SimpleConvergenceTest(np.linalg.norm(g_k), atol=grad_atol) if grad_atol else None
        converged = False

        for i in range(maxiters):
            g_k = grad(x_k).T if i > 0 else g_k
            d_k = -1*g_k
            for k in range(n):
                if grad_atol_convergence:
                    grad_atol_convergence.update(0.0) # dont love
                    grad_atol_convergence.update(np.linalg.norm(g_k))
                    if grad_atol_convergence.converged():
                        converged = True
                
                if converged: break

                alpha_k = self.line_search(f=self.f, x_k=x_k, d_k=d_k)  # minimizer of f(x_k + alpha*d_k)
                x_k += alpha_k*d_k
                g_k_plus_one = grad(x_k).T
                d_k = self.calc_conjugate_direction(d_k=d_k, g_k=g_k, g_k_plus_one=g_k_plus_one)
                g_k = g_k_plus_one
            
            if converged: break

        return build_result_object(self.f, x_k, i, converged)
