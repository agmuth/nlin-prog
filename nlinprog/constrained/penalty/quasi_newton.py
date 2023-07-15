import numpy as np
from typing import Optional, Union

from nlinprog.line_search import LineSearch
from nlinprog.utils import SimpleConvergenceTest, build_result_object
from nlinprog.unconstrained.quasi_newton import QuasiNewtonMethod, ApproxInverseHessian
from nlinprog.constrained.penalty.penalty_functions import INEQUALITY_PENALTY_FUNCTIONS_MAPPING, EQUALITY_PENALTY_FUNCTIONS_MAPPING, InequalityPenaltyFunction, EqualityPenaltyFunction, zero_func


class PenalizedQuasiNewtonMethod():
    def __init__(self, f: callable, g: Optional[callable]=None, h: Optional[callable]=None, g_penalty: Optional[Union[InequalityPenaltyFunction, str]]="relu-squared", h_penalty: Optional[Union[EqualityPenaltyFunction, str]]="squared", line_search_method: Union[LineSearch, str]="wolfe", inverse_hessian_method: Union[ApproxInverseHessian, str]="exact"):
        self.f = f
        self.g = g
        self.h = h
        self.g_penalized = zero_func if not g else INEQUALITY_PENALTY_FUNCTIONS_MAPPING[g_penalty]()(g) if isinstance(g_penalty, str) else g_penalty(g) 
        self.h_penalized = zero_func if not h else EQUALITY_PENALTY_FUNCTIONS_MAPPING[h_penalty]()(h) if isinstance(h_penalty, str) else h_penalty(h) 
        self.solver = QuasiNewtonMethod(f=f, line_search_method=line_search_method, inverse_hessian_method=inverse_hessian_method)

    def solve(self, x_0: np.ndarray, mu: float=2.0, beta: float=2.0, grad_atol: Optional[float]=1e-4, delta_x_atol: Optional[float]=1e-4, maxiters1: int=200, maxiters2: int=200, **kwargs):
        # init params for algorithm
        x_k = np.array(x_0).astype(np.float64)
        p_k = None
        penalty_func = lambda x: self.g_penalized(x) + self.h_penalized(x)
        grad_tol_convergence = SimpleConvergenceTest(-1., atol=grad_atol)
        converged = False

        for k in range(maxiters1):
            # check for convergence 
            if grad_tol_convergence and k > 0:
                grad_tol_convergence.update(np.linalg.norm(res_k.grad))
                if grad_tol_convergence.converged() and np.linalg.norm(p_k) < delta_x_atol:
                    converged = True

            if converged: break

            objective_func_k = lambda x: self.f(x) + mu * penalty_func(x)
            self.solver.f = objective_func_k
            res_k = self.solver.solve(x_0=x_k, maxiters=maxiters2, **kwargs)
            p_k = res_k.x - x_k
            x_k = res_k.x
            mu *= beta

        return build_result_object(self.f, x_k, k, converged)
