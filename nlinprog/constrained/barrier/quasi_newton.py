import numpy as np
from typing import Optional, Union

from nlinprog.unconstrained.line_search import LineSearch
from nlinprog.utils import SimpleConvergenceTest, build_result_object
from nlinprog.unconstrained.quasi_newton import QuasiNewtonMethod, ApproxInverseHessian
from nlinprog.constrained.barrier.barrier_functions import BARRIER_FUNCTIONS_MAPPING, BarrierFunction


class BarrierQuasiNewtonMethod():
    def __init__(self, f: callable, g: Optional[callable]=None, g_barrier: Optional[Union[BarrierFunction, str]]="inverse", line_search_method: Union[LineSearch, str]="wolfe", inverse_hessian_method: Union[ApproxInverseHessian, str]="exact"):
        self.f = f
        self.g = g
        self.g_barrier = BARRIER_FUNCTIONS_MAPPING[g_barrier]()(g) if isinstance(g_barrier, str) else g_barrier(g) 
        self.solver = QuasiNewtonMethod(f=f, line_search_method=line_search_method, inverse_hessian_method=inverse_hessian_method)

    def solve(self, x_0: np.ndarray, mu: float=2.0, beta: float=2.0, grad_atol: Optional[float]=1e-4, delta_x_atol: Optional[float]=1e-4, maxiters1: int=200, maxiters2: int=200, **kwargs):
        # init params for algorithm
        x_k = np.array(x_0).astype(np.float64)
        p_k = None
        barrier_func = lambda x: self.g_barrier(x)
        grad_tol_convergence = SimpleConvergenceTest(-1., atol=grad_atol)
        converged = False
        beta_inv = beta**-1

        for k in range(maxiters1):
            # check for convergence 
            if grad_tol_convergence and k > 0:
                grad_tol_convergence.update(np.linalg.norm(res_k.grad))
                if grad_tol_convergence.converged() and np.linalg.norm(p_k) < delta_x_atol:
                    converged = True

            if converged: break

            objective_func_k = lambda x: self.f(x) + mu * barrier_func(x)
            self.solver.f = objective_func_k
            res_k = self.solver.solve(x_0=x_k, maxiters=maxiters2, **kwargs)
            p_k = res_k.x - x_k
            x_k = res_k.x
            mu *= beta_inv

        return build_result_object(self.f, x_k, k, converged)
    
