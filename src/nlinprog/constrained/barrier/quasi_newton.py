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

    def solve(self, x_0: np.ndarray, mu: float=2.0, beta: float=2.0, penalty_atol: Optional[float]=1e-4, maxiters1: int=200, maxiters2: int=200, grad_atol: Optional[float]=1e-4):
        # init params for algorithm
        x_k = np.array(x_0).astype(np.float64)
        barrier_func = lambda x: self.g_barrier(x)
        barrier_tol_convergence = SimpleConvergenceTest(-1., atol=penalty_atol)
        converged = False
        beta_inv = beta**-1

        for k in range(maxiters1):
            # check for convergence 
            if barrier_tol_convergence:
                barrier_tol_convergence.update(0.0) 
                barrier_tol_convergence.update(mu*barrier_func(x_k))
                if barrier_tol_convergence.converged():
                    converged = True

            if converged: break

            objective_func_k = lambda x: self.f(x) + mu * barrier_func(x)
            self.solver.f = objective_func_k
            res_k = self.solver.solve(x_0=x_k, maxiters=maxiters2, grad_atol=grad_atol)
            x_k = res_k.x
            mu *= beta_inv

        return build_result_object(self.f, x_k, k, converged)
    


if __name__ == "__main__":
    import numpy as np
    f=lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
    # g=lambda x: np.array([
    #     (x[0] - 1)**3 - x[1] + 1,
    #     x.sum() - 2
    # ])
    g = lambda x: np.array(
        [
             np.square(x).sum() - 2.5
        ]
    )
    h=None
    x_start=np.array([-.5, -.5])
    x_min=np.ones(2)
    solver = BarrierQuasiNewtonMethod(f, g, g_barrier="inverse", line_search_method="wolfe", inverse_hessian_method="bfgs")
    res = solver.solve(x_0=x_start, penalty_atol=1e-4, grad_atol=1e-8)
    print(res)