import numpy as np
from typing import Optional, Union

from nlinprog.line_search import LineSearch
from nlinprog.utils import ConvergenceTest, build_result_object
from nlinprog.unconstrained.quasi_newton import QuasiNewtonMethod, ApproxInverseHessian
from nlinprog.constrained.barrier.barrier_functions import BARRIER_FUNCTIONS_MAPPING, BarrierFunction


class BarrierQuasiNewtonMethod():
    """Quasi Newton method for constrained nonlinear optimization using barrier methods."""
    def __init__(
        self, f: callable,
        g: Optional[callable]=None, 
        g_barrier: Optional[Union[BarrierFunction, str]]="inverse",
        line_search_method: Union[LineSearch, str]="wolfe",
        inverse_hessian_method: Union[ApproxInverseHessian, str]="exact"
    ):
        """init

        Parameters
        ----------
        f : callable
            Function to minimize
        g : Optional[callable], optional
            Function describing constraints of for m `g(x) <= 0`, by default None
        g_barrier : Optional[Union[BarrierFunction, str]], optional
            Barrier function to apply to `g`, by default "inverse"
        line_search_method : Union[LineSearch, str], optional
            Line search method to use in each step of algorithm, by default "wolfe"
        inverse_hessian_method : Union[ApproxInverseHessian, str], optional
            Method to calculate approximation of inverse hessian at each step of algorithm, by default "exact"
        """
        self.f = f
        self.g = g
        self.g_barrier = (
            BARRIER_FUNCTIONS_MAPPING[g_barrier]()(g) 
            if isinstance(g_barrier, str) 
            else g_barrier(g)
        ) 
        self.solver = QuasiNewtonMethod(f=f, line_search_method=line_search_method, inverse_hessian_method=inverse_hessian_method)

    def solve(
        self, 
        x_0: np.ndarray,
        mu: float=2.0,
        beta: float=2.0,
        atol1: Optional[float]=1e-4,
        rtol1: Optional[float]=1e-4,
        atol2: Optional[float]=1e-8,
        rtol2: Optional[float]=1e-8,
        maxiters1: int=200,
        maxiters2: int=200,
        *args,
        **kwargs
    ):
        """Run Quasi Newton Method

        Parameters
        ----------
        x_0 : np.ndarray
            Starting point.
        mu : float, optional
            Coefficient applied to barrier function, by default 2.0
        beta : float, optional
            Multiple by which to decrease mu at each iteration, by default 2.0
        atol1 : Optional[float], optional
            Absolute tolerance for early termination of outer algorithm, applied to sequence of f(x), by default 1e-4
        rtol1 : Optional[float], optional
            Relative tolerance for early termination of outer algorithm, applied to sequence of f(x), by default 1e-4
        atol2 : Optional[float], optional
            Absolute tolerance for early termination of inner algorithm, applied to sequence of f(x), by default 1e-8
        rtol2 : Optional[float], optional
            Relative tolerance for early termination of inner algorithm, applied to sequence of f(x), by default 1e-8
        maxiters1 : int, optional
            Maximum number of iterations for outer algorithm to run, by default 200
        maxiters2 : int, optional
            Maximum number of iterations for inner algorithm to run, by default 200

        Returns
        -------
        NonLinProgResult
            Result object.
        """
        # init params for algorithm
        x_k = np.array(x_0).astype(np.float64)
        beta_inv = beta**-1
        barrier_func = lambda x: self.g_barrier(x)
        
        convergence_test = ConvergenceTest(atol=atol1, rtol=rtol1)
        converged = False
        
        for k in range(maxiters1):
            # check for convergence 
            if np.all(self.g(x_k) <= 0):
                convergence_test.update(self.f(x_k))
                if convergence_test.converged:
                    converged = True
                    break

            objective_func_k = lambda x: self.f(x) + mu * barrier_func(x)
            self.solver.f = objective_func_k
            res_k = self.solver.solve(x_0=x_k, maxiters=maxiters2, atol=atol2, rtol=rtol2)
            x_k = res_k.x
            mu *= beta_inv

        return build_result_object(self.f, x_k, k, converged)
    
    
    
if __name__ == "__main__":
    f=lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
    g=lambda x: np.array([
        np.square(x).sum() - 2.5
    ])
    h=None
    x_start=np.array([-0.5, -0.5])
    x_min=np.ones(2)
    
    solver = BarrierQuasiNewtonMethod(f, g, line_search_method="wolfe", inverse_hessian_method="bfgs")
    res = solver.solve(x_0=x_start, atol1=1e-4, maxiters1=500)
    print(res)
