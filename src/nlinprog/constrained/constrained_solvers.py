import numpy as np
from typing import Optional, Union
from abc import ABC, abstractclassmethod

from nlinprog.numerical_differentiation import central_difference as jacobian
from nlinprog.line_search import LINE_SEARCH_MAPPING, LineSearch
from nlinprog.inverse_hessian import APPROX_INVERSE_HESSIAN_MAPPING, ApproxInverseHessian
from nlinprog.conjugate_gradient_direction import CALC_CONJUGATE_GRADIENT_DIRECTION_MAPPING, CalcConjugateGradientDirection
from nlinprog.utils import SimpleConvergenceTest, build_result_object
from nlinprog.unconstrained.unconstrained_solvers import QuasiNewtonMethod, ConjugateGradientMethod
from nlinprog.penalty_functions import INEQUALITY_PENALTY_FUNCTIONS_MAPPING, EQUALITY_PENALTY_FUNCTIONS_MAPPING, InequalityPenaltyFunction, EqualityPenaltyFunction, zero_func



class ConstrainedSolver(ABC):  
    @abstractclassmethod
    def solve(self, maxiters: int=200, grad_atol: Optional[float]=1e-4, func_rtol: Optional[float]=1e-3):
        pass


class PenalizedQuasiNewtonMethod(ConstrainedSolver):
    def __init__(self, f: callable, g: Optional[callable]=None, h: Optional[callable]=None, g_penalty: Optional[Union[InequalityPenaltyFunction, str]]="relu-squared", h_penalty: Optional[Union[EqualityPenaltyFunction, str]]="squared", line_search_method: Union[LineSearch, str]="wolfe", inverse_hessian_method: Union[ApproxInverseHessian, str]="exact"):
        self.f = f
        self.g = g
        self.h = h
        self.g_penalized = zero_func if not g else INEQUALITY_PENALTY_FUNCTIONS_MAPPING[g_penalty]()(g) if isinstance(g_penalty, str) else g_penalty(g) 
        self.h_penalized = zero_func if not h else INEQUALITY_PENALTY_FUNCTIONS_MAPPING[h_penalty]()(h) if isinstance(h_penalty, str) else h_penalty(h) 
        self.solver = QuasiNewtonMethod(f=f, line_search_method=line_search_method, inverse_hessian_method=inverse_hessian_method)

    def solve(self, x_0: np.ndarray, mu: float=2.0, beta: float=2.0, penalty_atol: Optional[float]=1e-4, maxiters1: int=200, maxiters2: int=200, grad_atol: Optional[float]=1e-4):
        # init params for algorithm
        x_k = np.array(x_0).astype(np.float64)
        penalty_func = lambda x: self.g_penalized(x) + self.h_penalized(x)
        penalty_tol_convergence = SimpleConvergenceTest(-1., atol=penalty_atol)
        converged = False

        for k in range(maxiters1):
            # check for convergence 
            if penalty_tol_convergence:
                penalty_tol_convergence.update(0.0) 
                penalty_tol_convergence.update(mu*penalty_func(x_k))
                if penalty_tol_convergence.converged():
                    converged = True

            if converged: break

            objective_func_k = lambda x: self.f(x) + mu * penalty_func(x)
            self.solver.f = objective_func_k
            res_k = self.solver.solve(x_0=x_k, maxiters=maxiters2, grad_atol=grad_atol)
            x_k = res_k.x

        return build_result_object(self.f, x_k, k, converged)
    

class PenalizedConjugateGradientMethod(PenalizedQuasiNewtonMethod):
    def __init__(self, f: callable, g: Optional[callable]=None, h: Optional[callable]=None, g_penalty: Optional[Union[InequalityPenaltyFunction, str]]="relu-squared", h_penalty: Optional[Union[EqualityPenaltyFunction, str]]="squared", line_search_method: Union[LineSearch, str]="wolfe", conjugate_gradient_direction_method: Union[CalcConjugateGradientDirection, str]="fletcher-reeves"):
        self.f = f
        self.g = g
        self.h = h
        self.g_penalized = zero_func if not g else INEQUALITY_PENALTY_FUNCTIONS_MAPPING[g_penalty]()(g) if isinstance(g_penalty, str) else g_penalty(g) 
        self.h_penalized = zero_func if not h else INEQUALITY_PENALTY_FUNCTIONS_MAPPING[h_penalty]()(h) if isinstance(h_penalty, str) else h_penalty(h) 
        self.solver = ConjugateGradientMethod(f=f, line_search_method=line_search_method, conjugate_gradient_direction_method=conjugate_gradient_direction_method)


if __name__ == "__main__":
    # rosenbrock constrained with cubic and a line
    f=lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
    g=lambda x: np.array([
        (x[0] - 1)**3 - x[1] + 1,
        x.sum() - 2
    ])
    x_start=np.array([1.0, -0.5])

    solver = PenalizedConjugateGradientMethod(f=f, g=g)
    res = solver.solve(x_0=x_start)

    print(res)

