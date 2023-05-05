import numpy as np
from abc import ABC, abstractclassmethod
from nlinprog.numerical_differentiation import hessian
from types import MappingProxyType
from typing import Optional, Union

from nlinprog.numerical_differentiation import central_difference as jacobian
from nlinprog.unconstrained.line_search import LINE_SEARCH_MAPPING, LineSearch
# from nlinprog.inverse_hessian import APPROX_INVERSE_HESSIAN_MAPPING, ApproxInverseHessian
from nlinprog.utils import SimpleConvergenceTest, build_result_object


class ApproxInverseHessian(ABC):
    @abstractclassmethod
    def __call__(self, f, x_k, H_k, p_k, q_k, *args, **kwargs):
        pass


class NumericalInverseHessian(ApproxInverseHessian):
    def __call__(self, f: callable, x_k: np.ndarray, *args, **kwargs) -> np.ndarray:
        H_k_plus_one = hessian(f)(x_k)
        H_k_plus_one = 0.5*(H_k_plus_one + H_k_plus_one.T) + 1e-4 * np.eye(x_k.shape[0])  # add jitter
        return np.linalg.inv(H_k_plus_one)


class IdentityInverseHessian(ApproxInverseHessian):
    def __call__(self, x_k: np.ndarray, *args, **kwargs) -> np.ndarray:
        return np.eye(x_k.shape[0])


class DFPInverseHessian(ApproxInverseHessian):
    def __call__(self, H_k: np.ndarray, p_k: np.ndarray, q_k: np.ndarray, *args, **kwargs) -> np.ndarray:
        H_mult_q = H_k @ q_k
        H_k_plus_one = np.array(H_k)
        H_k_plus_one += np.divide(np.outer(p_k, p_k), (p_k @ q_k.T))
        H_k_plus_one -= np.divide(np.outer(H_mult_q,  H_mult_q), (q_k.T @ H_mult_q))
        return H_k_plus_one


class BFGSInverseHessian(ApproxInverseHessian):
    def __call__(self, H_k: np.ndarray, p_k: np.ndarray, q_k: np.ndarray, *args, **kwargs) -> np.ndarray:
        q_dot_p_inv = (q_k.T @ p_k)**-1
        q_outer_p = np.outer(q_k, p_k)
        p_outer_q = q_outer_p.T
        p_outer_p = np.outer(p_k, p_k)
        H_k_plus_one = np.array(H_k)
        H_k_plus_one += p_outer_p * q_dot_p_inv
        H_k_plus_one += p_outer_q @ H_k @ q_outer_p * q_dot_p_inv**2
        H_k_plus_one -= (H_k @ q_outer_p + p_outer_q @ H_k) * q_dot_p_inv
        return H_k_plus_one
    

class BroydenInverseHessian(ApproxInverseHessian):
    def __init__(self, phi: float=0.5):
        self.phi = phi
        self.calc_dfp_inverse_hessian = DFPInverseHessian()

    def __call__(self, H_k: np.ndarray, p_k: np.ndarray, q_k: np.ndarray, *args, **kwargs) -> np.ndarray:
        H_k_plus_one = self.calc_dfp_inverse_hessian(H_k, p_k, q_k)
        v_k = np.sqrt(q_k.T @ H_k @ q_k) * (p_k / (p_k.T @ q_k) - H_k @ q_k / (q_k.T @ H_k @ q_k))
        H_k_plus_one += self.phi * np.outer(v_k, v_k)
        return H_k_plus_one



APPROX_INVERSE_HESSIAN_MAPPING = MappingProxyType(
    {
        "bfgs": BFGSInverseHessian,
        "dfp": DFPInverseHessian,
        "broyden": BroydenInverseHessian,
        "exact": NumericalInverseHessian,
        "identity": IdentityInverseHessian 
    }
)


class QuasiNewtonMethod():
    def __init__(self, f: callable, line_search_method: Union[LineSearch, str]="wolfe", inverse_hessian_method: Union[ApproxInverseHessian, str]="exact"):
        self.f = f
        self.line_search = LINE_SEARCH_MAPPING[line_search_method]() if isinstance(line_search_method, str) else line_search_method
        self.calc_inverse_hessian = APPROX_INVERSE_HESSIAN_MAPPING[inverse_hessian_method]() if isinstance(inverse_hessian_method, str) else inverse_hessian_method
        

    def solve(self, x_0: np.ndarray, maxiters: int=200, grad_atol: Optional[float]=1e-4):
        # init params for algorithm
        x_k = np.array(x_0).astype(np.float64)
        grad = jacobian(self.f)
        g_k = grad(x_k)
        H_k = np.eye(x_0.shape[0])  # inverse hessian

        # set up convergece tracking
        grad_atol_convergence = SimpleConvergenceTest(np.linalg.norm(g_k), atol=grad_atol) if grad_atol else None
        converged = False

        for k in range(maxiters):
            # check for convergence 
            if grad_atol_convergence:
                grad_atol_convergence.update(0.0) # dont love
                grad_atol_convergence.update(np.linalg.norm(g_k))
                if grad_atol_convergence.converged():
                    converged = True
            
            if converged: break

            # (modified/quasi) newton's method/step
            d_k = -1 * H_k @ g_k  # search direction for x_k
            alpha_k = self.line_search(f=self.f, x_k=x_k, d_k=d_k)  # minimizer of f(x_k + alpha*d_k)
            p_k = alpha_k*d_k # update to x_k
            x_k_plus_one = x_k + p_k
            g_k_plus_one = grad(x_k_plus_one)
            q_k = g_k_plus_one - g_k
            H_k = self.calc_inverse_hessian(f=self.f, x_k=x_k, H_k=H_k, p_k=p_k, q_k=q_k) # really h_k_plus_one but not used in this iter
            x_k, g_k = x_k_plus_one, g_k_plus_one # update for next iter

        return build_result_object(self.f, x_k, k, converged)

