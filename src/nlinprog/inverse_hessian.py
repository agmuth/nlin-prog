import numpy as np
from abc import ABC, abstractclassmethod
from nlinprog.numerical_differentiation import hessian
from types import MappingProxyType


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
