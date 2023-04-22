import numpy as np

from nlinprog.numerical_differentiation import hessian
from typing import Optional


def calc_exact_inverse_hessian(f: callable, x_k: np.ndarray, *args, **kwargs) -> np.ndarray:
    H = hessian(f)(x_k)
    H = 0.5*(H + H.T) + 1e-4 * np.eye(x_k.shape[0])  # add jitter
    return np.linalg.inv(H)


def calc_identity_inverse_hessian(x_k: np.ndarray, *args, **kwargs) -> np.ndarray:
    return np.eye(x_k.shape[0])


def calc_dfp_inverse_hessian(H_k: np.ndarray, p_k: np.ndarray, q_k: np.ndarray, *args, **kwargs) -> np.ndarray:
    # H_k_plus_one = np.array(H_k)
    # H_k_plus_one += p_k @ p_k.T / (q_k @ q_k.T)
    # H_k_plus_one -= H_k @ q_k @ q_k.T @ H_k / (q_k.T @ H_k @ q_k)
    H_mult_q = H_k @ q_k
    H_k_plus_one = np.array(H_k)
    H_k_plus_one += np.divide(np.outer(p_k, p_k), (p_k @ q_k.T))
    H_k_plus_one -= np.divide(np.outer(H_mult_q,  H_mult_q), (q_k.T @ H_mult_q))
    return H_k_plus_one


def calc_bfgs_inverse_hessian(H_k: np.ndarray, p_k: np.ndarray, q_k: np.ndarray, *args, **kwargs) -> np.ndarray:
    # H_k_plus_one = np.array(H_k)
    # H_k_plus_one += (1 + q_k.T @ H_k @ q_k) / (q_k.T @ q_k) * (p_k @ p_k.T) / (p_k.T @ q_k)
    # H_k_plus_one -= (p_k @ q_k.T @ H_k + H_k @ q_k @ q_k.T) / (q_k.T @ p_k)
    # return H_k_plus_one

    q_dot_p_inv = (q_k.T @ p_k)**-1
    q_outer_p = np.outer(q_k, p_k)
    p_outer_q = q_outer_p.T
    p_outer_p = np.outer(p_k, p_k)

    H_k_plus_one = np.array(H_k)
    H_k_plus_one += p_outer_p * q_dot_p_inv
    H_k_plus_one += p_outer_q @ H_k @ q_outer_p * q_dot_p_inv**2
    H_k_plus_one -= (H_k @ q_outer_p + p_outer_q @ H_k) * q_dot_p_inv
    return H_k_plus_one
    

def calc_broyden_inverse_hessian(H_k: np.ndarray, p_k: np.ndarray, q_k: np.ndarray, phi: Optional[float]=0.5, *args, **kwargs) -> np.ndarray:
    H_k_plus_one = calc_dfp_inverse_hessian(H_k, p_k, q_k)
    v_k = np.sqrt(q_k.T @ H_k @ q_k) * (p_k / (p_k.T @ q_k) - H_k @ q_k / (q_k.T @ H_k @ q_k))
    H_k_plus_one += phi * np.outer(v_k, v_k)
    return H_k_plus_one


def inverse_hessian_calculation_mapping(method: str) -> callable:
    method = method.lower()

    aliases_mapping = {
        "bfgs" : {
            "aliases" : ["bfgs"],
            "callable" : calc_bfgs_inverse_hessian,
        },
         "dfp" : {
            "aliases" : ["dfp"],
            "callable" : calc_dfp_inverse_hessian,
        },
         "broyden" : {
            "aliases" : ["broyden"],
            "callable" : calc_broyden_inverse_hessian,
        },
         "newton" : {
            "aliases" : ["exact", "newton"],
            "callable" : calc_exact_inverse_hessian,
        },
         "identity" : {
            "aliases" : ["identity", "steepest"],
            "callable" : calc_identity_inverse_hessian,
        },
    }

    for k, v in aliases_mapping.items():
        if any(method == alias for alias in v["aliases"]):
            return v["callable"]
    
    raise ValueError(f"method {method} is not supported.")
    
    
