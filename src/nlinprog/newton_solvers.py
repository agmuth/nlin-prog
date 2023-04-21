import numpy as np

from nlinprog.line_search import line_search_calculation_mapping
from nlinprog.numerical_differentiation import central_difference, hessian
from nlinprog.utils import  SimpleConvergenceTest, build_return_object
from typing import Optional


def calc_exact_inverse_hessian(f: callable, x_k: np.ndarray, *args, **kwargs) -> np.ndarray:
    H = hessian(f)(x_k)
    H = 0.5*(H + H.T) #TODO: add jitter
    return np.linalg.inv(H)


def calc_identity_inverse_hessian(x_k: np.ndarray, *args, **kwargs) -> np.ndarray:
    return np.eye(x_k.shape[0])


def calc_dfp_inverse_hessian(H_k: np.ndarray, p_k: np.ndarray, q_k: np.ndarray, *args, **kwargs) -> np.ndarray:
    H_k_plus_one = np.array(H_k)
    H_k_plus_one += p_k @ p_k.T / (q_k @ q_k.T)
    H_k_plus_one -= H_k @ q_k @ q_k.T @ H_k / (q_k.T @ H_k @ q_k)
    return H_k_plus_one


def calc_bfgs_inverse_hessian(H_k: np.ndarray, p_k: np.ndarray, q_k: np.ndarray, *args, **kwargs) -> np.ndarray:
    H_k_plus_one = np.array(H_k)
    H_k_plus_one += (1 + q_k.T @ H_k @ q_k) / (q_k.T @ q_k) * (p_k @ p_k.T) / (p_k.T @ q_k)
    H_k_plus_one -= (p_k @ q_k.T @ H_k + H_k @ q_k @ q_k.T) / (q_k.T @ p_k)
    return H_k_plus_one


def calc_broyden_inverse_hessian(H_k: np.ndarray, p_k: np.ndarray, q_k: np.ndarray, phi: Optional[float]=0.5, *args, **kwargs) -> np.ndarray:
    H_k_plus_one = calc_dfp_inverse_hessian(H_k, p_k, q_k)
    v_k = np.sqrt(q_k.t @ H_k @ q_k) * (p_k / (p_k.T @ q_k) - H_k @ q_k / (q_k.t @ H_k @ q_k))
    H_k_plus_one += phi * v_k @ v_k.T
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
         "newton" : {
            "aliases" : ["exact", "netwon"],
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
    
    


def newtons_method(
    f: callable,
    x0: np.ndarray,
    line_search_method: str,
    inverse_hessian_calc_method: str,
    H_0:Optional[np.ndarray]=None,
    atol:Optional[float]=None,
    rtol:Optional[float]=None, 
    maxiters: Optional[int]=1000,
    *args,
    **kwargs
) -> np.ndarray:

    # get callables
    line_search = line_search_calculation_mapping(line_search_method)
    calc_inverse_hessian = inverse_hessian_calculation_mapping(inverse_hessian_calc_method)

    # init params for algorithm
    grad = central_difference(f)
    x_k = np.array(x0)
    f_k = f(x_k)
    g_k = grad(x_k)
    H_k = np.array(H_0) if H_0 else np.eye(x0.shape[0])


    # set up convergece tracking
    grad_atol_convergence = SimpleConvergenceTest(np.linalg.norm(g_k), atol=atol) if atol else None
    func_rtol_convergence = SimpleConvergenceTest(np.linalg.norm(f_k), atol=np.finfo(float).eps, rtol=rtol) if rtol else None # add atol as machine eps as a safeguard
    converged = False

    k = 0
    while k < maxiters:
        k += 1
        # check for convergence 
        if grad_atol_convergence:
            grad_atol_convergence.update(0.0) # dont love
            grad_atol_convergence.update(np.linalg.norm(g_k)) #TODO: remove double calc
            if grad_atol_convergence.converged():
                converged = True

        if func_rtol_convergence:
            func_rtol_convergence.update(f_k)
            if func_rtol_convergence.converged():
                converged = True
        
        if converged: break

        # (modified/quasi) newton's method
        d_k = -1 * H_k @ g_k  # search direction for x_k
        alpha_k = line_search(f=f, x_k=x_k, d_k=d_k, *args, **kwargs)  # minimizer of f(x_k + alpha*d_k)
        p_k = alpha_k * d_k # update to x_k
        x_k_plue_one = x_k + p_k
        g_k_plus_one = grad(x_k_plue_one)
        q_k = g_k_plus_one - g_k
        H_k = calc_inverse_hessian(f=f, x_k=x_k, H_k=H_k, p_k=p_k, q_k=q_k, *args, **kwargs) # really h_k_plus_one but not used in this iter
        x_k, g_k = x_k_plue_one, g_k_plus_one # update for next iter

    return build_return_object(f, x_k, k, converged)


if __name__ == "__main__":
    f=lambda x: 0.26*np.sum(x*x) - 0.48*np.prod(x)
    x0=np.array([2., -8.])
    alpha_max = 2.0
    res = newtons_method(
        f, 
        x0,
        line_search_method="wolfe", 
        inverse_hessian_calc_method="steepest",
        atol=1e-2,
        alpha_max=2.0
    )

    res