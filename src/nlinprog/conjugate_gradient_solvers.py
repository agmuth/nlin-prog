import numpy as np
from typing import Optional

from nlinprog.numerical_differentiation import central_difference
from nlinprog.line_search import armijo_backtracking_line_search, wolfe_zoom_line_search
from nlinprog.utils import SimpleConvergenceTest


def calc_fletch_reeves_cg_beta(g_k: np.ndarray, g_k_plus_one: np.ndarray):
    beta_k = g_k_plus_one.T @ g_k_plus_one / (g_k.T @ g_k)
    return beta_k

def calc_polak_ribiere_cg_beta(g_k: np.ndarray, g_k_plus_one: np.ndarray):
    beta_k = (g_k_plus_one - g_k).T @ g_k_plus_one / (g_k.T @ g_k)
    return beta_k


def conjugate_gradient(f: callable, x0: np.ndarray, line_search: callable, cg_beta_calc: callable, atol:Optional[float]=None, rtol:Optional[float]=None, maxitters: Optional[int]=1000) -> np.ndarray:
    grad = central_difference(f)
    x_k = np.array(x0)
    n = x_k.shape[0]
    
    atol_convergence = SimpleConvergenceTest(np.linalg.norm(grad(x0)), atol=atol) if atol else None
    rtol_convergence = SimpleConvergenceTest(np.linalg.norm(f(x0)), atol=np.finfo(float).eps, rtol=rtol) if rtol else None # add atol as machine eps as a safeguard
    converged = False

    for i in range(maxitters//n):
        g_k = grad(x_k).T
        d_k = -1*g_k
        for k in range(n):
            if atol_convergence:
                atol_convergence.update(0.0) # dont love
                atol_convergence.update(np.linalg.norm(grad(x_k))) #TODO: remove double calc
                if atol_convergence.converged():
                    converged = True

            if rtol_convergence:
                rtol_convergence.update(f(x_k))
                if rtol_convergence.converged():
                    converged = True
            
            if converged: break

            alpha_k = line_search(f, x_k, d_k)
            x_k += alpha_k * d_k
            g_k_plus_one = grad(x_k).T
            beta_k = cg_beta_calc(g_k, g_k_plus_one)
            d_k *= beta_k
            d_k -= g_k_plus_one
            g_k = g_k_plus_one
        
        if converged: break

    return x_k


if __name__ == "__main__":
    f=lambda x: 0.26*np.sum(x*x) - 0.48*np.prod(x)
    x0=np.array([2., -8.])
    res = conjugate_gradient(
        f, 
        x0,
        armijo_backtracking_line_search, 
        # wolfe_zoom_line_search,
        # calc_fletch_reeves_cg_beta,
        calc_polak_ribiere_cg_beta,
        atol=1e-2
    )

    res