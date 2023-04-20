import numpy as np
from nlinprog.numerical_differentiation import central_difference
from nlinprog.utils import  SimpleConvergenceTest
from typing import Optional


def steepest_descent(f: callable, x0: np.ndarray, line_search: callable, atol:Optional[float]=None, rtol:Optional[float]=None, maxitters: Optional[int]=1000) -> np.ndarray:
    grad = central_difference(f)
    x_k = np.array(x0)
    atol_convergence = SimpleConvergenceTest(np.linalg.norm(grad(x0)), atol=atol) if atol else None
    rtol_convergence = SimpleConvergenceTest(np.linalg.norm(f(x0)), atol=np.finfo(float).eps, rtol=rtol) if rtol else None # add atol as machine eps as a safeguard
    converged = False

    for k in range(maxitters):
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
        
        d_k = -1*grad(x_k)
        alpha_k = line_search(f, x_k, d_k)
        u_k = alpha_k * d_k
        x_k += u_k

    return x_k


