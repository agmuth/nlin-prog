import numpy as np
from nlinprog.numerical_differentiation import central_difference
from typing import Optional


def steepest_descent(f: callable, x0: np.ndarray, line_search: callable, atol:Optional[float]=None, rtol:Optional[float]=None, maxitters: Optional[int]=100) -> np.ndarray:
    grad = central_difference(f)
    x_k = np.array(x0)
    convergence = False

    for k in range(maxitters):
        d_k = -1*grad(x_k)
        alpha_k = line_search(f, x_k, d_k)
        u_k = alpha_k * d_k
        x_k += u_k

        if atol and np.linalg.norm(d_k) < atol:
            convergence = True
        # if rtol and np.linalg.norm(u_k)/np.linalg.norm(x_k - u_k) < rtol:
        #     convergence = True

        if convergence:
            break

    return x_k


