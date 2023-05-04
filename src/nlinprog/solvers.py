import numpy as np
from typing import Optional

from nlinprog.numerical_differentiation import central_difference
from nlinprog.line_search import line_search_calculation_mapping
from nlinprog.inverse_hessian import inverse_hessian_calculation_mapping
from nlinprog.conjugate_gradient_direction import conjugate_gradient_direction_calculation_mapping
from nlinprog.utils import SimpleConvergenceTest, build_result_object
from nlinprog.penalty_functions import squared_penalty, relu_squared_penalty, zero_func

def penalized_newtwons_method(
        f: callable,
        x0: np.ndarray,
        g: Optional[callable]=zero_func,
        h: Optional[callable]=zero_func,
        mu0: Optional[float]=2.0,
        beta: Optional[float]=2.0,
        line_search_method:Optional[str]="wolfe",
        inverse_hessian_method:Optional[str]="bfgs",
        H_0:Optional[np.ndarray]=None,
        tol_penalty:Optional[float]=1e-4,
        atol_sub_problem:Optional[float]=None,
        rtol_sub_problem:Optional[float]=None,
        maxiters: Optional[int]=200,
        *args,
        **kwargs
    ) -> dict:

    g_penalty_func = relu_squared_penalty(g)
    h_penalty_func = squared_penalty(h)
    penalty_func = lambda x: g_penalty_func(x) + h_penalty_func(x)

    x_k = np.array(x0)
    mu_k = mu0

    # set up convergece tracking
    penalty_tol_convergence = SimpleConvergenceTest(np.linalg.norm(g_k), atol=tol_penalty)
    converged = False

    for k in range(maxiters):
        objective_func_k = lambda x: f(x) + mu_k * penalty_func(x)
        # check for convergence 
        if penalty_tol_convergence:
            penalty_tol_convergence.update(0.0) 
            penalty_tol_convergence.update(mu_k*penalty_func(x_k))
            if penalty_tol_convergence.converged():
                converged = True

        if converged: break

        res = newtons_method(
            f=objective_func_k,
            x0=x_k,
            line_search_method=line_search_method,
            inverse_hessian_method=inverse_hessian_method,
            H_0=H_0,
            tol_penalty=tol_penalty,
            atol_sub_problem=atol_sub_problem,
            rtol_sub_problem=rtol_sub_problem,
            maxiters=maxiters,
        )
        x_k = res["x"]
        mu_k *= beta

    return build_result_object(f, x_k, k, converged)


def newtons_method(
        f: callable,
        x0: np.ndarray,
        line_search_method: str,
        inverse_hessian_method: str,
        H_0:Optional[np.ndarray]=None,
        atol:Optional[float]=1e-4,
        rtol:Optional[float]=None, 
        maxiters: Optional[int]=200,
        *args,
        **kwargs
    ) -> dict:

    # get callables
    line_search = line_search_calculation_mapping(line_search_method)
    calc_inverse_hessian = inverse_hessian_calculation_mapping(inverse_hessian_method)

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

    for k in range(maxiters):
        # check for convergence 
        if grad_atol_convergence:
            grad_atol_convergence.update(0.0) # dont love
            grad_atol_convergence.update(np.linalg.norm(g_k))
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
        p_k = alpha_k*d_k # update to x_k
        x_k_plus_one = x_k + p_k
        g_k_plus_one = grad(x_k_plus_one)
        q_k = g_k_plus_one - g_k
        H_k = calc_inverse_hessian(f=f, x_k=x_k, H_k=H_k, p_k=p_k, q_k=q_k, *args, **kwargs) # really h_k_plus_one but not used in this iter
        x_k, g_k = x_k_plus_one, g_k_plus_one # update for next iter

    return build_result_object(f, x_k, k, converged)


def conjugate_gradient_method(
        f: callable,
        x0: np.ndarray,
        line_search_method: str,
        conjugate_gradient_direction_method: str,
        atol:Optional[float]=None,
        rtol:Optional[float]=None, 
        maxiters: Optional[int]=200,
        *args,
        **kwargs
    ) -> dict:

    # get callables
    line_search = line_search_calculation_mapping(line_search_method)
    calc_conjugate_direction = conjugate_gradient_direction_calculation_mapping(conjugate_gradient_direction_method)

    # init params for algorithm
    grad = central_difference(f)
    x_k = np.array(x0)
    f_k = f(x_k)
    g_k = grad(x_k)
    n = x_k.shape[0]

    # set up convergece tracking
    grad_atol_convergence = SimpleConvergenceTest(np.linalg.norm(g_k), atol=atol) if atol else None
    func_rtol_convergence = SimpleConvergenceTest(np.linalg.norm(f_k), atol=np.finfo(float).eps, rtol=rtol) if rtol else None # add atol as machine eps as a safeguard
    converged = False    

    for i in range(maxiters):
        g_k = grad(x_k).T
        d_k = -1*g_k
        for k in range(n):
            # check for convergence 
            if grad_atol_convergence:
                grad_atol_convergence.update(0.0) # dont love
                grad_atol_convergence.update(np.linalg.norm(g_k))
                if grad_atol_convergence.converged():
                    converged = True

            if func_rtol_convergence:
                func_rtol_convergence.update(f_k)
                if func_rtol_convergence.converged():
                    converged = True
            
            if converged: break

            alpha_k = line_search(f=f, x_k=x_k, d_k=d_k, *args, **kwargs)  # minimizer of f(x_k + alpha*d_k)
            x_k += alpha_k*d_k
            g_k_plus_one = grad(x_k).T
            d_k = calc_conjugate_direction(d_k, g_k, g_k_plus_one)
            g_k = g_k_plus_one
        
        if converged: break

    return build_result_object(f, x_k, i, converged)


if __name__ == "__main__":
    f=lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
    g=lambda x: np.array([
        (x[0] - 1)**3 - x[1] + 1,
        x.sum() - 2
    ])
    x_start=np.array([1.0, -0.5])

    res = penalized_newtwons_method(
        f=f,
        g=g,
        x0=x_start,
    )

    print(res)