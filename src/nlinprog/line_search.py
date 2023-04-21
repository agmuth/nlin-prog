import numpy as np
from typing import Optional
from nlinprog.numerical_differentiation import central_difference


def sufficient_decrease_condition(
    phi_of_alpha: np.ndarray,
    phi_of_zero: np.ndarray,
    phi_prime_of_zero: np.ndarray,
    alpha: float,
    c1: float
) -> bool:
    return np.all(phi_of_alpha <= phi_of_zero + c1 * phi_prime_of_zero * alpha)


def curvature_condition(phi_prime_of_alpha, phi_prime_of_zero, c2) -> bool:
    return np.all(np.abs(phi_prime_of_alpha) <= c2*np.abs(phi_prime_of_zero))


def armijo_backtracking_line_search(f: callable, x_k: np.ndarray, d_k: np.ndarray, alpha: Optional[float]=1.0, epsilon: Optional[float]=0.2, eta: Optional[float]=2.0, *args, **kwargs) -> float:
    f_grad = central_difference(f)
    phi = lambda alpha: f(x_k + alpha*d_k)
    phi_prime = lambda alpha: f_grad(x_k + alpha*d_k).T @ d_k

    zero = 0.0
    phi_of_zero = phi(zero)
    phi_prime_of_zero = phi_prime(zero)

    _armijo_condition = lambda alpha: sufficient_decrease_condition(phi(alpha), phi_of_zero, phi_prime_of_zero, alpha, epsilon*eta)

    while _armijo_condition(alpha): # alpha does not exceed bound -> increase until alpha violates bound and then take penultimate alpha
        alpha *= eta

    eta_inv = 1/eta
    while not _armijo_condition(alpha): # alpha exceeds bound -> decrese alpha until it meets bound
        alpha *= eta_inv

    return alpha


def wolfe_zoom_line_search(f: callable, x_k: np.ndarray, d_k: np.ndarray, alpha_max: Optional[float]=1.0, c1: Optional[float]=0.2, c2: Optional[float]=2.0, *args, **kwargs) -> float:
    # kwargs.setdefault("alpha_max", 1.0)
    
    f_grad = central_difference(f)
    phi = lambda alpha: f(x_k + alpha*d_k)
    phi_prime = lambda alpha: f_grad(x_k + alpha*d_k).T @ d_k

    zero = 0.0
    phi_of_zero = phi(zero)
    phi_prime_of_zero = phi_prime(zero)

    _sufficient_decrease_condition = lambda alpha: sufficient_decrease_condition(phi(alpha), phi_of_zero, phi_prime_of_zero, alpha, c1)
    _curvature_condition_negative_c2 = lambda alpha: curvature_condition(phi(alpha), phi_prime_of_zero, alpha, -c2)

    
    alpha_i = alpha_max
    i = 0
   
    while True:
        i += 1
        
        alpha_i_minus_one = alpha_i
        phi_of_alpha_i_minus_one = phi(alpha_i_minus_one) 
        alpha_i = phi_of_alpha_quadratic_interpolation(alpha_i_minus_one, phi_of_alpha_i_minus_one, phi_of_zero, phi_prime_of_zero)
        
        if i == 0:
            alpha_i = max(alpha_max, alpha_i)

        phi_of_alpha_i = phi(alpha_i)
        if _sufficient_decrease_condition(alpha_i) or (phi_of_alpha_i > phi_of_alpha_i_minus_one and i > 1):
            return zoom(phi, phi_prime, alpha_i_minus_one, alpha_i)
        
        phi_prime_of_alpha_i = phi_prime(alpha_i)
        if np.all(np.abs(phi_prime_of_alpha_i) <= -c2*phi_prime_of_zero):
            return alpha_i
        
        if phi_of_alpha_i >= 0:
            return zoom(phi, phi_prime, alpha_i, alpha_i_minus_one)


def zoom(phi: callable, phi_prime: callable, alpha_high: float, alpha_low: float, epsilon: Optional[float]=0.2) -> float:
    alpha_high = alpha_high
    alpha_low = alpha_low

    zero = 0.0
    phi_of_zero = phi(zero)
    phi_prime_of_zero = phi_prime(zero)

    armijo_condition = lambda alpha: np.all(phi(alpha) <= phi_of_zero + epsilon * phi_prime_of_zero * alpha)

    while True:
    #TODO: handle early termination
        alpha = 0.5*(alpha_high + alpha_low) # interpolate via bisection #TODO: circle back and make smarter choice
        phi_of_alpha = phi(alpha)
        if not armijo_condition(alpha) or phi_of_alpha > phi(alpha_low):
            alpha_high = alpha
        else:
            phi_prime_of_alpha = phi_prime(alpha)
            if np.all(np.abs(phi_prime_of_alpha) <= -(1-epsilon)*phi_prime_of_zero):
                return alpha
            if phi_prime_of_alpha*(alpha_high - alpha_low):
                alpha_high = alpha_low
            alpha_low = alpha



def phi_of_alpha_quadratic_interpolation(alpha_i: float, phi_of_alpha_i: float, phi_of_zero: float, phi_prime_of_zero: float) -> float:
    # equ. 3.58 numerical optimization
    alpha_i_plus_one = -0.5*(phi_prime_of_zero*alpha_i**2)/(phi_of_alpha_i - phi_of_zero - phi_prime_of_zero*alpha_i)
    return alpha_i_plus_one


def phi_of_alpha_cubic_interpolation1(alpha_i_minus_one: float, alpha_i: float, phi_of_alpha_i: float, phi_of_alpha_i_minus_one: float, phi_of_zero: float, phi_prime_of_zero: float) -> float:
    # equ. 3.58 1/2 numerical optimization
    a_and_b = (
        (alpha_i**2 * alpha_i_minus_one**2 * (alpha_i - alpha_i_minus_one))**-1 
        * np.array([[alpha_i_minus_one**2, -alpha_i**2], [-alpha_i_minus_one**3, alpha_i**3]]) 
        @ np.array([[phi_of_alpha_i_minus_one - phi_of_zero - phi_prime_of_zero*alpha_i_minus_one], [phi_of_alpha_i - phi_of_zero - phi_prime_of_zero*alpha_i]])
    ).flatten()
    a, b = a_and_b[0], a_and_b[1]
    alpha_i_plus_one = (3*a)**-1 * (-b + np.sqrt(b**2 - 3*a*phi_prime_of_zero))
    return alpha_i_plus_one


def phi_of_alpha_cubic_interpolation2(alpha_i_minus_one: float, alpha_i: float, phi_of_alpha_i: float, phi_of_alpha_i_minus_one: float, phi_prime_of_alpha_i: float, phi_prime_of_alpha_i_minus_one: float) -> float:
    # equ. 3.59 numerical optimization
    d1 = phi_prime_of_alpha_i_minus_one + phi_prime_of_alpha_i - 3*(phi_of_alpha_i_minus_one - phi_of_alpha_i)/(alpha_i_minus_one - alpha_i)
    d2 = np.sign(alpha_i - alpha_i_minus_one) * np.sqrt(d1**2 - phi_prime_of_alpha_i_minus_one*phi_prime_of_alpha_i)
    alpha_i_plus_one = alpha_i - (alpha_i - alpha_i_minus_one)*(phi_prime_of_alpha_i + d2 - d1)/(phi_prime_of_alpha_i - phi_prime_of_alpha_i_minus_one + 2*d2)
    return alpha_i_plus_one



def line_search_calculation_mapping(method: str) -> callable:
    method = method.lower()

    aliases_mapping = {
        "armijo" : {
            "aliases" : ["armijo"],
            "callable" : armijo_backtracking_line_search,
        },
         "wolfe" : {
            "aliases" : ["wolfe"],
            "callable" : wolfe_zoom_line_search,
        },
    }

    for k, v in aliases_mapping.items():
        if any(method == alias for alias in v["aliases"]):
            return v["callable"]
    
    raise ValueError(f"method {method} is not supported.")