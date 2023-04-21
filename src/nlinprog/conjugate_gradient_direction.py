import numpy as np


def calc_fletcher_reeves_conjugate_gradient_direction(d_k: np.ndarray, g_k: np.ndarray, g_k_plus_one: np.ndarray):
    beta_k = g_k_plus_one.T @ g_k_plus_one / (g_k.T @ g_k)
    return -1*g_k_plus_one + beta_k*d_k


def calc_polak_ribiere_conjugate_gradient_direction(d_k: np.ndarray, g_k: np.ndarray, g_k_plus_one: np.ndarray):
    beta_k = (g_k_plus_one - g_k).T @ g_k_plus_one / (g_k.T @ g_k)
    return -1*g_k_plus_one + beta_k*d_k


def conjugate_gradient_direction_calculation_mapping(method: str) -> callable:
    method = method.lower()

    aliases_mapping = {
        "fletcher-reeves" : {
            "aliases" : ["fr"],
            "callable" : calc_fletcher_reeves_conjugate_gradient_direction,
        },
         "polak-ribiere" : {
            "aliases" : ["pr"],
            "callable" : calc_fletcher_reeves_conjugate_gradient_direction,
        },
    }

    for k, v in aliases_mapping.items():
        if any(method == alias for alias in v["aliases"]):
            return v["callable"]
    
    raise ValueError(f"method {method} is not supported.")

