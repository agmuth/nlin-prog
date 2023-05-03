import numpy as np

def relu_squared_penalty(g: callable) ->  callable:
    def penalty_func(x: np.ndarray) -> float:
        g_plus_of_x = g(x)
        g_plus_of_x[g_plus_of_x < 0] = 0
        return 0.5*np.square(g_plus_of_x).sum()
    return penalty_func

def squared_penalty(h: callable) -> callable:
    def penalty_func(x: np.ndarray) -> float:
        h_of_x = h(x)
        return np.sum(np.square(h_of_x))
    return penalty_func

def zero_penalty(f: callable) -> callable:
    def penalty_func(x: np.ndarray) -> float:
        return 0.0
    return penalty_func

def zero_func(x: np.ndarray) -> float:
    return 0.0