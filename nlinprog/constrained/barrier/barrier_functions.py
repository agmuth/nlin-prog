import numpy as np
from abc import ABC, abstractclassmethod
from types import MappingProxyType

ZERO_MINUS = -1e-16

class BarrierFunction(ABC):
    def __init__(self):
        eps = 1e-8
        self.ZERO_MINUS = -1*eps
        self.ZERO_PLUS = eps

    @abstractclassmethod
    def __call__(self):
        pass


class InverseBarrierFunction(BarrierFunction):
    def __call__(self, g: callable) -> callable:
        def barrier_func(x: np.ndarray) -> float:
            g_of_x = g(x)
            g_of_x[g_of_x >= self.ZERO_MINUS] = self.ZERO_MINUS
            return -1*np.sum(g_of_x**-1)
        return barrier_func
    
class LogarithmicBarrierFunction(BarrierFunction):
    def __call__(self, g: callable) -> callable:
        def barrier_func(x: np.ndarray) -> float:
            g_of_x = g(x)
            g_of_x[g_of_x < -1] = -1
            return -1*np.sum(np.log(-1*g_of_x))
        return barrier_func
    
BARRIER_FUNCTIONS_MAPPING = MappingProxyType(
    {
        "inverse": InverseBarrierFunction,
        "logarithmic": LogarithmicBarrierFunction
    }
)