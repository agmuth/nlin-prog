import numpy as np
from abc import ABC, abstractclassmethod
from types import MappingProxyType

class CalcConjugateGradientDirection(ABC):
    @abstractclassmethod
    def __call__(self):
        pass


class FletcherReevesConjugateGradientDirection(CalcConjugateGradientDirection):
    def __call__(self, d_k: np.ndarray, g_k: np.ndarray, g_k_plus_one: np.ndarray) -> np.ndarray:
        beta_k = g_k_plus_one.T @ g_k_plus_one / (g_k.T @ g_k)
        return -1*g_k_plus_one + beta_k*d_k
    

class PolakRibiereConjugateGradientDirection(CalcConjugateGradientDirection):
    def __call__(self, d_k: np.ndarray, g_k: np.ndarray, g_k_plus_one: np.ndarray) -> np.ndarray:
        beta_k = (g_k_plus_one - g_k).T @ g_k_plus_one / (g_k.T @ g_k)
        return -1*g_k_plus_one + beta_k*d_k


CALC_CONJUGATE_GRADIENT_DIRECTION_MAPPING = MappingProxyType(
    {
        "fletcher-reeves": FletcherReevesConjugateGradientDirection,
        "polak-ribier": PolakRibiereConjugateGradientDirection
    }
)
