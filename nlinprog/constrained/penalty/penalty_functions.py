from abc import ABC, abstractclassmethod
from types import MappingProxyType

import numpy as np


class PenaltyFunction(ABC):
    @abstractclassmethod
    def __call__(self):
        pass


class InequalityPenaltyFunction(PenaltyFunction):
    @abstractclassmethod
    def __call__(self, g: callable) -> callable:
        pass


class EqualityPenaltyFunction(PenaltyFunction):
    @abstractclassmethod
    def __call__(self, h: callable) -> callable:
        pass


class ReluSquaredPenalty(InequalityPenaltyFunction):
    def __call__(self, g: callable) -> callable:
        """Construct + return the relu squared function wrt to `g`"""

        def penalty_func(x: np.ndarray) -> float:
            g_plus_of_x = g(x)
            g_plus_of_x[g_plus_of_x < 0] = 0
            return 0.5 * np.square(g_plus_of_x).sum()

        return penalty_func


class SquaredPenalty(EqualityPenaltyFunction):
    """Construct + return the squared penalty function wrt to `h`"""

    def __call__(self, h: callable) -> callable:
        def penalty_func(x: np.ndarray) -> float:
            h_of_x = h(x)
            return np.sum(np.square(h_of_x))

        return penalty_func


def zero_func(x: np.ndarray) -> float:
    """Null penalty function"""
    return 0.0


INEQUALITY_PENALTY_FUNCTIONS_MAPPING = MappingProxyType(
    {"relu-squared": ReluSquaredPenalty}
)
EQUALITY_PENALTY_FUNCTIONS_MAPPING = MappingProxyType({"squared": SquaredPenalty})
