from typing import Optional, Union

from nlinprog.unconstrained.line_search import LineSearch
from nlinprog.unconstrained.conjugate_gradient import ConjugateGradientMethod, CalcConjugateGradientDirection
from nlinprog.constrained.penalty.quasi_newton import PenalizedQuasiNewtonMethod
from nlinprog.constrained.penalty.penalty_functions import INEQUALITY_PENALTY_FUNCTIONS_MAPPING, EQUALITY_PENALTY_FUNCTIONS_MAPPING, InequalityPenaltyFunction, EqualityPenaltyFunction, zero_func


class BarrrierConjugateGradientMethod(PenalizedQuasiNewtonMethod):
    def __init__(self, f: callable, g: Optional[callable]=None, h: Optional[callable]=None, g_penalty: Optional[Union[InequalityPenaltyFunction, str]]="relu-squared", h_penalty: Optional[Union[EqualityPenaltyFunction, str]]="squared", line_search_method: Union[LineSearch, str]="wolfe", conjugate_gradient_direction_method: Union[CalcConjugateGradientDirection, str]="fletcher-reeves"):
        self.f = f
        self.g = g
        self.h = h
        self.g_penalized = zero_func if not g else INEQUALITY_PENALTY_FUNCTIONS_MAPPING[g_penalty]()(g) if isinstance(g_penalty, str) else g_penalty(g) 
        self.h_penalized = zero_func if not h else EQUALITY_PENALTY_FUNCTIONS_MAPPING[h_penalty]()(h) if isinstance(h_penalty, str) else h_penalty(h) 
        self.solver = ConjugateGradientMethod(f=f, line_search_method=line_search_method, conjugate_gradient_direction_method=conjugate_gradient_direction_method)



