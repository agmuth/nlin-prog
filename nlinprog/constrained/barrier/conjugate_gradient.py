from typing import Optional, Union

from nlinprog.unconstrained.line_search import LineSearch
from nlinprog.unconstrained.conjugate_gradient import ConjugateGradientMethod, CalcConjugateGradientDirection
from nlinprog.constrained.barrier.quasi_newton import BarrierQuasiNewtonMethod
from nlinprog.constrained.barrier.barrier_functions import BARRIER_FUNCTIONS_MAPPING, BarrierFunction


class BarrierConjugateGradientMethod(BarrierQuasiNewtonMethod):
    def __init__(self, f: callable, g: Optional[callable]=None, g_barrier: Optional[Union[BarrierFunction, str]]="inverse", line_search_method: Union[LineSearch, str]="wolfe", conjugate_gradient_direction_method: Union[CalcConjugateGradientDirection, str]="fletcher-reeves"):
        self.f = f
        self.g = g
        self.g_barrier = BARRIER_FUNCTIONS_MAPPING[g_barrier]()(g) if isinstance(g_barrier, str) else g_barrier(g) 
        self.solver = ConjugateGradientMethod(f=f, line_search_method=line_search_method, conjugate_gradient_direction_method=conjugate_gradient_direction_method)



