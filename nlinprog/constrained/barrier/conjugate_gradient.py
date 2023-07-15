from typing import Optional, Union

from nlinprog.line_search import LineSearch
from nlinprog.unconstrained.conjugate_gradient import ConjugateGradientMethod, CalcConjugateGradientDirection
from nlinprog.constrained.barrier.quasi_newton import BarrierQuasiNewtonMethod
from nlinprog.constrained.barrier.barrier_functions import BARRIER_FUNCTIONS_MAPPING, BarrierFunction


class BarrierConjugateGradientMethod(BarrierQuasiNewtonMethod):
    """Conjugate Gradient method for constrained nonlinear optimization using barrier methods."""
    def __init__(
        self, f: callable, 
        g: Optional[callable]=None, 
        g_barrier: Optional[Union[BarrierFunction, str]]="inverse", 
        line_search_method: Union[LineSearch, str]="wolfe", 
        conjugate_gradient_direction_method: Union[CalcConjugateGradientDirection, str]="fletcher-reeves"
    ):
        """init

        Parameters
        ----------
        f : callable
            Function to minimize
        g : Optional[callable], optional
            Function describing constraints of for m `g(x) <= 0`, by default None
        g_barrier : Optional[Union[BarrierFunction, str]], optional
            Barrier function to apply to `g`, by default "inverse"
        line_search_method : Union[LineSearch, str], optional
            Line search method to use in each step of algorithm, by default "wolfe"
        conjugate_gradient_direction_method : Union[CalcConjugateGradientDirection, str], optional
            Method to calculate conjugate gradient at each step of algorithm, by default "fletcher-reeves"
        """
        self.f = f
        self.g = g
        self.g_barrier = (
            BARRIER_FUNCTIONS_MAPPING[g_barrier]()(g) 
            if isinstance(g_barrier, str) 
            else g_barrier(g)
        ) 
        self.solver = ConjugateGradientMethod(
            f=f, 
            line_search_method=line_search_method, 
            conjugate_gradient_direction_method=conjugate_gradient_direction_method
        )



