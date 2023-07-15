from typing import Optional, Union

from nlinprog.constrained.penalty.penalty_functions import (
    EQUALITY_PENALTY_FUNCTIONS_MAPPING, INEQUALITY_PENALTY_FUNCTIONS_MAPPING,
    EqualityPenaltyFunction, InequalityPenaltyFunction, zero_func)
from nlinprog.constrained.penalty.quasi_newton import \
    PenalizedQuasiNewtonMethod
from nlinprog.line_search import LineSearch
from nlinprog.unconstrained.conjugate_gradient import (
    CalcConjugateGradientDirection, ConjugateGradientMethod)


class PenalizedConjugateGradientMethod(PenalizedQuasiNewtonMethod):
    """Conjugate Gradient method for constrained nonlinear optimization using penalty methods."""

    def __init__(
        self,
        f: callable,
        g: Optional[callable] = None,
        h: Optional[callable] = None,
        g_penalty: Optional[Union[InequalityPenaltyFunction, str]] = "relu-squared",
        h_penalty: Optional[Union[EqualityPenaltyFunction, str]] = "squared",
        line_search_method: Union[LineSearch, str] = "wolfe",
        conjugate_gradient_direction_method: Union[
            CalcConjugateGradientDirection, str
        ] = "fletcher-reeves",
    ):
        """init

        Parameters
        ----------
        f : callable
            Function to minimize
        g : Optional[callable], optional
            Function describing constraints of form `g(x) <= 0`, by default None
        h : Optional[callable], optional
            Function describing constraints of form `h(x) = 0`, by default None
        g_penalty : Optional[Union[InequalityPenaltyFunction, str]], optional
            Penalty function to apply to `g`, by default "relu-squared"
        h_penalty : Optional[Union[EqualityPenaltyFunction, str]], optional
            Penalty function to apply to `h`, by default "squared"
        line_search_method : Union[LineSearch, str], optional
            Line search method to use in each step of algorithm, by default "wolfe"
        conjugate_gradient_direction_method : Union[CalcConjugateGradientDirection, str], optional
            Method to calculate conjugate gradient at each step of algorithm, by default "fletcher-reeves"
        """
        self.f = f
        self.g = g
        self.h = h
        self.g_penalized = (
            zero_func
            if not g
            else INEQUALITY_PENALTY_FUNCTIONS_MAPPING[g_penalty]()(g)
            if isinstance(g_penalty, str)
            else g_penalty(g)
        )
        self.h_penalized = (
            zero_func
            if not h
            else EQUALITY_PENALTY_FUNCTIONS_MAPPING[h_penalty]()(h)
            if isinstance(h_penalty, str)
            else h_penalty(h)
        )
        self.solver = ConjugateGradientMethod(
            f=f,
            line_search_method=line_search_method,
            conjugate_gradient_direction_method=conjugate_gradient_direction_method,
        )
