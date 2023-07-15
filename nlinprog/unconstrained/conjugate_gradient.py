from abc import ABC, abstractclassmethod
from types import MappingProxyType
from typing import Optional, Union

import numpy as np

from nlinprog.line_search import LINE_SEARCH_MAPPING, LineSearch
from nlinprog.numerical_differentiation import central_difference as jacobian
from nlinprog.utils import ConvergenceTest, build_result_object


class CalcConjugateGradientDirection(ABC):
    eps = 1e-8

    @abstractclassmethod
    def __call__(self):
        pass


class FletcherReevesConjugateGradientDirection(CalcConjugateGradientDirection):
    def __call__(
        self, d_k: np.ndarray, g_k: np.ndarray, g_k_plus_one: np.ndarray
    ) -> np.ndarray:
        """Computes the conjugate gradient direction accoring to Fletcher-Reeves method.
        ref: pg. 278 Linear and Nonlinear Programming Luenberger + Ye

        Parameters
        ----------
        d_k : np.ndarray
            Negative gradient of `f` at `x_{k}`.
        g_k : np.ndarray
            Gradient of `f` at `x_{k}`.
        g_k_plus_one : np.ndarray
            Gradient of `f` at `x_{k+1}`.

        Returns
        -------
        np.ndarray
            Conjugate gradient direction.
        """

        beta_k = g_k_plus_one.T @ g_k_plus_one / max(g_k.T @ g_k, self.eps)
        return -1 * g_k_plus_one + beta_k * d_k


class PolakRibiereConjugateGradientDirection(CalcConjugateGradientDirection):
    def __call__(
        self, d_k: np.ndarray, g_k: np.ndarray, g_k_plus_one: np.ndarray
    ) -> np.ndarray:
        """Computes the conjugate gradient direction accoring to Polak-Ribiere method.
        ref: pg. 278 Linear and Nonlinear Programming Luenberger + Ye

        Parameters
        ----------
        d_k : np.ndarray
            Negative gradient of `f` at `x_{k}`.
        g_k : np.ndarray
            Gradient of `f` at `x_{k}`.
        g_k_plus_one : np.ndarray
            Gradient of `f` at `x_{k+1}`.

        Returns
        -------
        np.ndarray
            Conjugate gradient direction.
        """
        beta_k = (g_k_plus_one - g_k).T @ g_k_plus_one / max(g_k.T @ g_k, self.eps)
        return -1 * g_k_plus_one + beta_k * d_k


CALC_CONJUGATE_GRADIENT_DIRECTION_MAPPING = MappingProxyType(
    # supported methods for calculating CG
    {
        "fletcher-reeves": FletcherReevesConjugateGradientDirection,
        "polak-ribiere": PolakRibiereConjugateGradientDirection,
    }
)


class ConjugateGradientMethod:
    """Conjugate Gradient method for unconstrained nonlinear optimization."""

    def __init__(
        self,
        f: callable,
        line_search_method: Union[LineSearch, str] = "wolfe",
        conjugate_gradient_direction_method: Union[
            CalcConjugateGradientDirection, str
        ] = "fletcher-reeves",
    ):
        """init

        Parameters
        ----------
        f : callable
            Funciton to minimize.
        line_search_method : Union[LineSearch, str], optional
            Line search method to use in each step of algorithm, by default "wolfe"
        conjugate_gradient_direction_method : Union[CalcConjugateGradientDirection, str], optional
            Method to calculate conjugate gradient at each step of algorithm, by default "fletcher-reeves"
        """
        self.f = f
        self.line_search = (
            LINE_SEARCH_MAPPING[line_search_method](c1=1e-4, c2=0.1, eta=2, epsilon=0.2)
            if isinstance(line_search_method, str)
            else line_search_method
        )
        self.calc_conjugate_direction = (
            CALC_CONJUGATE_GRADIENT_DIRECTION_MAPPING[
                conjugate_gradient_direction_method
            ]()
            if isinstance(conjugate_gradient_direction_method, str)
            else conjugate_gradient_direction_method
        )

    def solve(
        self,
        x_0: np.ndarray,
        maxiters: int = 200,
        atol: Optional[float] = 1e-4,
        rtol: Optional[float] = 1e-4,
    ):
        """Run Conjugate Gradient Method

        Parameters
        ----------
        x_0 : np.ndarray
            Starting point.
        maxiters : int, optional
            Maximum number of iterations for algorithm to run, by default 200
        atol : Optional[float], optional
            Absolute tolerance for early termination, applied to sequence of f(x), by default 1e-4
        rtol : Optional[float], optional
            Absolute tolerance for early termination, applied to sequence of f(x), by default 1e-4

        Returns
        -------
        NonLinProgResult
            Result object.
        """
        # set up convergence tracking
        convergence_test = ConvergenceTest(atol=atol, rtol=rtol)
        convergence_test.update(self.f(x_0))
        converged = False

        # init params for algorithm
        x_k = np.array(x_0).astype(np.float64)
        grad = jacobian(self.f)
        n = x_k.shape[0]

        for i in range(maxiters):
            # re init at the begining of each outer loop
            g_k = grad(x_k).T
            d_k = -1 * g_k
            for k in range(n):
                # one full step of algorithm consists of `n` inner steps
                alpha_k = self.line_search(
                    f=self.f, x_k=x_k, d_k=d_k
                )  # minimizer of f(x_k + alpha*d_k)
                x_k += alpha_k * d_k
                g_k_plus_one = grad(x_k).T
                d_k = self.calc_conjugate_direction(
                    d_k=d_k, g_k=g_k, g_k_plus_one=g_k_plus_one
                )
                g_k = g_k_plus_one

            convergence_test.update(self.f(x_k))
            if convergence_test.converged:
                converged = True
                break

            if converged:
                break

        return build_result_object(self.f, x_k, i, converged)
