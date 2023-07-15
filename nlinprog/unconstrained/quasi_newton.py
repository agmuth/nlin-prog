from abc import ABC, abstractclassmethod
from types import MappingProxyType
from typing import Optional, Union

import numpy as np

from nlinprog.line_search import LINE_SEARCH_MAPPING, LineSearch
from nlinprog.numerical_differentiation import central_difference as jacobian
from nlinprog.numerical_differentiation import hessian
from nlinprog.utils import ConvergenceTest, build_result_object


class ApproxInverseHessian(ABC):
    @abstractclassmethod
    def __call__(self, f, x_k, H_k, p_k, q_k, *args, **kwargs):
        pass


class NumericalInverseHessian(ApproxInverseHessian):
    def __call__(self, f: callable, x_k: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Calculates the inverse hessian of `f` at `x_k`.

        Parameters
        ----------
        f : callable
            Function to calculate inverse hessian of.
        x_k : np.ndarray
            Point to calcualte inverse hessian at.

        Returns
        -------
        np.ndarray
            Inverse Hessian.
        """
        H_k_plus_one = hessian(f)(x_k)
        H_k_plus_one = 0.5 * (H_k_plus_one + H_k_plus_one.T) + 1e-4 * np.eye(
            x_k.shape[0]
        )  # add jitter
        return np.linalg.inv(H_k_plus_one)


class IdentityInverseHessian(ApproxInverseHessian):
    def __call__(self, x_k: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Returns Identity matrix of shape compatible with `x_k`."""
        return np.eye(x_k.shape[0])


class DFPInverseHessian(ApproxInverseHessian):
    def __call__(
        self, H_k: np.ndarray, p_k: np.ndarray, q_k: np.ndarray, *args, **kwargs
    ) -> np.ndarray:
        """Calculates the Davidon-Fletched-Powell approximation of the inverse hessian.
        ref: pg. 290 Linear and Nonlinear Programming Luenberger + Ye

        Parameters
        ----------
        H_k : np.ndarray
            Previous DFP estimate.
        p_k : np.ndarray
            Update vector applied to x_k (a_k * d_k).
        q_k : np.ndarray
            Difference between gradients at timestamps k+1 and k.

        Returns
        -------
        np.ndarray
            DFP estiamte for timestamp k+1.
        """
        H_mult_q = H_k @ q_k
        H_k_plus_one = np.array(H_k)
        H_k_plus_one += np.divide(np.outer(p_k, p_k), (p_k @ q_k.T))
        H_k_plus_one -= np.divide(np.outer(H_mult_q, H_mult_q), (q_k.T @ H_mult_q))
        return H_k_plus_one


class BFGSInverseHessian(ApproxInverseHessian):
    def __call__(
        self, H_k: np.ndarray, p_k: np.ndarray, q_k: np.ndarray, *args, **kwargs
    ) -> np.ndarray:
        """Calculates the Broyden-Fletcher-Goldfard-Shanno approximation of the inverse hessian.
        ref: pg. 294 Linear and Nonlinear Programming Luenberger + Ye

        Parameters
        ----------
        H_k : np.ndarray
            Previous BFGS estimate.
        p_k : np.ndarray
            Update vector applied to x_k (a_k * d_k).
        q_k : np.ndarray
            Difference between gradients at timestamps k+1 and k.

        Returns
        -------
        np.ndarray
            BFGS estiamte for timestamp k+1.
        """
        q_dot_p_inv = (q_k.T @ p_k) ** -1
        q_outer_p = np.outer(q_k, p_k)
        p_outer_q = q_outer_p.T
        p_outer_p = np.outer(p_k, p_k)
        H_k_plus_one = np.array(H_k)
        H_k_plus_one += p_outer_p * q_dot_p_inv
        H_k_plus_one += p_outer_q @ H_k @ q_outer_p * q_dot_p_inv**2
        H_k_plus_one -= (H_k @ q_outer_p + p_outer_q @ H_k) * q_dot_p_inv
        return H_k_plus_one


class BroydenInverseHessian(ApproxInverseHessian):
    def __init__(self, phi: float = 0.5):
        """init

        Parameters
        ----------
        phi : float, optional
            Weight given to BFGS inverse hessian 1-phi given to DFP inverse hessian, by default 0.5
        """
        self.phi = phi
        self.calc_dfp_inverse_hessian = DFPInverseHessian()
        self.calc_bfgs_inverse_hessian = BFGSInverseHessian()

    def __call__(
        self, H_k: np.ndarray, p_k: np.ndarray, q_k: np.ndarray, *args, **kwargs
    ) -> np.ndarray:
        """Calculates the Broydenof the inverse hessian.
        ref: pg. 293 Linear and Nonlinear Programming Luenberger + Ye

        Parameters
        ----------
        H_k : np.ndarray
            Previous BFGS estimate.
        p_k : np.ndarray
            Update vector applied to x_k (a_k * d_k).
        q_k : np.ndarray
            Difference between gradients at timestamps k+1 and k.

        Returns
        -------
        np.ndarray
            Broyden estiamte for timestamp k+1.
        """
        # H_k_plus_one = self.calc_dfp_inverse_hessian(H_k, p_k, q_k)
        # v_k = np.sqrt(q_k.T @ H_k @ q_k) * (p_k / (p_k.T @ q_k) - H_k @ q_k / (q_k.T @ H_k @ q_k))
        # H_k_plus_one += self.phi * np.outer(v_k, v_k)
        # return H_k_plus_one
        return (1 - self.phi) * self.calc_dfp_inverse_hessian(
            H_k, p_k, q_k
        ) + self.phi * self.calc_bfgs_inverse_hessian(H_k, p_k, q_k)


APPROX_INVERSE_HESSIAN_MAPPING = MappingProxyType(
    # supported methods for estimating inverse hessian
    {
        "bfgs": BFGSInverseHessian,
        "dfp": DFPInverseHessian,
        "broyden": BroydenInverseHessian,
        "exact": NumericalInverseHessian,
        "identity": IdentityInverseHessian,
    }
)


class QuasiNewtonMethod:
    """Quasi Newton method for unconstrained nonlinear optimization."""

    def __init__(
        self,
        f: callable,
        line_search_method: Union[LineSearch, str] = "wolfe",
        inverse_hessian_method: Union[ApproxInverseHessian, str] = "exact",
    ):
        """init

        Parameters
        ----------
        f : callable
            Funciton to minimize.
        line_search_method : Union[LineSearch, str], optional
            Line search method to use in each step of algorithm, by default "wolfe"
        inverse_hessian_method : Union[ApproxInverseHessian, str], optional
            Method to calculate approximation of inverse hessian at each step of algorithm, by default "exact"
        """
        self.f = f
        self.line_search = (
            LINE_SEARCH_MAPPING[line_search_method](c1=1e-4, c2=0.9, eta=2, epsilon=0.2)
            if isinstance(line_search_method, str)
            else line_search_method
        )
        self.calc_inverse_hessian = (
            APPROX_INVERSE_HESSIAN_MAPPING[inverse_hessian_method]()
            if isinstance(inverse_hessian_method, str)
            else inverse_hessian_method
        )

    def solve(
        self,
        x_0: np.ndarray,
        maxiters: int = 200,
        atol: Optional[float] = 1e-4,
        rtol: Optional[float] = 1e-4,
    ):
        """Run Quasi Newton Method

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
        g_k = grad(x_k)
        H_k = np.eye(x_0.shape[0])  # inverse hessian

        for k in range(maxiters):
            # (modified/quasi) newton's method/step
            d_k = -1 * H_k @ g_k  # search direction for x_k
            alpha_k = self.line_search(
                f=self.f, x_k=x_k, d_k=d_k
            )  # minimizer of `f(x_k + alpha*d_k)`
            p_k = alpha_k * d_k  # update to x_k
            x_k_plus_one = x_k + p_k
            g_k_plus_one = grad(x_k_plus_one)
            q_k = g_k_plus_one - g_k
            H_k = self.calc_inverse_hessian(
                f=self.f, x_k=x_k, H_k=H_k, p_k=p_k, q_k=q_k
            )  # really `h_k_plus_one` but not used in this iter
            x_k, g_k = x_k_plus_one, g_k_plus_one  # update for next iter

            # check for convergence
            f_k = self.f(x_k)
            convergence_test.update(f_k)
            if convergence_test.converged:
                converged = True
                break

        return build_result_object(self.f, x_k, k, converged)
