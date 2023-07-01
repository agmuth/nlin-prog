import numpy as np
from typing import Optional, NamedTuple
from nlinprog.numerical_differentiation import central_difference, hessian

class SimpleConvergenceTest():
    def __init__(self, val: float, atol: Optional[float]=None, rtol: Optional[float]=None):
        self.atol = atol
        self.rtol = rtol
        self.history = np.empty(2)
        self.counter = -1
        self.update(val)

    def update(self, val) -> None:
        self.history[self.counter%2] = val
        self.counter += 1

    def converged(self) -> bool:
        if self.atol is None and self.rtol is None:
            return True
        if self.counter < 2:
            return False
        if self.atol and np.abs(self.history[(self.counter+1)%2]) < self.atol:
            return True
        if self.rtol and self.history[(self.counter+1)%2]/self.history[self.counter%2] < 1+self.rtol:
            return True
        return False


class ResultsObject(NamedTuple):
    x: np.ndarray
    func: np.ndarray
    grad: np.ndarray
    hess: np.ndarray
    iters: int
    converged: bool

def build_result_object(f: callable, x: np.ndarray, iters: int, converged: bool) -> dict:
    return ResultsObject(x, f(x), central_difference(f)(x), hessian(f)(x), iters, converged)