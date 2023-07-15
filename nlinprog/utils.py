import numpy as np
from typing import Optional
from nlinprog.numerical_differentiation import central_difference, hessian
from  dataclasses import dataclass
from abc import ABC, abstractproperty
from nlinprog.constants import ATOL, RTOL

class ConvergenceTest:
    def __init__(self, atol: Optional[float]=0.0, rtol: Optional[float]=0.0):
        self.atol = atol
        self.rtol = rtol
        self.history = np.array([None, None])
        self.counter = -1
        self.history[self.counter%2] = np.inf
    
    def update(self, val: float):
        self.counter += 1
        self.history[self.counter%2] = val
    
    @property
    def converged(self) -> bool:
        if self.atol > 0:
            return np.linalg.norm(self.history[self.counter%2] - self.history[(self.counter-1)%2]) < self.atol
        if self.rtol > 0:
            return (1 - (self.history[self.counter%2] / self.history[(self.counter-1)%2]) < self.rtol)
        return False


    


@dataclass
class NonLinProgResult:
    x: np.ndarray
    func: np.ndarray
    grad: np.ndarray
    hess: np.ndarray
    iters: int
    converged: bool

def build_result_object(f: callable, x: np.ndarray, iters: int, converged: bool) -> NonLinProgResult:
    return NonLinProgResult(x, f(x), central_difference(f)(x), hessian(f)(x), iters, converged)