import numpy as np
from typing import Optional

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
        if self.atol and np.abs(self.history[0] - self.history[1]) < self.atol:
            return True
        if self.rtol and self.history[(self.counter+1)%2]/self.history[self.counter%2] < 1+self.rtol:
            return True
        return False

