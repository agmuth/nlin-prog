import numpy as np 
from typing import Union


class ConstrainedOptimizationTestFunction():
    def __init__(self, f: callable, g: Union[callable, None], h: Union[callable, None], x_min: np.ndarray, x_start: np.ndarray):
        self.f = f
        self.g = g
        self.h = h
        self.x_min = x_min
        self.x_start = x_start

constrained_rosenbrock1 = ConstrainedOptimizationTestFunction(
    f=lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2,
    g=lambda x: np.array([
        (x[0] - 1)**3 - x[1] + 1,
        x.sum() - 2
    ]),
    h=None,
    x_start=np.array([1.0, -0.5]),
    x_min=np.zeros(2),
)

CONSTRAINED_OPTIMIZATION_TEST_FUNCTIONS = [x[1] for x in globals().items() if isinstance(x[1], ConstrainedOptimizationTestFunction)]