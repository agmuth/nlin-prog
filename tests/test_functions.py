import numpy as np 
from typing import Union

class UnconstrainedOptimizationTestFunction():
    def __init__(self, f: callable, x_min: np.ndarray, x_start: np.ndarray):
        self.f = f
        self.x_min = x_min
        self.x_start = x_start

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

sphere_func_2d = UnconstrainedOptimizationTestFunction(
    f=lambda x: np.sum(x**2),
    x_min=np.zeros(2),
    x_start=1*np.ones(2)
)

bazaraa_ex = UnconstrainedOptimizationTestFunction(
    f=lambda x: (x[0] - 2)**4 + (x[0] - 2*x[1])**2,
    x_min=np.array([2., 1.]),
    x_start=np.array([0.0, 3.0])
)

booth_func = UnconstrainedOptimizationTestFunction(
    f=lambda x: (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2,
    x_min=np.array([1., 3.]),
    x_start=np.array([-8., -8.])
)

matyas_func = UnconstrainedOptimizationTestFunction(
    f=lambda x: 0.26*np.sum(x*x) - 0.48*np.prod(x),
    x_min=np.array([0., 0.]),
    x_start=np.array([2., -8.])
)


rosebrock_10d_func = UnconstrainedOptimizationTestFunction(
    f=lambda x: 100*np.sum((x[1:] - x[:-1]**2)**2 + (1 - x[:-1]**2)),
    x_min=np.ones(10),
    x_start=-5*np.ones(1)
)

# goldstien_price_func = UnconstrainedOptimizationTestFunction(
#     f=lambda x: (
#         (
#             1 
#             + (x.sum() + 1)**2 
#             * (19 - 14*x[0] + 3*x[0]**2 - 14*x[1] + 6*x.prod() + 3*x[1]**2) 
#         )
#         * (
#             30
#             + (2*x[0] - 3*x[1])**2
#             * (18 - 32*x[0] + 12*x[0]**2 + 48*x[1] - 36*x.prod() + 27*x[1]**2) 
#         ) 
#     ),
#     x_min=np.array([0., -1.]),
#     x_start=np.array([-2., 2.])
# )

UNCONSTRAINED_OPTIMIZATION_TEST_FUNCTIONS = [x[1] for x in globals().items() if isinstance(x[1], UnconstrainedOptimizationTestFunction)]
CONSTRAINED_OPTIMIZATION_TEST_FUNCTIONS = [x[1] for x in globals().items() if isinstance(x[1], ConstrainedOptimizationTestFunction)]