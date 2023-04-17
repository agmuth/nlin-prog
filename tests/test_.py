import numpy as np 


def test_():
    n = 2
    eps = 1e-4
    assert np.allclose(
        np.zeros(n),
        np.zeros(n) - eps/10, 
        atol=eps
    )