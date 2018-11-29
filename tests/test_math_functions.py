from ml_utils.math_functions import get_function
from ml_utils.optimization import minimize_with_restarts
import numpy as np

def test_x_lim_dims():
    funcs = ('twosines', 'ackley', )
    dims = np.arange(1, 20)
    for dim in dims:
        _, X_LIM, _, _ = get_function(f'twosines-{dim}d')
        assert len(X_LIM) == dim

def test_get_function():
    name = 'egg-2d'

    f, X_LIM, min_loc, min_val = get_function(name)

    assert f.__name__ == name


if __name__ == '__main__':
    # test_x_lim_dims()
    test_get_function()