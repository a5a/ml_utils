from ml_utils.math_functions import get_function
from ml_utils.optimization import minimize_with_restarts
import numpy as np

def test_x_lim_dims():
    funcs = ('twosines', 'ackley', )
    dims = np.arange(1, 20)
    for dim in dims:
        _, X_LIM, _, _ = get_function(f'twosines-{dim}d')
        assert len(X_LIM) == dim




if __name__ == '__main__':
    test_x_lim_dims()
