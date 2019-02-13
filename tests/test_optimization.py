import numpy as np

from ml_utils.math_functions import get_function
from ml_utils.optimization import minimize_with_restarts, sample_then_minimize


def test_twosines_minimum():
    f, X_LIM, _, _ = get_function('twosines-4d')
    result = minimize_with_restarts(f, X_LIM, num_restarts=10,
                                    hard_bounds=X_LIM)
    print(result)


def test_sample_then_minimize():
    f, X_LIM, _, _ = get_function('branin-2d')
    np.random.seed(1)
    result1 = sample_then_minimize(f, X_LIM, num_samples=10000, num_local=1)
    np.random.seed(1)
    result2 = sample_then_minimize(f, X_LIM, num_samples=10000,
                                   evaluate_sequentially=False,
                                   num_local=1)
    print(result1.x, result1.fun)
    print(result2.x, result2.fun)


if __name__ == '__main__':
    # test_twosines_minimum()
    test_sample_then_minimize()
