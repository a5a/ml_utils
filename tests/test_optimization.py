from ml_utils.math_functions import get_function
from ml_utils.optimization import minimize_with_restarts


def test_twosines_minimum():
    f, X_LIM, _, _ = get_function('twosines-4d')
    result = minimize_with_restarts(f, X_LIM, num_restarts=10,
                                    hard_bounds=X_LIM)
    print(result)


if __name__ == '__main__':
    test_twosines_minimum()
