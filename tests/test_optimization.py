from ml_utils.math_functions import get_function
from ml_utils.optimization import minimize_with_restarts, sample_then_minimize


def test_twosines_minimum():
    f, X_LIM, _, _ = get_function('twosines-4d')
    result = minimize_with_restarts(f, X_LIM, num_restarts=10,
                                    hard_bounds=X_LIM)
    print(result)


def test_sample_then_minimize():
    f, X_LIM, _, _ = get_function('branin-2d')
    result1 = sample_then_minimize(f, X_LIM)
    result2 = sample_then_minimize(f, X_LIM, evaluate_sequentially=False)
    print(result1.x, result1.fun)
    print(result2.x, result2.fun)

if __name__ == '__main__':
    test_twosines_minimum()
    test_sample_then_minimize()
