import time
import numpy as np

from ml_utils.misc import time_limit, TimeoutException, \
    unnormalise_x_given_lims


def test_timed_run():
    def long_function_call(t):
        time.sleep(t)
        print(f"Completed function call of duration {t}s")

    # Test good
    print("This should run successfully:")
    try:
        with time_limit(4):
            long_function_call(3)
    except TimeoutException:
        print(f"Execution timed out!")

    # Test bad
    print("This should time out:")
    try:
        with time_limit(2):
            long_function_call(3)
    except TimeoutException:
        print(f"Execution timed out!")


def test_unnormalise_x_given_lims():
    x1 = -1 * np.ones(3)
    x2 = 1 * np.ones(3)

    lims = np.array([
        [-5, 5],
        [1, 5],
        [-5, -3]
    ])

    x1_unnorm = unnormalise_x_given_lims(x1, lims)
    x2_unnorm = unnormalise_x_given_lims(x2, lims)

    assert np.allclose(x1_unnorm, lims[:, 0])
    assert np.allclose(x2_unnorm, lims[:, 1])


if __name__ == '__main__':
    # test_timed_run()
    test_unnormalise_x_given_lims()
