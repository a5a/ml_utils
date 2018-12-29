import time

from ml_utils.misc import time_limit, TimeoutException


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


if __name__ == '__main__':
    test_timed_run()
