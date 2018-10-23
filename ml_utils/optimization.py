"""Optimization utilities"""

import numpy as np
import scipy as sp
from scipy import optimize


def minimize_with_restarts(optimiser_func, restart_bounds, num_restarts=5,
                           min_successes=3, max_tries=None, hard_bounds=None,
                           jac=None, minimize_options=None, verbose=False):
    """
    Runs scipy.optimize.minimize() with random restarts

    """
    # Hard upper limit to kill the optimization if we keep on failing
    if max_tries is None:
        max_tries = (num_restarts + min_successes) * 3

    # If options (maxiter) or jac is provided, pass that to minimize
    # minimize_options is a dict like {'maxiter':100} or None
    if jac is None:
        def minimizer(x):
            return optimize.minimize(optimiser_func,
                                     x,
                                     bounds=hard_bounds,
                                     options=minimize_options)
    else:
        def minimizer(x):
            return optimize.minimize(optimiser_func,
                                     x,
                                     jac=jac,
                                     bounds=hard_bounds,
                                     options=minimize_options)

    if type(restart_bounds) is list:
        restart_bounds = np.array(restart_bounds)
    best_eval = None
    best_opt_result = None
    nfev = 0
    ncrashes = 0
    n_runs = 0
    continue_trying = True
    # for ii in range(num_restarts):
    while continue_trying:
        x0 = (restart_bounds[:, 1] - restart_bounds[:, 0]) \
             * np.random.random_sample((restart_bounds.shape[0],)) \
             + restart_bounds[:, 0]

        if verbose:
            print("multistart iteration", n_runs, 'out of', num_restarts)
            print("starting optimisation from x =", x0)
            print(
                f"n_runs = {n_runs}, ncrashes = {ncrashes}, max_tries = "
                f"{max_tries}")

        try:
            opt_result = minimizer(x0)
            nfev += opt_result.nfev
            if opt_result.status == 1:
                if verbose:
                    print("optimisation failed!")
            else:
                curr_x = opt_result.x
                if best_opt_result is None:
                    best_opt_result = opt_result
                    best_eval = (curr_x, optimiser_func(curr_x))
                    if verbose:
                        print("Updating best to", best_eval)
                else:
                    if optimiser_func(curr_x) < best_eval[1]:
                        best_opt_result = opt_result
                        best_eval = (curr_x, optimiser_func(curr_x))
                        if verbose:
                            print("Updating best to", best_eval)
        except (np.linalg.LinAlgError, sp.linalg.LinAlgError):
            if verbose:
                print("multistart iteration {} failed".format(n_runs))
            ncrashes += 1

        # While we haven't reached the maximum number of run as well
        # as the minimum number of successful optimizations, we continue
        n_runs += 1
        if n_runs >= num_restarts and (n_runs - ncrashes) > min_successes:
            if verbose:
                print("Reached desired number of restarts and successes.")
            continue_trying = False
        elif n_runs >= max_tries:
            if verbose:
                print("Maximum number of tries reached. " +
                      "Not enough successes, but stopping anyway.")
            continue_trying = False

    if ncrashes == n_runs:  # if all optimizations failed
        print("All multi-started optimizations encountered LinAlgErrors!")
    if verbose:
        print("Completed multigrad with", num_restarts,
              " restarts and total nfev =", nfev)
    return best_opt_result



if __name__ == '__main__':
    def f(x):
        raise np.linalg.LinAlgError


    minimize_with_restarts(f, [[0, 20]], num_restarts=20, min_successes=2,
                           verbose=True)