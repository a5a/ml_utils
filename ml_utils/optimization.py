"""Optimization utilities"""
from typing import Callable, Optional, Dict

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


def sample_then_minimize(
        optimiser_func: Callable, sampling_bounds: np.ndarray,
        num_samples: Optional[int] = 1000, num_local: Optional[int] = 5,
        jac: Optional[Callable] = None,
        minimize_options: Optional[Dict] = None,
        evaluate_sequentially: Optional[bool] = True,
        verbose: Optional[bool] = False) -> optimize.OptimizeResult:
    """Samples from the func and then optimizes the most promising locations

    Parameters
    ----------
    optimiser_func
        Function to be minimized. Inputs are expected to be 2D.

    sampling_bounds
        Bounds for sampling and optimization

    num_samples
        Number of initial samples to take. Sampling is done uniformly
        using the bounds as limits

    num_local
        Number of local optimizations. This is the number of most promising
        samples used as starting points for minimize()
    jac
        If available, the jacobian of optimiser_func

    minimize_options
        Options passed to minimize(), e.g. maxiter

    evaluate_sequentially
        Whether the optimiser_func can return the result for multiple inputs.
        This is not the case for e.g. the log likelihood of a GP, but may
        be possible for an acquisition function. Default behaviour is to
        evaluate the optimiser_func sequentially.

    verbose

    Returns
    -------
    scipy OptimizeResult of the best local optimization
    """
    x_samples = np.random.uniform(sampling_bounds[:, 0],
                                  sampling_bounds[:, 1],
                                  (num_samples, sampling_bounds.shape[0]))
    if evaluate_sequentially:
        if verbose:
            print(f"Evaluating {num_samples} locations sequentially")

        f_samples = np.zeros(num_samples)
        for ii in range(num_samples):
            f_samples[ii] = optimiser_func(x_samples[ii])
    else:
        if verbose:
            print(f"Evaluating {num_samples} locations")

        f_samples = optimiser_func(x_samples)

    best_indexes = f_samples.argsort()[-num_local:][::-1]
    x_locals = x_samples[best_indexes]

    if verbose:
        print(f"Locally optimizing the top {num_local} locations")

    best_result = None
    best_f = np.inf
    for ii in range(num_local):
        x0 = np.atleast_2d(x_locals[ii])
        res = sp.optimize.minimize(
            optimiser_func, x0, jac=jac,
            options=minimize_options)  # type: optimize.OptimizeResult

        if res.fun < best_f:
            best_result = res
            best_f = res.fun

    if verbose:
        print(f"Best result found: {best_result.x} "
              f"has function value {best_result.fun}")

    return best_result

if __name__ == '__main__':
    def f(x):
        raise np.linalg.LinAlgError


    minimize_with_restarts(f, [[0, 20]], num_restarts=20, min_successes=2,
                           verbose=True)
