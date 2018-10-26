# -*- coding: utf-8 -*-

"""
Test functions for global optimisation.

Some of these are based on the following library:
    BayesOpt, an efficient C++ library for Bayesian optimization.
"""

import numpy as np
import GPy


def draw_from_a_gp(dim, x_lim, n=None, kern=None, seed=None):
    """
    A draw from a known GP kernel

    :dim (int): Dimensionality of x

    :x_lim (ndarray): limits of x

    :n (int): Total number of points used to generate the surface

    :kern (GPy.kern): kernel from which to draw the function

    Returns

    :f (func): Function object that provides the function value at a given x
    """
    assert dim == x_lim.shape[0]
    np.random.seed(seed)

    if n is None:
        n = 300

    n_for_each_dim = int(n ** (1 / dim))
    if kern is None:
        kern = GPy.kern.RBF(dim, lengthscale=0.4, variance=1.)

    single_x_vecs = []
    for ii in range(dim):
        single_x_vecs.append(np.linspace(
            x_lim[ii, 0], x_lim[ii, 1], n_for_each_dim))
    x = np.dstack((np.meshgrid(*single_x_vecs))).reshape(-1, dim)
    kern_K = kern.K(x, x) + 1e-5 * np.eye(len(x))

    draw = np.random.multivariate_normal(
        np.zeros(len(x)), kern_K).reshape(-1, 1)

    model = GPy.models.GPRegression(x, draw, kern)
    model.optimize()

    def pred_func(t):
        t = t.reshape(-1, dim)
        m, v = model.predict(t)
        return m

    # __name__ is used to define the filename of an experiment,
    # so making it more descriptive here.
    pred_func.__name__ = "draw-from-gp-{}d".format(dim)

    return pred_func


def hartmann6(x):
    """
    https://github.com/automl/HPOlib2/blob/master/hpolib/benchmarks
    /synthetic_functions/hartmann6.py

    in original [0, 1]^6 space:
    global optimum: (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573),
    min function value = -3.32237
    """
    # Re-scale input space from [-1, 1]^6 (X_LIM) to
    # [0,1]^6 (original definition of the function)
    # so that code below still works
    x = (x + 1) / 2.

    alpha = [1.00, 1.20, 3.00, 3.20]
    A = np.array([[10.00, 3.00, 17.00, 3.50, 1.70, 8.00],
                  [0.05, 10.00, 17.00, 0.10, 8.00, 14.00],
                  [3.00, 3.50, 1.70, 10.00, 17.00, 8.00],
                  [17.00, 8.00, 0.05, 10.00, 0.10, 14.00]])
    P = 0.0001 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                           [2329, 4135, 8307, 3736, 1004, 9991],
                           [2348, 1451, 3522, 2883, 3047, 6650],
                           [4047, 8828, 8732, 5743, 1091, 381]])

    external_sum = 0
    for i in range(4):
        internal_sum = 0
        for j in range(6):
            internal_sum += A[i, j] * (x[:, j] - P[i, j]) ** 2
        external_sum += alpha[i] * np.exp(-internal_sum)

    return external_sum


def ackley(x):
    """
    Ackley function
    x has to be 2D (NxD)
    Bounds = -1, 1
    min x = np.zeros
    y = 0
    """
    x = np.atleast_2d(x).copy()
    x *= 32.768  # rescale x to [-32.768, 32.768]
    n = x.shape[1]
    a = 20
    b = 0.2
    c = 2 * np.pi
    ccx = np.cos(c * x)
    return -a * np.exp(-b * np.sqrt(np.sum(x ** 2, 1) / n)) - \
           np.exp(np.sum(ccx, 1) / n) + a + np.exp(1)


def branin(x):
    """
    Branin function -- x = Nx2
    limits: -1 < x < 1

    unscaled x or y:
    Min = 0.1239 0.8183
    Min = 0.5428 0.1517  => 0.3979
    Min = 0.9617 0.1650

    for scaled x and y (current implementation)

    Min = -0.7522 0.6366
    Min = 0.08559 -0.6966  => 0.0013
    Min = 0.9234-0.6699

    """
    x = np.atleast_2d(x)
    # scaling x from [-1, 1] to [0, 1] to keep remaining func unchanged
    x = (x + 1) / 2
    a = x[:, 0] * 15 - 5
    b = x[:, 1] * 15

    # dividing output by 300 to reduce the range of the
    # function to roughly [0, 1.03]
    return ((b - (5.1 / (4 * np.pi ** 2)) * a ** 2 + 5 * a / np.pi - 6) ** 2 +
            10 * (1 - 1 / (8 * np.pi)) * np.cos(a) + 10) / 300


def camelback(x):
    """
    Camelback function
    -1 < x[:,0] < 1
    -1 < x[:,1] < 1

    Global minima: f(x) = -1.0316 (0.0449, -0.7126), (-0.0449, 0.7126)
    """
    x = np.atleast_2d(x).copy()
    x[:, 0] = x[:, 0] * 2

    tmp1 = (4 - 2.1 * x[:, 0] ** 2 + (x[:, 0] ** 4) / 3) * x[:, 0] ** 2
    tmp2 = x[:, 0] * x[:, 1]
    tmp3 = (-4 + 4 * x[:, 1] ** 2) * x[:, 1] ** 2
    return tmp1 + tmp2 + tmp3


def michalewicz(x):
    """
    Michalewicz Function
    %old Bounds [0,pi]
    % Bounds [-1, 1]
    %Min = -4.687 (n=5)
    min loc (before transforming x) [2.20, 1.57]
    min loc = [0.31885383,  0.]
    """
    x = np.atleast_2d(x)

    x = (x + 1) * np.pi / 2  # transform from [0,pi] to [-1,1]

    n = x.shape[1]
    m = 1
    ii = np.arange(1, n + 1)

    return -np.sum(np.sin(x) * (np.sin(ii * x ** 2 / np.pi)) ** (2 * m), 1)


def quadratic(x):
    """
    Simple quadratic function
    min at x = 0.53
    f(x_min) = 10.2809
    """

    x = np.atleast_2d(x)
    x -= 0.53
    return np.sum(x ** 2, 1) + 10


def rosenbrock(x):
    """
    Rosenbrock function 2D
    -1 < x < 1
    min y = 0
    """
    x = np.atleast_2d(x)
    x /= 2.048  # scaling x from [-2.048, 2.048] to [-1, 1]
    return (1 - x[:, 0]) ** 2 + 100 * (x[:, 1] - x[:, 0] ** 2) ** 2


# def levy(x):
#     """
#     Levy function
#     https://www.sfu.ca/~ssurjano/levy.html
#
#     f(x_min) = 0
#
#     """
#     x = np.atleast_2d(x * 10.)
#
#     def w(xx, ii):
#         """Helper function"""
#         return 1 + (xx[:, ii] - 1) / 4
#
#     return np.sin(np.pi * w(x, 0)) ** 2 + \
#            np.sum(np.hstack([(w(x, ii) - 1) ** 2 * (
#                    1 + 10 * np.sin(np.pi * w(x, ii) + 1) ** 2)
#                              for ii in range(x.shape[1] - 1)]), axis=0) + \
#            (w(x, -1) - 1) ** 2 * (1 + np.sin(2 * np.pi * w(x, -1) + 1) ** 2)


# def shc(x):
#     """
#     Six-Hump Camel function
#     https://www.sfu.ca/~ssurjano/camel6.html
#
#     Original space is x_0 = [-3, 3] and x_1 = [-2, 2]
#
#     Here both are re-scaled to [-1, 1] and cropped to avoid huge values
#
#     f(x_min) = -1.0316
#
#     """
#     scaling = np.array([2., 1.])
#
#     x = np.atleast_2d(x) * scaling
#
#     return (4 - 2.0 * x[:, 0] ** 2 + x[:, 0] ** 4 / 3) * x[:, 0] ** 2 + \
#            x[:, 0] * x[:, 1] + (-4 + 4 * x[:, 1] ** 2) * x[:, 1] ** 2


def get_function(target_func, dim=None, big=False):
    """
    Get objects and limits for a chosen function

    Returns (f, X_LIM, min_loc, min_val)
    """
    min_val = None

    if target_func.startswith('camelback-2d'):
        f_ = camelback
        min_loc = np.array([[0.0449, -0.7126], [-0.0449, 0.7126]])

        if not big:
            f = lambda x: f_(x) / 5
            min_val = -1.0316 / 5
        else:
            f = f_
            min_val = -1.0316

        text = 'Camelback function'
        X_LIM = np.array([[-1, 1], [-1, 1]])

    elif target_func == 'hartmann-6d':
        f = hartmann6
        min_loc = np.array(
            [-0.59662, -0.699978, -0.046252, -0.449336, -0.376696, 0.3146])
        min_val = -3.32237
        X_LIM = np.vstack([[-1, 1]] * 6)

    elif target_func.startswith('branin-2d'):
        f = branin
        min_loc = np.array([0.08559, -0.6966])
        min_val = 0.0013
        text = 'Branin function'
        X_LIM = np.array([[-1, 1], [-1, 1]])

    elif target_func.startswith('michalewicz'):
        f = michalewicz
        if len(target_func.split("-")) > 1:
            dim = int(target_func.split("-")[1][:-1])
        else:
            dim = 2

        if dim == 2:
            min_loc = np.array([0.3188, 0.])
        elif dim == 3:
            min_loc = np.array([0.3188, 0., -0.1694])
        elif dim == 4:
            min_loc = np.array([0.3191, 0., -0.1691, 0.2203])
        elif dim == 5:
            min_loc = np.array([0.3191, 0., -0.1691, 0.2203, 0.09419])
        else:
            raise NotImplementedError

        min_val = -4.687

        text = 'Michalewicz function'
        X_LIM = np.vstack([[-1, 1]] * dim)

    elif target_func.startswith('ackley'):
        f_ = ackley

        if len(target_func.split("-")) > 1:
            dim = int(target_func.split("-")[1][:-1])
        else:
            dim = 1

        min_loc = np.zeros(dim)
        if not big:
            f = lambda x: f_(x) / 20
            min_val = 0
        else:
            f = f_
        X_LIM = np.vstack([np.array([-1., 1])] * dim)
        text = 'Ackley function'

    elif target_func.startswith('rosenbrock-2d'):
        f_ = rosenbrock
        if not big:
            f = lambda x: f_(x) / 50
        else:
            f = f_

        min_loc = np.array([1., 1.])
        min_val = 0

        X_LIM = np.array([[-1, 1], [-1, 1]])
        text = 'Rosenbrock function'

    # elif target_func.startswith('levy'):
    #     f = levy
    #
    #     if len(target_func.split("-")) > 1:
    #         dim = int(target_func.split("-")[1][:-1])
    #     else:
    #         dim = 2
    #
    #     min_loc = np.ones(dim)
    #     min_val = 0
    #     f.__name__ = f'levy-{dim}d'
    #     X_LIM = np.vstack([np.array([-1., 1])] * dim)
    #     text = 'Levy function'
    #
    # elif target_func.startswith('shc'):
    #     f = shc
    #     min_loc = np.array([[0.0898 / 2, -0.7126],
    #                         [-0.0898 / 2, 0.7126]])
    #     min_val = -1.0316
    #     X_LIM = np.array([[-1, 1], [-1, 1]])
    #     text = 'SHC function'

    elif target_func.startswith('quadratic'):
        f = quadratic
        min_loc = np.array([0.53, 0.53])
        min_val = 10.2809
        X_LIM = np.array([[-1, 1], [-1, 1]])
        text = 'Quadratic function'
    else:
        print("target_func with name", target_func, "doesn't exist!")
        raise NotImplementedError

    f.__name__ = target_func
    return (f, X_LIM, min_loc, min_val)


def plot_test_func(target_func, *args):
    import pylab as plt
    f, X_LIM, _, _ = get_function(target_func, *args)
    n_x1 = 800
    n_x2 = 800
    x1_vals = np.linspace(X_LIM[0, 0], X_LIM[0, 1], n_x1)
    x2_vals = np.linspace(X_LIM[1, 0], X_LIM[1, 1], n_x2)[::-1]

    xx = np.dstack(np.meshgrid(x1_vals, x2_vals)).reshape(-1, 2)

    yy = f(xx)
    yy_grid = yy.reshape(n_x2, n_x1)

    plt.figure(figsize=(7, 7))
    plt.imshow(yy_grid, extent=(
        x1_vals.min(), x1_vals.max(), x2_vals.min(), x2_vals.max()))
    plt.title(target_func)
    print("function values between {:.2f} and {:.2f}".format(np.min(yy),
                                                             np.max(yy)))
