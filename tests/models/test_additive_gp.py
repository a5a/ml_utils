from pprint import pprint

import GPy
import matplotlib.pyplot as plt
import numpy as np

from bayesopt.acquisition import EI
from ml_utils.models.gp import GP
from ml_utils.models.additive_gp import create_additive_kernel, AdditiveGP, \
    StationaryUniformCat


def test_creation():
    k = create_additive_kernel(GPy.kern.RBF, GPy.kern.Matern52, [0, 1, 2],
                               [3, 4])

    print(k)

    x = np.random.random((3, 5))
    y = np.sum(np.sin(x), 1).reshape(-1, 1)

    gp = AdditiveGP(x, y, k, lik_variance=0.1)
    gp.optimize()

    x_new = np.random.random((10, 5))

    mu, var = gp.predict(x_new)
    mu_sub, var_sub = gp.predict_latent_subspace(x_new, 0)
    # pprint([[m] + [v] for m, v in zip(mu, var)])
    pprint([[m] + [v] for m, v in zip(mu_sub, var_sub)])

    # print(k.K(x))
    # k_rbf = GPy.kern.RBF(3, active_dims=[0, 1, 2])
    # k_mat = GPy.kern.Matern52(2, active_dims=[3, 4])
    #
    # k_rbf_val = k_rbf.K(x)
    # k_mat_val = k_mat.K(x)
    #
    # print(k_rbf_val + k_mat_val)


def test_subspace_learning():
    x1 = np.random.rand(20)[:, None]
    x2 = np.random.rand(20)[:, None]
    x1 = np.hstack((x1, 0.01 * np.random.rand(*x1.shape)))
    x2 = np.hstack((0.01 * np.random.rand(*x2.shape), x2))

    a = 7

    def f(x):
        return np.sum(np.sin(a * x), 1)

    x_ob = np.vstack((x1, x2))
    y_ob = f(x_ob).reshape(-1, 1)

    # ------ Test grid -------- #
    x1, x2 = np.mgrid[0:1:50j, 0:1:50j]
    X = np.vstack((x1.flatten(), x2.flatten())).T
    y = f(X)
    y -= np.mean(y)

    x_test = np.hstack([np.linspace(0, 1, 100).reshape(-1, 1)] * 2)

    k = create_additive_kernel(GPy.kern.RBF, GPy.kern.RBF, [0], [1])
    gp = AdditiveGP(x_ob, y_ob, k)

    gp.optimize()

    # x_test = np.random.rand(1000, 2)
    mu0, var0 = gp.predict_latent_subspace(x_test, 0)
    mu1, var1 = gp.predict_latent_subspace(x_test, 1)

    true_values = f(x_test)

    # print(np.abs(mu-true_values))

    f, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(x_test[:, 0], mu0.flatten())
    ax1.plot(x_test[:, 0], np.sin(a * x_test[:, 0]))
    ax2.plot(x_test[:, 1], mu1.flatten())
    ax2.plot(x_test[:, 1], np.sin(a * x_test[:, 1]))

    plt.show()

    mu, var = gp.predict(X)

    # # ------ Plot figures -------- #
    figure, axes = plt.subplots(2, 1, figsize=(10, 10))
    sub1 = axes[0].contourf(x1, x2, y.reshape(50, 50))
    axes[0].plot(x_ob[:, 0], x_ob[:, 1], 'r ^')
    axes[0].set_title('objective func')

    sub2 = axes[1].contourf(x1, x2, mu.reshape(50, 50))
    axes[1].plot(x_ob[:, 0], x_ob[:, 1], 'r ^')
    axes[1].set_title('prediction')
    plt.show()


# def test_kernel_with_delta():
#     k_continuous = GPy.kern.RBF(1, active_dims=[0])
#     k = KernelWithDelta(k_continuous, [1])
#     print(k.lengthscale)
#     # DOESN'T WORK :(
#
#
# def test_rbf_with_delta():
#     x = np.arange(10).reshape(-1, 1)
#     x = np.hstack((x, x))
#
#     k = RBFWithDelta([0], [1])
#     k = GPy.kern.RBF(1, active_dims=[0])
#     k.K(x)


def test_stationary_with_cat():
    np.random.seed(40)

    x_cont = np.random.rand(4, 3)
    x_cat = np.random.randint(0, 2, size=(4, 2))

    x = np.hstack((x_cont, x_cat))
    print(x_cat)
    print(x)
    k_rbf = GPy.kern.RBF(3, active_dims=[0, 1, 2])

    k = StationaryUniformCat(kernel=k_rbf, cat_dims=[3, 4])
    print(k)
    K_ = k.K(x, x[:-1, :])


def test_mixed_kernel_gradients():
    np.random.seed(42)
    n = 15

    x_cont = np.sort(np.random.rand(n, 3), 0)
    x_cat = np.random.randint(0, 4, size=(n, 2))
    # x_cat = np.arange(2*n).reshape(n ,2)

    x = np.hstack((x_cont, x_cat))
    print(x)

    y = 1 / x.shape[1] * np.sum(np.sin(4 * x), 1).reshape(-1,
                                                          1) + 0.01 * np.random.randn(
        len(x), 1)
    y = (y - np.mean(y)) / np.std(y)
    # print(x_cat)
    # print(x)
    k_rbf = GPy.kern.RBF(3, active_dims=[0, 1, 2])

    k = StationaryUniformCat(kernel=k_rbf, cat_dims=[3, 4])

    hp_bounds = np.array([[1., 1.],  # kernel variance
                          [1e-4, 3],  # lengthscale
                          # [1e-6, 1e6],  # likelihood variance
                          ])
    gp_opt_params = {'method': 'multigrad',
                     'num_restarts': 10,
                     'restart_bounds': hp_bounds,
                     # likelihood variance
                     'hp_bounds': hp_bounds,
                     'verbose': False}

    gp = GP(x, y, k, opt_params=gp_opt_params, lik_variance=0.1,
            lik_variance_fixed=True)
    print(gp)
    gp.optimize()
    print(gp)


def test_cont_cat_inputs():
    from testFunctions.syntheticFunctions import func1C, func2C, func1C1D

    np.random.seed(41)

    n = 100
    n_test = 500

    x_cont = np.sort(np.random.rand(n, 1) * 2 - 1, 0)
    x_cat = np.random.randint(0, 3, n).reshape(-1, 1)


    # x_cont = np.vstack((np.linspace(-1, 1, n).reshape(-1, 1),
    #                     np.linspace(-1, 1, n).reshape(-1, 1)))
    # x_cat = np.vstack((np.zeros((n, 1)), np.ones((n, 1))))

    x = np.hstack((x_cont, x_cat))

    x_cont_test = np.linspace(-1, 1, n_test).reshape(-1, 1)
    # x_cat_test = 0 * np.ones((n_test, 1))
    # x_cat_test = 1 * np.ones((n_test, 1))
    x_cat_test = 2 * np.ones((n_test, 1))
    # x_cat_test = np.arange(n_test).reshape(-1, 1)

    # x_test = np.hstack((x_cont_test, x_cat_test))
    x_test = np.hstack((x_cont_test, x_cat_test))
    # x_test = x
    y = np.zeros((len(x), 1))

    f = func1C1D

    for ii in range(len(x)):
        y[ii] = -f(x_cat[ii], x_cont[ii]) + 0.1* np.random.rand()

    # y = (y - np.mean(y))/np.std(y)
    # y_sum = y[:n] + y[n:]

    # plt.plot(x_cont, y, '*')
    # plt.show()

    k_rbf = GPy.kern.RBF(1, active_dims=[0])
    k = StationaryUniformCat(kernel=k_rbf, cat_dims=[1])

    hp_bounds = np.array([[1., 1.],  # kernel variance
                          [1e-4, 3],  # lengthscale
                          [1e-6, 100],  # likelihood variance
                          ])
    gp_opt_params = {'method': 'multigrad',
                     'num_restarts': 10,
                     'restart_bounds': hp_bounds,
                     # likelihood variance
                     'hp_bounds': hp_bounds,
                     'verbose': False}

    gp = GP(x, y, k, opt_params=gp_opt_params,
            # lik_variance=1, lik_variance_fixed=True,
            y_norm='meanstd')
    gp.optimize()

    mu, var = gp.predict(x_test)
    print(gp)


    min0 = np.min(gp.Y_raw[np.where(gp.X[:,-1] == 0)])
    min1 = np.min(gp.Y_raw[np.where(gp.X[:,-1] == 1)])
    min2 = np.min(gp.Y_raw[np.where(gp.X[:,-1] == 2)])

    acq_0 = EI(gp, min0)
    acq_1 = EI(gp, min1)
    acq_2 = EI(gp, min2)
    acq_random = EI(gp, np.min(gp.Y_raw))

    acq_0_vals = acq_0.evaluate(np.hstack((x_cont_test, 0*np.ones((n_test, 1)))))
    acq_1_vals = acq_1.evaluate(np.hstack((x_cont_test, 1*np.ones((n_test, 1)))))
    acq_2_vals = acq_2.evaluate(np.hstack((x_cont_test, 2*np.ones((n_test, 1)))))
    acq_random = acq_random.evaluate(np.hstack((x_cont_test, np.arange(n_test).reshape(-1, 1))))

    f, (ax1, ax2)= plt.subplots(2, 1)
    ax1.plot(x_cont_test, mu.flatten(), 'r*')
    ax1.plot(x_cont, y, '*')
    # plt.plot(x_cont[:n], y_sum, 'g*')

    ax2.plot(x_cont_test, acq_0_vals, label='acq0')
    ax2.plot(x_cont_test, acq_1_vals, label='acq1')
    ax2.plot(x_cont_test, acq_2_vals, label='acq2')
    ax2.plot(x_cont_test, acq_random, label='acq_r')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    # test_creation()
    # test_subspace_learning()
    # test_kernel_with_delta()
    # test_rbf_with_delta()
    # test_stationary_with_cat()
    # test_mixed_kernel_gradients()
    test_cont_cat_inputs()
