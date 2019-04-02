from pprint import pprint

import GPy
import matplotlib.pyplot as plt
import numpy as np

from bayesopt.acquisition import EI
from ml_utils.models.gp import GP
from ml_utils.models.additive_gp import AdditiveGP, \
    StationaryUniformCat, MixtureViaSumAndProduct

#
#
#
# def test_subspace_learning():
#     x1 = np.random.rand(20)[:, None]
#     x2 = np.random.rand(20)[:, None]
#     x1 = np.hstack((x1, 0.01 * np.random.rand(*x1.shape)))
#     x2 = np.hstack((0.01 * np.random.rand(*x2.shape), x2))
#
#     a = 7
#
#     def f(x):
#         return np.sum(np.sin(a * x), 1)
#
#     x_ob = np.vstack((x1, x2))
#     y_ob = f(x_ob).reshape(-1, 1)
#
#     # ------ Test grid -------- #
#     x1, x2 = np.mgrid[0:1:50j, 0:1:50j]
#     X = np.vstack((x1.flatten(), x2.flatten())).T
#     y = f(X)
#     y -= np.mean(y)
#
#     x_test = np.hstack([np.linspace(0, 1, 100).reshape(-1, 1)] * 2)
#
#     k = create_additive_kernel(GPy.kern.RBF, GPy.kern.RBF, [0], [1])
#     gp = AdditiveGP(x_ob, y_ob, k)
#
#     gp.optimize()
#
#     # x_test = np.random.rand(1000, 2)
#     mu0, var0 = gp.predict_latent_subspace(x_test, 0)
#     mu1, var1 = gp.predict_latent_subspace(x_test, 1)
#
#     true_values = f(x_test)
#
#     # print(np.abs(mu-true_values))
#
#     f, (ax1, ax2) = plt.subplots(2, 1)
#     ax1.plot(x_test[:, 0], mu0.flatten())
#     ax1.plot(x_test[:, 0], np.sin(a * x_test[:, 0]))
#     ax2.plot(x_test[:, 1], mu1.flatten())
#     ax2.plot(x_test[:, 1], np.sin(a * x_test[:, 1]))
#
#     plt.show()
#
#     mu, var = gp.predict(X)
#
#     # # ------ Plot figures -------- #
#     figure, axes = plt.subplots(2, 1, figsize=(10, 10))
#     sub1 = axes[0].contourf(x1, x2, y.reshape(50, 50))
#     axes[0].plot(x_ob[:, 0], x_ob[:, 1], 'r ^')
#     axes[0].set_title('objective func')
#
#     sub2 = axes[1].contourf(x1, x2, mu.reshape(50, 50))
#     axes[1].plot(x_ob[:, 0], x_ob[:, 1], 'r ^')
#     axes[1].set_title('prediction')
#     plt.show()


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
from testFunctions.syntheticFunctions import sin_plus_linear, sin_plus_exp


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

    gp = AdditiveGP(x, y, k, opt_params=gp_opt_params, lik_variance=0.1,
                    lik_variance_fixed=True)
    print(gp)
    gp.optimize()
    print(gp)


def test_cont_cat_inputs():
    from testFunctions.syntheticFunctions import func1C, func2C, func1C1D
    from sklearn.preprocessing import OneHotEncoder
    np.random.seed(41)

    # magic numbers
    n = 20
    n_test = 500
    n_cat = 15

    # training data X
    x_cont = np.sort(np.random.rand(n, 1) * 2 - 1, 0)
    # x_cat = np.random.randint(0, n_cat, n).reshape(-1, 1)
    x_cat = np.hstack([np.arange(n_cat)] * (int(n / n_cat) + 1))[:n].reshape(
        -1, 1)
    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    enc.fit(x_cat)
    x_cat_one_hot = enc.transform(x_cat)

    x = np.hstack((x_cont, x_cat))
    x_one_hot = np.hstack((x_cont, x_cat_one_hot))

    # test data X
    x_cont_test = np.linspace(-1, 1, n_test).reshape(-1, 1)
    # x_cat_test = 0 * np.ones((n_test, 1))
    # x_cat_test = 1 * np.ones((n_test, 1))
    # x_cat_test = 2 * np.ones((n_test, 1))
    # x_cat_test = np.arange(n_test).reshape(-1, 1) + 3

    # x_cat_test = np.sort(np.random.randint(0, n_cat, n_test).reshape(-1, 1), 0)
    x_cat_test = np.sort(
        np.hstack([np.arange(n_cat)] * (int(n_test / n_cat) + 1))[
        :n_test].reshape(-1, 1), 0)

    x_cat_test_one_hot = enc.transform(x_cat_test)

    # x_test = np.hstack((x_cont_test, x_cat_test))
    x_test = np.hstack((x_cont_test, x_cat_test))
    x_test_one_hot = np.hstack((x_cont_test, x_cat_test_one_hot))
    # x_test = x

    # Function values
    y = np.zeros((len(x), 1))
    f = func1C1D
    for ii in range(len(x)):
        y[ii] = -f(x_cat[ii], x_cont[ii]) + 0.1 * np.random.rand()

    y = (y - np.mean(y)) / np.std(y)
    # y_sum = y[:n] + y[n:]

    # plt.plot(x_cont, y, '*')
    # plt.show()

    # Mixed kernel GP
    k_rbf = GPy.kern.RBF(1, active_dims=[0])
    k1 = StationaryUniformCat(kernel=k_rbf, cat_dims=[1])

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

    gp1 = AdditiveGP(x, y, k1, opt_params=gp_opt_params,
                     # lik_variance=1, lik_variance_fixed=True,
                     y_norm='meanstd')
    gp1.optimize()

    # Single kernel GP
    k2 = GPy.kern.RBF(4)

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

    gp2 = GP(x_one_hot, y, k2, opt_params=gp_opt_params,
             # lik_variance=1, lik_variance_fixed=True,
             y_norm='meanstd')
    gp2.optimize()

    # Check outputs
    # mu, var = gp.predict_latent_continuous(x_test)
    mu1, var1 = gp1.predict(x_test)
    mu1, var1 = mu1.flatten(), var1.flatten()
    print(gp1)

    mu2, var2 = gp2.predict(x_test_one_hot)
    mu2, var2 = mu2.flatten(), var2.flatten()
    print(gp2)

    idx_0 = np.where(gp1.X[:, -1] == 0)
    idx_1 = np.where(gp1.X[:, -1] == 1)
    idx_2 = np.where(gp1.X[:, -1] == 2)

    # Acquisition funcs
    min0 = np.min(gp1.Y_raw[idx_0])
    min1 = np.min(gp1.Y_raw[idx_1])
    f, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(x_cont_test, mu1, 'r')
    ax1.fill_between(x_cont_test.flatten(), mu1 - 2 * np.sqrt(var1),
                     mu1 + 2 * np.sqrt(var1), color='r', alpha=0.2)
    ax1.plot(x_cont[idx_0], y[idx_0], 'g*', label='0')
    ax1.plot(x_cont[idx_1], y[idx_1], 'b*', label='1')
    ax1.plot(x_cont[idx_2], y[idx_2], 'm*', label='2')
    min2 = np.min(gp1.Y_raw[idx_2])

    acq_0 = EI(gp1, min0)
    acq_1 = EI(gp1, min1)
    acq_2 = EI(gp1, min2)
    acq_random = EI(gp1, np.min(gp1.Y_raw))

    acq_0_vals = acq_0.evaluate(
        np.hstack((x_cont_test, 0 * np.ones((n_test, 1)))))
    acq_1_vals = acq_1.evaluate(
        np.hstack((x_cont_test, 1 * np.ones((n_test, 1)))))
    acq_2_vals = acq_2.evaluate(
        np.hstack((x_cont_test, 2 * np.ones((n_test, 1)))))
    acq_random = acq_random.evaluate(
        np.hstack((x_cont_test, 10 + np.arange(n_test).reshape(-1, 1))))

    f, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(x_cont_test, mu1, 'r')
    ax1.fill_between(x_cont_test.flatten(), mu1 - 2 * np.sqrt(var1),
                     mu1 + 2 * np.sqrt(var1), color='r', alpha=0.2)
    ax1.plot(x_cont[idx_0], y[idx_0], 'g*', label='0')
    ax1.plot(x_cont[idx_1], y[idx_1], 'b*', label='1')
    ax1.plot(x_cont[idx_2], y[idx_2], 'm*', label='2')
    # plt.plot(x_cont[:n], y_sum, 'g*')

    ax2.plot(x_cont_test, acq_0_vals, label='acq0')
    ax2.plot(x_cont_test, acq_1_vals, label='acq1')
    ax2.plot(x_cont_test, acq_2_vals, label='acq2')
    ax2.plot(x_cont_test, acq_random, label='acq_r')

    plt.legend()
    plt.show()

    # Comparing one-hot to mixed-kernel GP

    f, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(x_cont_test, mu1, 'r')
    ax1.fill_between(x_cont_test.flatten(), mu1 - 2 * np.sqrt(var1),
                     mu1 + 2 * np.sqrt(var1), color='r', alpha=0.2)
    for ii in range(n_cat):
        idx = np.where(gp1.X[:, -1] == ii)
        ax1.plot(x_cont[idx], y[idx], '*', label=str(ii))

    ax2.plot(x_cont_test, mu2, 'r')
    ax2.fill_between(x_cont_test.flatten(), mu2 - 2 * np.sqrt(var2),
                     mu2 + 2 * np.sqrt(var2), color='r', alpha=0.2)
    for ii in range(n_cat):
        idx = np.where(gp1.X[:, -1] == ii)
        ax2.plot(x_cont[idx], y[idx], '*', label=str(ii))

    plt.show()


def test_subspace_pred():
    n = 100

    x = np.random.rand(n, 2)
    x = np.hstack((x, np.random.randint(0, 2, (n, 2))))
    y = np.sum(np.sin(x[:, :2]), 1).reshape(-1, 1)
    y -= np.mean(y)

    k_rbf = GPy.kern.RBF(2, active_dims=[0, 1])
    k = StationaryUniformCat(kernel=k_rbf, cat_dims=[2, 3])

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

    gp = AdditiveGP(x, y, k, opt_params=gp_opt_params,
                    # lik_variance=1, lik_variance_fixed=True,
                    y_norm='meanstd')
    gp.optimize()

    mu1, var1 = gp.predict_latent_continuous(x)
    mu2, var2 = gp.predict_latent(x, kern=gp.kern.kernel)

    print(np.allclose(mu1, mu2))
    print(np.allclose(var1, var2))


def test_cont_cat_inputs_sin_linear_func():
    from sklearn.preprocessing import OneHotEncoder

    # f = sin_plus_linear
    f = sin_plus_exp

    # TRAIN
    n = 100
    x_cont = 2 * np.random.rand(n)[:, None] - 1
    x_cat = np.random.randint(0, 2, n)[:, None]
    x = np.hstack((x_cat, x_cont))

    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    enc.fit(x_cat)
    x_cat_one_hot = enc.transform(x_cat)

    x_one_hot = np.hstack((x_cat_one_hot, x_cont))

    y = np.zeros(n)
    for ii in range(n):
        y[ii] = f(x_cat[ii], x_cont[ii])

    y = y.reshape(-1, 1)
    y += 0.1 * np.random.randn(*y.shape)

    # TEST
    n_test = 100
    x_cont_test = np.sort(2 * np.random.rand(n_test)[:, None] - 1, 0)
    x_cat_test = 1 * np.ones((n_test, 1))
    x_test = np.hstack((x_cat_test, x_cont_test))

    x_cat_one_hot_test = enc.transform(x_cat_test)

    x_one_hot_test = np.hstack((x_cat_one_hot_test, x_cont_test))

    y_test = np.zeros(n_test)
    for ii in range(n_test):
        y_test[ii] = f(x_cat_test[ii], x_cont_test[ii])

    y_test = y_test.reshape(-1, 1)

    # Mixed kernel GP
    k_rbf = GPy.kern.RBF(1, active_dims=[0])
    k1 = StationaryUniformCat(kernel=k_rbf, cat_dims=[1])

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

    gp1 = AdditiveGP(x, y, k1, opt_params=gp_opt_params,
                     # lik_variance=1, lik_variance_fixed=True,
                     y_norm='meanstd')
    gp1.optimize()

    # Single kernel GP
    k2 = GPy.kern.RBF(3)

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

    gp2 = GP(x_one_hot, y, k2, opt_params=gp_opt_params,
             # lik_variance=1, lik_variance_fixed=True,
             y_norm='meanstd')
    gp2.optimize()

    # Check outputs
    # mu, var = gp.predict_latent_continuous(x_test)
    mu1, var1 = gp1.predict(x_test)
    mu1, var1 = mu1.flatten(), var1.flatten()
    print(gp1)

    mu2, var2 = gp2.predict(x_one_hot_test)
    mu2, var2 = mu2.flatten(), var2.flatten()
    print(gp2)

    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    ax1.plot(x_cont_test, mu1, 'r-')
    ax1.fill_between(x_cont_test.flatten(), mu1 - 2 * np.sqrt(var1),
                     mu1 + 2 * np.sqrt(var1), color='r', alpha=0.2)
    ax2.plot(x_cont_test, mu2, 'g-')
    ax2.fill_between(x_cont_test.flatten(), mu2 - 2 * np.sqrt(var2),
                     mu2 + 2 * np.sqrt(var2), color='g', alpha=0.2)

    plt.show()


def test_combination_kernel_hps():
    n = 100
    x = np.random.rand(n, 5)
    y = np.sin(np.sum(3 * x, 1))
    y = (y.reshape(-1, 1) - np.mean(y)) / np.std(y)

    k1 = GPy.kern.RBF(3, active_dims=[0, 1, 2])
    k2 = GPy.kern.Matern52(2, active_dims=[3, 4])

    k = k1 + k2
    print(k)
    k = k + k1 * k2
    print(k)

    gp = GP(x, y, k)
    gp.optimize()
    print(gp)


def test_kernel_mixture_via_sum_and_product():
    np.random.seed(1)
    n = 100
    x = np.sort(np.random.rand(n, 5), 0)
    y = np.sin(np.sum(3 * x, 1))
    y = (y.reshape(-1, 1) - np.mean(y)) / np.std(y)

    k1 = GPy.kern.RBF(3, active_dims=[0, 1, 2])
    k2 = GPy.kern.Matern52(2, active_dims=[3, 4])

    k = MixtureViaSumAndProduct(5, k1, k2, mix=0.5, fix_variances=True)

    hp_bounds = np.array([[1e-4, 3],  # k1
                          [1e-4, 3],  # k2
                          [1e-6, 100],  # likelihood variance
                          ])
    gp_opt_params = {'method': 'multigrad',
                     'num_restarts': 10,
                     'restart_bounds': hp_bounds,
                     # likelihood variance
                     'hp_bounds': hp_bounds,
                     'verbose': False}

    # print(k)
    # print(k.param_array)

    gp = GP(x, y, k, opt_params=gp_opt_params)
    print(gp)
    gp.optimize()
    print(gp)

    Kxx = k.K(x, x)
    plt.imshow(Kxx)
    plt.show()


if __name__ == '__main__':
    # test_creation()
    # test_subspace_learning()
    # test_kernel_with_delta()
    # test_rbf_with_delta()
    # test_stationary_with_cat()
    # test_mixed_kernel_gradients()
    # test_cont_cat_inputs()
    # test_subspace_pred()
    # test_cont_cat_inputs_sin_linear_func()
    # test_combination_kernel_hps()
    test_kernel_mixture_via_sum_and_product()
