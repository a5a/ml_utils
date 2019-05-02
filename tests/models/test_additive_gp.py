import GPy
import matplotlib.pyplot as plt
import numdifftools as nd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from ml_utils.models.additive_gp import MixtureViaSumAndProduct, \
    CategoryOverlapKernel, \
    GPWithSomeFixedDimsAtStart, RBFCategoryOverlapKernel
# from bayesopt.acquisition import EI
from ml_utils.models.gp import GP


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


# def test_stationary_with_cat():
#     np.random.seed(40)
#
#     x_cont = np.random.rand(4, 3)
#     x_cat = np.random.randint(0, 2, size=(4, 2))
#
#     x = np.hstack((x_cont, x_cat))
#     print(x_cat)
#     print(x)
#     k_rbf = GPy.kern.RBF(3, active_dims=[0, 1, 2])
#
#     k = StationaryUniformCat(kernel=k_rbf, cat_dims=[3, 4])
#     print(k)
#     K_ = k.K(x, x[:-1, :])


# def test_mixed_kernel_gradients():
#     np.random.seed(42)
#     n = 15
#
#     x_cont = np.sort(np.random.rand(n, 3), 0)
#     x_cat = np.random.randint(0, 4, size=(n, 2))
#     # x_cat = np.arange(2*n).reshape(n ,2)
#
#     x = np.hstack((x_cont, x_cat))
#     print(x)
#
#     y = 1 / x.shape[1] * np.sum(np.sin(4 * x), 1).reshape(-1,
#                                                           1) + 0.01 * np.random.randn(
#         len(x), 1)
#     y = (y - np.mean(y)) / np.std(y)
#     # print(x_cat)
#     # print(x)
#     k_rbf = GPy.kern.RBF(3, active_dims=[0, 1, 2])
#
#     k = StationaryUniformCat(kernel=k_rbf, cat_dims=[3, 4])
#
#     hp_bounds = np.array([[1., 1.],  # kernel variance
#                           [1e-4, 3],  # lengthscale
#                           # [1e-6, 1e6],  # likelihood variance
#                           ])
#     gp_opt_params = {'method': 'multigrad',
#                      'num_restarts': 10,
#                      'restart_bounds': hp_bounds,
#                      # likelihood variance
#                      'hp_bounds': hp_bounds,
#                      'verbose': False}
#
#     gp = AdditiveGP(x, y, k, opt_params=gp_opt_params, lik_variance=0.1,
#                     lik_variance_fixed=True)
#     print(gp)
#     gp.optimize()
#     print(gp)


# def test_cont_cat_inputs():
#     from testFunctions.syntheticFunctions import func1C, func2C, func1C1D
#     from sklearn.preprocessing import OneHotEncoder
#     np.random.seed(41)
#
#     # magic numbers
#     n = 20
#     n_test = 500
#     n_cat = 15
#
#     # training data X
#     x_cont = np.sort(np.random.rand(n, 1) * 2 - 1, 0)
#     # x_cat = np.random.randint(0, n_cat, n).reshape(-1, 1)
#     x_cat = np.hstack([np.arange(n_cat)] * (int(n / n_cat) + 1))[:n].reshape(
#         -1, 1)
#     enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
#     enc.fit(x_cat)
#     x_cat_one_hot = enc.transform(x_cat)
#
#     x = np.hstack((x_cont, x_cat))
#     x_one_hot = np.hstack((x_cont, x_cat_one_hot))
#
#     # test data X
#     x_cont_test = np.linspace(-1, 1, n_test).reshape(-1, 1)
#     # x_cat_test = 0 * np.ones((n_test, 1))
#     # x_cat_test = 1 * np.ones((n_test, 1))
#     # x_cat_test = 2 * np.ones((n_test, 1))
#     # x_cat_test = np.arange(n_test).reshape(-1, 1) + 3
#
#     # x_cat_test = np.sort(np.random.randint(0, n_cat, n_test).reshape(-1, 1), 0)
#     x_cat_test = np.sort(
#         np.hstack([np.arange(n_cat)] * (int(n_test / n_cat) + 1))[
#         :n_test].reshape(-1, 1), 0)
#
#     x_cat_test_one_hot = enc.transform(x_cat_test)
#
#     # x_test = np.hstack((x_cont_test, x_cat_test))
#     x_test = np.hstack((x_cont_test, x_cat_test))
#     x_test_one_hot = np.hstack((x_cont_test, x_cat_test_one_hot))
#     # x_test = x
#
#     # Function values
#     y = np.zeros((len(x), 1))
#     f = func1C1D
#     for ii in range(len(x)):
#         y[ii] = -f(x_cat[ii], x_cont[ii]) + 0.1 * np.random.rand()
#
#     y = (y - np.mean(y)) / np.std(y)
#     # y_sum = y[:n] + y[n:]
#
#     # plt.plot(x_cont, y, '*')
#     # plt.show()
#
#     # Mixed kernel GP
#     k_rbf = GPy.kern.RBF(1, active_dims=[0])
#     k1 = StationaryUniformCat(kernel=k_rbf, cat_dims=[1])
#
#     hp_bounds = np.array([[1., 1.],  # kernel variance
#                           [1e-4, 3],  # lengthscale
#                           [1e-6, 100],  # likelihood variance
#                           ])
#     gp_opt_params = {'method': 'multigrad',
#                      'num_restarts': 10,
#                      'restart_bounds': hp_bounds,
#                      # likelihood variance
#                      'hp_bounds': hp_bounds,
#                      'verbose': False}
#
#     gp1 = AdditiveGP(x, y, k1, opt_params=gp_opt_params,
#                      # lik_variance=1, lik_variance_fixed=True,
#                      y_norm='meanstd')
#     gp1.optimize()
#
#     # Single kernel GP
#     k2 = GPy.kern.RBF(4)
#
#     hp_bounds = np.array([[1., 1.],  # kernel variance
#                           [1e-4, 3],  # lengthscale
#                           [1e-6, 100],  # likelihood variance
#                           ])
#     gp_opt_params = {'method': 'multigrad',
#                      'num_restarts': 10,
#                      'restart_bounds': hp_bounds,
#                      # likelihood variance
#                      'hp_bounds': hp_bounds,
#                      'verbose': False}
#
#     gp2 = GP(x_one_hot, y, k2, opt_params=gp_opt_params,
#              # lik_variance=1, lik_variance_fixed=True,
#              y_norm='meanstd')
#     gp2.optimize()
#
#     # Check outputs
#     # mu, var = gp.predict_latent_continuous(x_test)
#     mu1, var1 = gp1.predict(x_test)
#     mu1, var1 = mu1.flatten(), var1.flatten()
#     print(gp1)
#
#     mu2, var2 = gp2.predict(x_test_one_hot)
#     mu2, var2 = mu2.flatten(), var2.flatten()
#     print(gp2)
#
#     idx_0 = np.where(gp1.X[:, -1] == 0)
#     idx_1 = np.where(gp1.X[:, -1] == 1)
#     idx_2 = np.where(gp1.X[:, -1] == 2)
#
#     # Acquisition funcs
#     min0 = np.min(gp1.Y_raw[idx_0])
#     min1 = np.min(gp1.Y_raw[idx_1])
#     f, (ax1, ax2) = plt.subplots(2, 1)
#     ax1.plot(x_cont_test, mu1, 'r')
#     ax1.fill_between(x_cont_test.flatten(), mu1 - 2 * np.sqrt(var1),
#                      mu1 + 2 * np.sqrt(var1), color='r', alpha=0.2)
#     ax1.plot(x_cont[idx_0], y[idx_0], 'g*', label='0')
#     ax1.plot(x_cont[idx_1], y[idx_1], 'b*', label='1')
#     ax1.plot(x_cont[idx_2], y[idx_2], 'm*', label='2')
#     min2 = np.min(gp1.Y_raw[idx_2])
#
#     acq_0 = EI(gp1, min0)
#     acq_1 = EI(gp1, min1)
#     acq_2 = EI(gp1, min2)
#     acq_random = EI(gp1, np.min(gp1.Y_raw))
#
#     acq_0_vals = acq_0.evaluate(
#         np.hstack((x_cont_test, 0 * np.ones((n_test, 1)))))
#     acq_1_vals = acq_1.evaluate(
#         np.hstack((x_cont_test, 1 * np.ones((n_test, 1)))))
#     acq_2_vals = acq_2.evaluate(
#         np.hstack((x_cont_test, 2 * np.ones((n_test, 1)))))
#     acq_random = acq_random.evaluate(
#         np.hstack((x_cont_test, 10 + np.arange(n_test).reshape(-1, 1))))
#
#     f, (ax1, ax2) = plt.subplots(2, 1)
#     ax1.plot(x_cont_test, mu1, 'r')
#     ax1.fill_between(x_cont_test.flatten(), mu1 - 2 * np.sqrt(var1),
#                      mu1 + 2 * np.sqrt(var1), color='r', alpha=0.2)
#     ax1.plot(x_cont[idx_0], y[idx_0], 'g*', label='0')
#     ax1.plot(x_cont[idx_1], y[idx_1], 'b*', label='1')
#     ax1.plot(x_cont[idx_2], y[idx_2], 'm*', label='2')
#     # plt.plot(x_cont[:n], y_sum, 'g*')
#
#     ax2.plot(x_cont_test, acq_0_vals, label='acq0')
#     ax2.plot(x_cont_test, acq_1_vals, label='acq1')
#     ax2.plot(x_cont_test, acq_2_vals, label='acq2')
#     ax2.plot(x_cont_test, acq_random, label='acq_r')
#
#     plt.legend()
#     plt.show()
#
#     # Comparing one-hot to mixed-kernel GP
#
#     f, (ax1, ax2) = plt.subplots(2, 1)
#     ax1.plot(x_cont_test, mu1, 'r')
#     ax1.fill_between(x_cont_test.flatten(), mu1 - 2 * np.sqrt(var1),
#                      mu1 + 2 * np.sqrt(var1), color='r', alpha=0.2)
#     for ii in range(n_cat):
#         idx = np.where(gp1.X[:, -1] == ii)
#         ax1.plot(x_cont[idx], y[idx], '*', label=str(ii))
#
#     ax2.plot(x_cont_test, mu2, 'r')
#     ax2.fill_between(x_cont_test.flatten(), mu2 - 2 * np.sqrt(var2),
#                      mu2 + 2 * np.sqrt(var2), color='r', alpha=0.2)
#     for ii in range(n_cat):
#         idx = np.where(gp1.X[:, -1] == ii)
#         ax2.plot(x_cont[idx], y[idx], '*', label=str(ii))
#
#     plt.show()


# def test_subspace_pred():
#     n = 100
#
#     x = np.random.rand(n, 2)
#     x = np.hstack((x, np.random.randint(0, 2, (n, 2))))
#     y = np.sum(np.sin(x[:, :2]), 1).reshape(-1, 1)
#     y -= np.mean(y)
#
#     k_rbf = GPy.kern.RBF(2, active_dims=[0, 1])
#     k = StationaryUniformCat(kernel=k_rbf, cat_dims=[2, 3])
#
#     hp_bounds = np.array([[1., 1.],  # kernel variance
#                           [1e-4, 3],  # lengthscale
#                           [1e-6, 100],  # likelihood variance
#                           ])
#     gp_opt_params = {'method': 'multigrad',
#                      'num_restarts': 10,
#                      'restart_bounds': hp_bounds,
#                      # likelihood variance
#                      'hp_bounds': hp_bounds,
#                      'verbose': False}
#
#     gp = AdditiveGP(x, y, k, opt_params=gp_opt_params,
#                     # lik_variance=1, lik_variance_fixed=True,
#                     y_norm='meanstd')
#     gp.optimize()
#
#     mu1, var1 = gp.predict_latent_continuous(x)
#     mu2, var2 = gp.predict_latent(x, kern=gp.kern.kernel)
#
#     print(np.allclose(mu1, mu2))
#     print(np.allclose(var1, var2))

#
# def test_cont_cat_inputs_sin_linear_func():
#     from sklearn.preprocessing import OneHotEncoder
#
#     # f = sin_plus_linear
#     f = sin_plus_exp
#
#     # TRAIN
#     n = 100
#     x_cont = 2 * np.random.rand(n)[:, None] - 1
#     x_cat = np.random.randint(0, 2, n)[:, None]
#     x = np.hstack((x_cat, x_cont))
#
#     enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
#     enc.fit(x_cat)
#     x_cat_one_hot = enc.transform(x_cat)
#
#     x_one_hot = np.hstack((x_cat_one_hot, x_cont))
#
#     y = np.zeros(n)
#     for ii in range(n):
#         y[ii] = f(x_cat[ii], x_cont[ii])
#
#     y = y.reshape(-1, 1)
#     y += 0.1 * np.random.randn(*y.shape)
#
#     # TEST
#     n_test = 100
#     x_cont_test = np.sort(2 * np.random.rand(n_test)[:, None] - 1, 0)
#     x_cat_test = 1 * np.ones((n_test, 1))
#     x_test = np.hstack((x_cat_test, x_cont_test))
#
#     x_cat_one_hot_test = enc.transform(x_cat_test)
#
#     x_one_hot_test = np.hstack((x_cat_one_hot_test, x_cont_test))
#
#     y_test = np.zeros(n_test)
#     for ii in range(n_test):
#         y_test[ii] = f(x_cat_test[ii], x_cont_test[ii])
#
#     y_test = y_test.reshape(-1, 1)
#
#     # Mixed kernel GP
#     k_rbf = GPy.kern.RBF(1, active_dims=[0])
#     k1 = StationaryUniformCat(kernel=k_rbf, cat_dims=[1])
#
#     hp_bounds = np.array([[1., 1.],  # kernel variance
#                           [1e-4, 3],  # lengthscale
#                           [1e-6, 100],  # likelihood variance
#                           ])
#     gp_opt_params = {'method': 'multigrad',
#                      'num_restarts': 10,
#                      'restart_bounds': hp_bounds,
#                      # likelihood variance
#                      'hp_bounds': hp_bounds,
#                      'verbose': False}
#
#     gp1 = AdditiveGP(x, y, k1, opt_params=gp_opt_params,
#                      # lik_variance=1, lik_variance_fixed=True,
#                      y_norm='meanstd')
#     gp1.optimize()
#
#     # Single kernel GP
#     k2 = GPy.kern.RBF(3)
#
#     hp_bounds = np.array([[1., 1.],  # kernel variance
#                           [1e-4, 3],  # lengthscale
#                           [1e-6, 100],  # likelihood variance
#                           ])
#     gp_opt_params = {'method': 'multigrad',
#                      'num_restarts': 10,
#                      'restart_bounds': hp_bounds,
#                      # likelihood variance
#                      'hp_bounds': hp_bounds,
#                      'verbose': False}
#
#     gp2 = GP(x_one_hot, y, k2, opt_params=gp_opt_params,
#              # lik_variance=1, lik_variance_fixed=True,
#              y_norm='meanstd')
#     gp2.optimize()
#
#     # Check outputs
#     # mu, var = gp.predict_latent_continuous(x_test)
#     mu1, var1 = gp1.predict(x_test)
#     mu1, var1 = mu1.flatten(), var1.flatten()
#     print(gp1)
#
#     mu2, var2 = gp2.predict(x_one_hot_test)
#     mu2, var2 = mu2.flatten(), var2.flatten()
#     print(gp2)
#
#     f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
#     ax1.plot(x_cont_test, mu1, 'r-')
#     ax1.fill_between(x_cont_test.flatten(), mu1 - 2 * np.sqrt(var1),
#                      mu1 + 2 * np.sqrt(var1), color='r', alpha=0.2)
#     ax2.plot(x_cont_test, mu2, 'g-')
#     ax2.fill_between(x_cont_test.flatten(), mu2 - 2 * np.sqrt(var2),
#                      mu2 + 2 * np.sqrt(var2), color='g', alpha=0.2)
#
#     plt.show()


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
    # np.random.seed(1)
    n = 100
    x = np.sort(np.random.rand(n, 5), 0)
    y = np.sin(np.sum(3 * x, 1))
    y = (y.reshape(-1, 1) - np.mean(y)) / np.std(y)

    k1 = GPy.kern.RBF(3, active_dims=[0, 1, 2], ARD=True)
    k2 = GPy.kern.Matern52(2, active_dims=[3, 4], ARD=True)

    k = MixtureViaSumAndProduct(5, k1, k2, mix=0.5, fix_inner_variances=False)

    k3 = GPy.kern.RBF(3, active_dims=[0, 1, 2], ARD=True)
    k4 = GPy.kern.Matern52(2, active_dims=[3, 4], ARD=True)
    k_test = k3 + k4

    hp_bounds = np.array([[1e-4, 3],  # k1
                          [1e-4, 3],  # k2
                          [1e-4, 3],
                          [1e-4, 3],
                          [1e-4, 3],
                          [1e-4, 3],
                          [1e-4, 3],
                          [1e-6, 3],  # likelihood variance
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
    gp_test = GP(x, y, k_test, opt_params=gp_opt_params)
    print(gp)
    print(gp_test)
    print(gp.gradient)

    def ll(theta):
        old_theta = gp.param_array
        gp.param_array = theta
        loglik = gp.log_likelihood()
        gp.param_array = old_theta
        return loglik

    G = nd.Gradient(ll, step=0.01)(gp.param_array)
    print(G)

    print(gp_test.gradient)

    def ll(theta):
        old_theta = gp_test.param_array
        gp_test.param_array = theta
        loglik = gp_test.log_likelihood()
        gp_test.param_array = old_theta
        return loglik

    G = nd.Gradient(ll, step=0.01)(gp_test.param_array)
    print(G)
    print(np.allclose(gp.gradient, gp_test.gradient))

    gp.optimize()
    print(gp)

    Kxx = k.K(x, x)
    plt.imshow(Kxx)
    plt.show()


def test_kernel_mixture_learning():
    # np.random.seed(1)
    n = 100
    x = np.sort(np.random.rand(n, 5), 0)
    y = np.sin(np.sum(3 * x, 1))
    y = (y.reshape(-1, 1) - np.mean(y)) / np.std(y)

    k1 = GPy.kern.RBF(3, active_dims=[0, 1, 2], ARD=True)
    k2 = GPy.kern.Matern52(2, active_dims=[3, 4], ARD=True)

    k = MixtureViaSumAndProduct(5, k1, k2, mix=0.5, fix_inner_variances=False,
                                fix_mix=False)

    hp_bounds = np.array([
        [1e-6, 1],  # mix
        [1e-4, 3],  # k1
        [1e-4, 3],  # k2
        [1e-4, 3],
        [1e-4, 3],
        [1e-4, 3],
        [1e-4, 3],
        [1e-4, 3],
        [1e-6, 3],  # likelihood variance
    ])
    gp_opt_params = {'method': 'samplegrad',
                     'num_local': 3,
                     'num_samples': 500,
                     'hp_bounds': hp_bounds,
                     'verbose': False}

    # print(k)
    # print(k.param_array)

    gp = GP(x, y, k, opt_params=gp_opt_params)

    gp.optimize()
    print(gp)

    Kxx = k.K(x, x)
    plt.imshow(Kxx)
    plt.show()


def test_cat_kernel():
    n = 10
    x = np.random.randint(0, 2, (n, 5))

    k_cat = CategoryOverlapKernel(2, active_dims=[0, 1])

    print(x[:, :2])
    print(k_cat.K(x, x))
    print(k_cat)


def test_gp_with_fixed_dims():
    n_x = 50
    n_test = 20
    categorical_dims = [0, 1]
    continuous_dims = [2, 3, 4]
    x_var = np.random.randn(n_x, len(continuous_dims))
    x_fixed = np.random.randint(0, 2, (n_x, len(categorical_dims)))

    x = np.hstack((x_fixed, x_var))

    y = np.sum(np.sin(4 * x), 1)
    y = (y - np.mean(y)) / np.std(y)
    y = y.reshape(-1, 1)

    k_cont = GPy.kern.Matern52(len(continuous_dims) + len(categorical_dims),
                               ARD=False)
    gp1 = GP(x, y, k_cont)
    gp2 = GPWithSomeFixedDimsAtStart(x, y, k_cont, fixed_dim_vals=[1, 1])

    x_test_cont = np.random.randn(n_test, len(continuous_dims))
    x_test_fixed = np.vstack([[1, 1]] * n_test)

    x_test1 = np.hstack((
        x_test_fixed,
        x_test_cont
    ))

    x_test2 = x_test_cont

    mu1, var1 = gp1.predict(x_test1)
    mu2, var2 = gp2.predict(x_test2)

    print(np.allclose(mu1, mu2))
    print(np.allclose(var1, var2))


def test_extreme_case_for_mixture_kernel():
    # seed = 1
    # np.random.seed(seed)
    n = 33
    d_x = 30
    d_h = 3

    x = np.random.rand(n, d_x)
    h = np.random.randint(0, 200, (n, d_h))
    print(h)
    z = np.hstack((h, x))
    # y = np.sin(np.sum(3 * z, 1))
    # y = (y.reshape(-1, 1) - np.mean(y)) / np.std(y)

    k1 = CategoryOverlapKernel(d_h, active_dims=list(range(d_h)))
    k2 = GPy.kern.Matern52(d_x, active_dims=np.arange(d_h, d_x + d_h))

    k = MixtureViaSumAndProduct(d_x + d_h, k1, k2, mix=1.0,
                                fix_inner_variances=False)

    k_vals = k.K(z, z)

    plt.imshow(k_vals)
    plt.show()

    # print(z)
    print(k_vals)


def test_mixture_kernel_computations():
    seed = 1
    np.random.seed(seed)
    n = 20
    d_x = 3
    d_h = 3

    x = np.random.rand(n, d_x)
    h = np.random.randint(0, 20, (n, d_h))
    z = np.hstack((h, x))
    y = np.sin(np.sum(3 * z, 1))
    y = (y.reshape(-1, 1) - np.mean(y)) / np.std(y)

    k1 = CategoryOverlapKernel(d_h, active_dims=list(range(d_h)))
    k2 = GPy.kern.Matern52(d_x, active_dims=np.arange(d_h, d_x + d_h))

    # kernel values
    for mix in (0, 0.1, 0.5, 0.9, 1.):
        k = MixtureViaSumAndProduct(d_x + d_h, k1, k2, mix=mix,
                                    variance=0.8,
                                    fix_inner_variances=False,
                                    fix_variance=False)
        k_vals = k.K(z, z)

        k_vals_2 = 0.8 * (0.5 * (1 - mix) * (k1.K(z) + k2.K(z))
                   + mix * k1.K(z) * k2.K(z))

        print(f"mix = {mix} similar? "
              f"{np.allclose(k_vals, k_vals_2)}")
    # print("\nKernel values tested")

    # kernel gradient
    delta = 0.0001
    gradient = []
    gp = GP(z, y, k, lik_variance_fixed=True, lik_variance=1e-6)
    # perturb hps away from 1.
    gp.param_array = gp.param_array * np.random.rand(*gp.param_array.shape)
    curr_p = np.array(k.param_array)
    for ii in range(len(curr_p)):
        gp.param_array = curr_p

        delta_p = curr_p[ii] * delta

        p_plus = curr_p.copy()
        p_plus[ii] += delta_p
        p_minus = curr_p.copy()
        p_minus[ii] -= delta_p

        gp.param_array = p_plus
        k = gp.kern
        k_plus = k.K(z)
        gp.param_array = p_minus
        k = gp.kern
        k_minus = k.K(z)

        g = (k_plus - k_minus) / (2 * delta_p)

        gradient.append(np.sum(g))

    gp.param_array = curr_p

    gradient = np.array(gradient)

    k = gp.kern
    k.update_gradients_full(1., z, z)
    k_grad = k.gradient
    # print(gradient)
    # print(k_grad)
    # print(gp)

    print(f"kernel gradients wrt theta: "
          f"{np.allclose(gradient, k_grad, rtol=1e-4)}")


def test_rbf_cat_kernel():
    np.random.seed(2)
    C = [2, 4, 3]
    first_idx = [0, 2, 6]
    n_x = 5
    x = np.hstack([np.random.randint(0, c, (n_x, 1)) for c in C])

    x_oh = []
    for x_row in x:
        row = np.zeros((np.sum(C)))
        # print(row)
        for ii in range(len(x_row)):
            # print(ii)
            row[first_idx[ii]+x_row[ii]] = 1

        x_oh.append(row)

    x_oh = np.vstack(x_oh)

    k_ref = GPy.kern.RBF(np.sum(C))
    k_ref_vals = k_ref.K(x_oh)

    converter = OneHotEncoder(categories=([np.arange(c) for c in C]),
                              sparse=False)
    # Calling fit() makes the object usable
    converter.fit(np.zeros((1, len(C))))

    k_new = RBFCategoryOverlapKernel(C, converter,
                                     active_dims=np.arange(len(C)))  # cat
    k_new_vals = k_new.K(x)

    print(k_ref_vals - k_new_vals)



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
    # test_kernel_mixture_via_sum_and_product()
    # test_kernel_mixture_learning()
    # test_cat_kernel()
    # test_gp_with_fixed_dims()
    # test_extreme_case_for_mixture_kernel()
    # test_mixture_kernel_computations()
    test_rbf_cat_kernel()
