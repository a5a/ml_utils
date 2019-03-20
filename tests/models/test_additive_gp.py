from pprint import pprint

import GPy
import matplotlib.pyplot as plt
import numpy as np

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


def test_kernel_with_delta():
    k_continuous = GPy.kern.RBF(1, active_dims=[0])
    k = KernelWithDelta(k_continuous, [1])
    print(k.lengthscale)
    # DOESN'T WORK :(


def test_rbf_with_delta():
    x = np.arange(10).reshape(-1, 1)
    x = np.hstack((x, x))

    k = RBFWithDelta([0], [1])
    k = GPy.kern.RBF(1, active_dims=[0])
    k.K(x)


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
    np.random.seed(40)

    x_cont = np.random.rand(4, 3)
    x_cat = np.random.randint(0, 2, size=(4, 2))

    x = np.hstack((x_cont, x_cat))

    y = np.sum(np.sin(x), 1).reshape(-1, 1)
    # print(x_cat)
    print(x)
    k_rbf = GPy.kern.RBF(3, active_dims=[0, 1, 2])

    k = StationaryUniformCat(kernel=k_rbf, cat_dims=[3, 4])

    gp = GP(x, y, k)
    gp.optimize()


if __name__ == '__main__':
    # test_creation()
    # test_subspace_learning()
    # test_kernel_with_delta()
    # test_rbf_with_delta()
    # test_stationary_with_cat()
    test_mixed_kernel_gradients()
