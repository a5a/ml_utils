import numpy as np
import GPy
from ml_utils.models import GP
from ml_utils.lipschitz import estimate_lipschitz_constant
import pylab as plt


def create_data_and_surrogate():
    x = np.arange(4).astype(float)[:, None]
    x_dense = np.linspace(np.min(x), np.max(x), 100)[:, None]

    y = 5 * np.cos(x) + np.random.randn(*x.shape)
    y -= np.mean(y)
    kern = GPy.kern.RBF(1)

    hp_bounds = np.array([[1e-6, 1e6],  # kernel variance
                          [1e-4, 1e3],  # lengthscale
                          # [1e-6, 1e6]  # likelihood variance
                          ])

    hyper_priors = [GPy.priors.Gamma(a=1.0, b=0.1),
                    # lengthscale prior keeps it small-ish
                    GPy.priors.Gamma(a=1.5, b=1.5),
                    # GPy.priors.Gamma(a=1.0, b=0.1)
                    ]

    gp_opt_params = {'method': 'multigrad',
                     'num_restarts': 10,
                     'restart_bounds': np.array([[1e-6, 10],  # kernel variance
                                                 [1e-4, 5],  # lengthscale
                                                 ]),
                     'hp_bounds': hp_bounds,
                     'verbose': False}

    surrogate = GP(x, y, kern, lik_variance=1e-3, lik_variance_fixed=True,
                   opt_params=gp_opt_params,
                   hyper_priors=hyper_priors)
    surrogate.optimize()

    return x, y, surrogate, x_dense


def test_lipschitz():
    x, y, surrogate, x_dense = create_data_and_surrogate()
    x_bounds = np.array([[np.min(x), np.max(x)]])
    dmu = surrogate.dmu_dx(x_dense).sum(-1)
    idx_highest_dmu = np.argmax(np.abs(dmu))
    highest_dmu_value = dmu[idx_highest_dmu]
    lipschitz_constant_from_grid_search = np.sqrt(highest_dmu_value ** 2)

    lipschitz_from_optimization = estimate_lipschitz_constant(surrogate,
                                                              x_bounds)

    print(f"lipschitz_constant_from_grid_search = "
          f"{lipschitz_constant_from_grid_search}")
    print(f"lipschitz_from_optimization = "
          f"{lipschitz_from_optimization}")

    # mu = surrogate.predict(x_dense)[0]
    # plt.plot(mu, label='mu')
    # plt.plot(dmu, label='dmu')
    # plt.plot(np.arange(len(mu)), np.zeros(len(mu)), 'k')
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    test_lipschitz()
