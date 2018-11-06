import numpy as np
from ml_utils.models import GP
import GPy
import pylab as plt


def create_data_and_surrogate():
    x = np.arange(4).astype(float)[:, None]
    x_dense = np.linspace(np.min(x), np.max(x), 100)[:, None]

    y = 5 * np.sin(x)
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


def test_predictive_gradients():
    x = np.arange(5)[:, None]
    y = np.sin(x)
    y -= np.mean(y)

    kern = GPy.kern.RBF(1)

    mygp = GP(x, y, kern, lik_variance=1.)
    gpygp = GPy.models.GPRegression(x, y, kern)

    dmu_dx1 = mygp.dposterior_dx(x)[0]
    dmu_dx2 = gpygp.predictive_gradients(x)[0]
    dvar_dx1 = mygp.dposterior_dx(x)[1]
    dvar_dx2 = gpygp.predictive_gradients(x)[1]

    mu1, var1 = mygp.predict(x)
    mu2, var2 = gpygp.predict(x)

    assert np.allclose(mu1, mu2)
    assert np.allclose(var1, var2)
    assert np.allclose(dmu_dx1, dmu_dx2)
    assert np.allclose(dvar_dx1, dvar_dx2)


def test_optimization_doesnt_crash():
    x = np.arange(5)[:, None]
    y = np.sin(x)
    y -= np.mean(y)

    kern = GPy.kern.RBF(1)
    mygp = GP(x, y, kern, lik_variance=1.)
    mygp.optimize()


def test_dmu_dx():
    x, y, surrogate, x_dense = create_data_and_surrogate()
    mu = surrogate.predict(x_dense)[0]
    dmu = surrogate.dmu_dx(x_dense).sum(-1)
    plt.plot(mu, label='mu')
    plt.plot(dmu, label='dmu')
    plt.plot(np.arange(len(mu)), np.zeros(len(mu)), 'k')
    plt.legend()
    plt.show()


def test_ard():
    x = np.hstack([np.arange(4).astype(float)[:, None]]*2)

    y = 5 * np.sin(x).sum(-1).reshape(-1, 1)
    y -= np.mean(y)
    kern = GPy.kern.RBF(2, ARD=True)

    hp_bounds = np.array([[1e-6, 1e6],  # kernel variance
                          *[[1e-4, 1e3]]*x.shape[1],  # lengthscale
                          # [1e-6, 1e6]  # likelihood variance
                          ])

    hyper_priors = [GPy.priors.Gamma(a=1.0, b=0.1),
                    # lengthscale prior keeps it small-ish
                    *[GPy.priors.Gamma(a=1.5, b=1.5)]*x.shape[1],
                    # GPy.priors.Gamma(a=1.0, b=0.1)
                    ]

    gp_opt_params = {'method': 'multigrad',
                     'num_restarts': 10,
                     'restart_bounds': np.array([[1e-6, 10],  # kernel variance
                                                 *[[1e-4, 5]]*x.shape[1],  # lengthscale
                                                 ]),
                     'hp_bounds': hp_bounds,
                     'verbose': False}

    surrogate = GP(x, y, kern, lik_variance=1e-3, lik_variance_fixed=True,
                   opt_params=gp_opt_params,
                   hyper_priors=hyper_priors)
    surrogate.optimize()





if __name__ == '__main__':
    # test_predictive_gradients()
    # test_optimization_doesnt_crash()
    # test_dmu_dx()
    test_ard()
