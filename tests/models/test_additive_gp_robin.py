import GPy
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
import numpy as np
from ml_utils.models.gp import GP
from ml_utils.models.additive_gp import AdditiveGP, \
    StationaryUniformCat, MixtureViaSumAndProduct, CategoryOverlapKernel
from ml_utils.optimization import sample_then_minimize
from testFunctions.syntheticFunctions import sin_plus_linear, sin_plus_exp
from sklearn.preprocessing import OneHotEncoder


def generate_x_y(f, n, cat_ub=2, x_cat=None):
    np.random.seed(42)
    # Cat + Cont Inputs
    x_cont = 2 * np.random.rand(n)[:, None] - 1

    if x_cat is not None:
        x_cont = np.sort(x_cont, 0)
    else:
        x_cat = np.random.randint(0, cat_ub, n)[:, None]

    x = np.hstack((x_cat, x_cont))

    # Cat(One Hot)+ Cont Inputs
    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    enc.fit(np.random.randint(0, cat_ub, n)[:, None])
    x_cat_one_hot = enc.transform(x_cat)
    x_one_hot = np.hstack((x_cat_one_hot, x_cont))

    # Generate Outputs
    y = np.zeros(n)
    for ii in range(n):
        y[ii] = f(x_cat[ii], x_cont[ii])

    y = y.reshape(-1, 1)
    y += 0.05 * np.random.randn(*y.shape)

    return x_cont, x, x_one_hot, y


def test_regression_with_cont_cat_inputs(f, n_ob=50, n_test=100, cat_ub=2):
    # TRAIN
    x_cont_ob, x_ob, x_one_hot_ob, y_ob = generate_x_y(f, n_ob, cat_ub=cat_ub)
    # y_ob = (y_ob - np.mean(y_ob))/np.std(y_ob)

    # TEST
    x_cat_test = 1 * np.ones((n_test, 1))
    x_cont_test, x_test, x_one_hot_test, y_test = generate_x_y(f, n_test,
                                                               cat_ub=cat_ub,
                                                               x_cat=x_cat_test)

    # ---- Define GP models  ---- #
    # # Mixed kernel GP
    # k_rbf = GPy.kern.RBF(1, active_dims=[1])
    # k1 = StationaryUniformCat(kernel=k_rbf, cat_dims=[0])
    #
    # hp_bounds = np.array([[1., 1.],  # kernel variance
    #                       [1e-4, 3],  # lengthscale
    #                       [1e-6, 100],  # likelihood variance
    #                       ])
    # gp_opt_params = {'method': 'multigrad',
    #                  'num_restarts': 10,
    #                  'restart_bounds': hp_bounds,
    #                  # likelihood variance
    #                  'hp_bounds': hp_bounds,
    #                  'verbose': False}
    #
    # gp1 = AdditiveGP(x_ob, y_ob, k1, opt_params=gp_opt_params,
    #                  y_norm='meanstd')
    # gp1.optimize()

    hp_bounds = np.array([
        # [1., 1.],  # kernel variance
        [1e-4, 3],  # lengthscale
        [1e-6, 1],  # likelihood variance
    ])
    gp_opt_params = {'method': 'multigrad',
                     'num_restarts': 10,
                     'restart_bounds': hp_bounds,
                     # likelihood variance
                     'hp_bounds': hp_bounds,
                     'verbose': False}

    hp_bounds2 = np.array([
        [1., 1.],  # kernel variance
        [1e-4, 3],  # lengthscale
        [1e-6, 1],  # likelihood variance
    ])
    gp_opt_params2 = {'method': 'multigrad',
                      'num_restarts': 10,
                      'restart_bounds': hp_bounds2,
                      # likelihood variance
                      'hp_bounds': hp_bounds2,
                      'verbose': False}

    # SumProduct kernel
    # k_cat = GPy.kern.RBF(1, active_dims=[0])  # cat
    k_cat = CategoryOverlapKernel(1, active_dims=[0])  # cat
    k_cont = GPy.kern.RBF(1, active_dims=[1])  # cont

    k = MixtureViaSumAndProduct(2, k_cat, k_cont, mix=0.5, fix_variances=True)

    gp1 = GP(x_ob, y_ob, k, opt_params=gp_opt_params, y_norm='meanstd')
    print(gp1)
    gp1.optimize()
    # def f(x):
    #     gp1.param_array = x
    #     return -gp1.log_likelihood()

    # res = sample_then_minimize(
    #     f,
    #     hp_bounds,
    #     num_samples=1000,
    #     num_local=2,
    #     evaluate_sequentially=True,
    #     verbose=False)
    #
    # gp1.param_array = res.x

    # Single kernel one hot GP
    k_cont = GPy.kern.RBF(3)
    gp2 = GP(x_one_hot_ob, y_ob, k_cont, opt_params=gp_opt_params2,
             # lik_variance=1, lik_variance_fixed=True,
             y_norm='meanstd')
    gp2.optimize()

    # ---- Check GP predictions  ---- #
    # mu, var = gp.predict_latent_continuous(x_test)
    mu1, var1 = gp1.predict(x_test)
    mu1, var1 = mu1.flatten(), var1.flatten()
    print(gp1)

    mu2, var2 = gp2.predict(x_one_hot_test)
    mu2, var2 = mu2.flatten(), var2.flatten()
    print(gp2)

    # ---- Plots results  ---- #
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    axes = [ax1, ax2]
    mu = [mu1, mu2]
    var = [var1, var2]
    color = ['b', 'g']
    label = ['addgp', 'gp']
    for i in range(len(axes)):
        axes[i].fill_between(x_cont_test.flatten(),
                             mu[i] - 2 * np.sqrt(var[i]),
                             mu[i] + 2 * np.sqrt(var[i]), color=color[i],
                             alpha=0.2)
        axes[i].plot(x_cont_test, mu[i], color[i] + '-', label=label[i])
        axes[i].plot(x_cont_test, y_test, 'k--', label='true')
        axes[i].plot(x_cont_ob, y_ob, 'rx', label='true')
        axes[i].legend()

    plt.show()
    # plt.savefig("sin_plus_exp.pdf", bbox_inches='tight')


if __name__ == '__main__':
    test_regression_with_cont_cat_inputs(f=sin_plus_exp)
