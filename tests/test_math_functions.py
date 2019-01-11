from ml_utils.math_functions import get_function, gsobol, plot_test_func
from ml_utils.optimization import minimize_with_restarts
import numpy as np
import pylab as plt


def test_x_lim_dims():
    funcs = ('twosines', 'ackley',)
    dims = np.arange(1, 20)
    for dim in dims:
        _, X_LIM, _, _ = get_function(f'twosines-{dim}d')
        assert len(X_LIM) == dim


def test_get_function():
    name = 'gsobol-4d'

    f, X_LIM, min_loc, min_val = get_function(name)

    assert f.__name__ == name


def test_gsobol():
    x = np.linspace(-1, 1).reshape(-1, 1)
    x_in = np.hstack((x,x,x,x,x,x,x,x,x,x,x,x,x,))
    print(gsobol(np.array([0.1]*x_in.shape[1])))
    y = gsobol(x_in)
    plt.plot(x.flatten(), y.flatten())
    plt.show()

def test_matern():
    # plot_test_func('matern-2d')
    # plt.show()
    f, X_LIM, min_loc, min_val = get_function('matern-2d')
    print(f(np.array([0., 0.])))
    pass


if __name__ == '__main__':
    # test_x_lim_dims()
    # test_get_function()
    # test_gsobol()
    test_matern()