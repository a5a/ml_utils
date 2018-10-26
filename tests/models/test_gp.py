import numpy as np
from ml_utils.models import GP
import GPy


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

if __name__ == '__main__':
    test_predictive_gradients()
    test_optimization_doesnt_crash()
