from pprint import pprint

import GPy
from typing import Optional, List, Union

# from scipy.spatial.distance import pdist
from ml_utils.models import GP
import numpy as np


class AdditiveGP(GP):
    def predict_latent_subspace(self, x, subspace_idx, full_cov: bool = False):
        """
        Predict the latent function in the selected subspace
        """
        kernel = self.kern.parameters[subspace_idx]
        mu = kernel.K(x, self.X) @ self.alpha
        var = kernel.K(x, x) - \
            kernel.K(x, self.X) @ self.Ka_inv @ kernel.K(self.X, x)

        if not full_cov:
            var = np.diag(var).reshape(mu.shape)
        return mu, var


def create_additive_kernel(k1_class, k2_class,
                           active_dims1, active_dims2,
                           k1_args=None, k2_args=None):
    """
    Creates additive kernel of kern_type + delta
    """
    assert active_dims1 is not None
    assert active_dims2 is not None

    if k1_args is None:
        k1_args = {}

    if k2_args is None:
        k2_args = {}

    k1 = k1_class(len(active_dims1), active_dims=active_dims1, **k1_args)
    k2 = k2_class(len(active_dims2), active_dims=active_dims2, **k2_args)

    k_combined = k1 + k2

    return k_combined


# class KernelWithDelta():
#     """
#     Recursion depth errors....
#     """
#
#     def __init__(self, kern, delta_dims):
#         self.delta_dims = delta_dims
#         self.kern = kern
#
#     def K(self, X, X0=None):
#         # todo
#         raise NotImplementedError
#
#     def __getattr__(self, name):
#         """
#         This redirects any methods that are not explicitly set
#         to self.original_model.
#
#         This function is only called if the called attribute
#         doesn't already exist.
#
#         If original_model.name is a method, then it will be run.
#         Otherwise the element will just be accessed.
#         """
#         if self.verbose:
#             print("Calling function '{}' of embedded model".format(name))
#         if callable(getattr(self.kern, name)):
#             try:
#                 return getattr(self.kern, name)(*args, **kwargs)
#             except NameError:  # no args provided
#                 return getattr(self.kern, name)()
#         else:
#             return getattr(self.kern, name)
#
#
# class RBFWithDelta(GPy.kern.RBF):
#     def __init__(self, active_dims_k=None, active_dims_delta=None,
#                  name='rbfdelta', **kwargs):
#         self.delta_dims = active_dims_delta
#         input_dim = len(active_dims_k)
#         super().__init__(input_dim, active_dims=active_dims_k, name=name,
#                          **kwargs)
#
#     def K(self, X, X2=None):
#         if X2 is None:
#             X2 = X.copy()
#
#         kernel_K = super().K(X, X2)
#
#         X_delta = X[:, self.delta_dims]
#         X2_delta = X2[:, self.delta_dims]
#
#         distances = pdist(X, X2)


class StationaryUniformCat(GPy.kern.Kern):
    """
    Kernel that is a combination of a stationary kernel and a
    categorical kernel. Each cat input has the same weight and is
    the cat variables' contribution to K() is normalised to [0, 1]

    TODO: Convert into GPy.kern.Kern class for optimisation purposes
    TODO: Need access to parameters (variance, lengthscale)
    TODO: Deal with gradients
    """

    def __init__(self, kernel: GPy.kern.RBF, cat_dims):
        self.cat_dims = cat_dims
        self.kernel = kernel

        name = 'StationaryUniformCat'
        input_dim = self.kernel.input_dim + len(self.cat_dims)
        active_dim = np.arange(input_dim)
        super().__init__(input_dim, active_dim, name)

        self.link_parameters(self.kernel)

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X

        k_kernel = self.kernel.K(X, X2)

        k_cat = self.K_cat(X, X2)
        return k_kernel + k_cat

    def K_cat(self, X, X2):
        X_cat = X[:, self.cat_dims]
        X2_cat = X2[:, self.cat_dims]

        # Counting the number of categories that are the same using GPy's
        # broadcasting approach
        diff = X_cat[:, None] - X2_cat[None, :]
        # nonzero location = different cat
        diff[np.where(np.abs(diff))] = 1
        # invert, to now count same cats
        diff1 = np.logical_not(diff)
        # dividing by number of cat variables to keep this term in range [0,1]
        k_cat = self.kernel.variance * np.sum(diff1, -1) / len(self.cat_dims)
        return k_cat

    def update_gradients_full(self, dL_dK, X, X2=None):
        # Kernel gradients
        self.kernel.update_gradients_full(dL_dK, X, X2=X2)

        # Update the variance gradient using the contribution of the categories
        stat_grad = self.kernel.variance.gradient
        cat_kern_contrib = np.sum(self.K_cat(X, X2=X2)) / self.kernel.variance

        self.kernel.variance.gradient = stat_grad + cat_kern_contrib
        # print(self.gradient)