from pprint import pprint
import pylab as plt
import GPy
from typing import Optional, List, Union

# from scipy.spatial.distance import pdist
from paramz.transformations import Logexp

from ml_utils.models import GP
import numpy as np


class AdditiveGP(GP):
    """
    OBSOLETE

    Utility subclass with some useful shortcuts related to the
    cont-cat input GP
    """

    def __init__(self, *args, **kwargs):
        print("This class is obsolete!")
        super().__init__(*args, **kwargs)

    def predict_latent_continuous(self, x_star: np.ndarray,
                                  full_cov: bool = False):
        """
        Predict the latent space given the continuous kernel only
        """
        return super().predict_latent(x_star, full_cov, kern=self.kern.kernel)


# def create_additive_kernel(k1_class, k2_class,
#                            active_dims1, active_dims2,
#                            k1_args=None, k2_args=None):
#     """
#     Creates additive kernel of kern_type + delta
#     """
#     assert active_dims1 is not None
#     assert active_dims2 is not None
#
#     if k1_args is None:
#         k1_args = {}
#
#     if k2_args is None:
#         k2_args = {}
#
#     k1 = k1_class(len(active_dims1), active_dims=active_dims1, **k1_args)
#     k2 = k2_class(len(active_dims2), active_dims=active_dims2, **k2_args)
#
#     k_combined = k1 + k2
#
#     return k_combined


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


class MixtureViaSumAndProduct(GPy.kern.Kern):
    """
    Kernel of the form

    k = (1-mix)*(k1 + k2) + mix*k1*k2
    """

    def __init__(self, input_dim, k1, k2, active_dims=None, mix=0.5,
                 fix_variances=False):
        super().__init__(input_dim, active_dims, 'MixtureViaSumAndProduct')

        acceptable_kernels = (GPy.kern.RBF, GPy.kern.Matern52,
                              CategoryOverlapKernel)

        assert isinstance(k1, acceptable_kernels)
        assert isinstance(k2, acceptable_kernels)

        self.mix = mix
        self.k1 = k1
        self.k2 = k2

        self.fix_variances = fix_variances
        if self.fix_variances:
            self.k1.unlink_parameter(self.k1.variance)
            self.k2.unlink_parameter(self.k2.variance)

        self.link_parameters(self.k1, self.k2)

    def get_dk_dtheta(self, k, X, X2=None):
        if X2 is None:
            X2 = X
        X_sliced, X2_sliced = X[:, k.active_dims], X2[:, k.active_dims]

        if isinstance(k, (GPy.kern.RBF, GPy.kern.Matern52)):
            dk_dr = k.dK_dr_via_X(X_sliced, X2_sliced)

            # dr/dl
            if k.ARD:
                tmp = k._inv_dist(X_sliced, X2_sliced)
                dr_dl = -np.dstack([tmp * np.square(
                    X_sliced[:, q:q + 1] - X2_sliced[:, q:q + 1].T) /
                                    k.lengthscale[q] ** 3
                                    for q in range(k.input_dim)])
                dk_dl = dk_dr[..., None] * dr_dl
            else:
                r = k._scaled_dist(X_sliced, X2_sliced)
                dr_dl = - r / k.lengthscale
                dk_dl = dk_dr * dr_dl

            # # For testing the broadcast multiplication
            # dk_dl_slow = []
            # for ii in range(dr_dl.shape[-1]):
            #     dr_dlj = dr_dl[...,ii]
            #     dk_dlj = dk_dr * dr_dlj
            #     dk_dl_slow.append(dk_dlj)
            #
            # dk_dl_slow = np.dstack(dk_dl_slow)

        else:
            raise NotImplementedError

        # Return variance grad as well, if not fixed
        if not self.fix_variances:
            return k.K(X, X2) / k.variance, dk_dl
        else:
            return dk_dl

    def update_gradients_full(self, dL_dK, X, X2=None):

        # This gets the values of dk/dtheta as a NxN matrix (no summations)
        if X2 is None:
            X2 = X
        dk1_dtheta1 = self.get_dk_dtheta(self.k1, X, X2)  # N x N
        dk2_dtheta2 = self.get_dk_dtheta(self.k2, X, X2)  # N x N

        # Separate the variance and lengthscale grads (for ARD purposes)
        if self.fix_variances:
            dk1_dl1 = dk1_dtheta1
            dk2_dl2 = dk2_dtheta2
            dk1_dvar1 = []
            dk2_dvar2 = []
        else:
            dk1_dvar1, dk1_dl1 = dk1_dtheta1
            dk2_dvar2, dk2_dl2 = dk2_dtheta2

        # Evaluate each kernel over its own subspace
        k1_xx = self.k1.K(X, X2)  # N x N
        k2_xx = self.k2.K(X, X2)  # N x N

        # dk/dl for l1 and l2
        # ARD requires a summation along last axis for each lengthscale
        if self.k1.ARD:
            dk_dl1 = np.sum(dL_dK[..., None] * (dk1_dl1 * (1 - self.mix)
                                     + self.mix * dk1_dl1 * k2_xx[..., None]),
                            (0, 1))
        else:
            dk_dl1 = np.sum(dL_dK * (dk1_dl1 * (1 - self.mix)
                                     + self.mix * dk1_dl1 * k2_xx))

        if self.k2.ARD:
            dk_dl2 = np.sum(dL_dK[..., None] * (dk2_dl2 * (1 - self.mix)
                                     + self.mix * dk2_dl2 * k1_xx[..., None]),
                            (0, 1))
        else:
            dk_dl2 = np.sum(dL_dK * (dk2_dl2 * (1 - self.mix)
                                     + self.mix * dk2_dl2 * k1_xx))

        # dk/dvar for var1 and var 2
        if self.fix_variances:
            dk_dvar1 = []
            dk_dvar2 = []
        else:
            dk_dvar1 = np.sum(dL_dK * (dk1_dvar1 * (1 - self.mix)
                                       + self.mix * dk1_dvar1 * k2_xx))
            dk_dvar2 = np.sum(dL_dK * (dk2_dvar2 * (1 - self.mix)
                                       + self.mix * dk2_dvar2 * k1_xx))

        # Combining the gradients into one vector and updating
        dk_dtheta1 = np.hstack((dk_dvar1, dk_dl1))
        dk_dtheta2 = np.hstack((dk_dvar2, dk_dl2))
        self.k1.gradient = dk_dtheta1
        self.k2.gradient = dk_dtheta2

    def K(self, X, X2=None):
        k1_xx = self.k1.K(X, X2)
        k2_xx = self.k2.K(X, X2)
        return (1 - self.mix) * (k1_xx + k2_xx) + self.mix * k1_xx * k2_xx


class CategoryOverlapKernel(GPy.kern.Kern):
    """
    Kernel that counts the number of categories that are the same
    between inputs and returns the normalised similarity score:

    variance * 1/N_c * (degree of overlap)
    """

    def __init__(self, input_dim, variance=1.0, active_dims=None,
                 name='catoverlap'):
        super().__init__(input_dim, active_dims=active_dims, name=name)
        self.variance = GPy.core.parameterization.Param('variance',
                                                        variance, Logexp())
        self.link_parameter(self.variance)

    def K(self, X, X2):
        # Counting the number of categories that are the same using GPy's
        # broadcasting approach
        diff = X[:, None] - X2[None, :]
        # nonzero location = different cat
        diff[np.where(np.abs(diff))] = 1
        # invert, to now count same cats
        diff1 = np.logical_not(diff)
        # dividing by number of cat variables to keep this term in range [0,1]
        k_cat = self.variance * np.sum(diff1, -1) / self.input_dim
        return k_cat

    def update_gradients_full(self, dL_dK, X, X2=None):
        self.variance.gradient = np.sum(self.K(X, X2) * dL_dK) / self.variance


class StationaryUniformCat(GPy.kern.Kern):
    """
    OBSOLETE

    Kernel that is a combination of a stationary kernel and a
    categorical kernel. Each cat input has the same weight and is
    the cat variables' contribution to K() is normalised to [0, 1]
    """

    def __init__(self, kernel: GPy.kern.RBF, cat_dims):
        print("This class is obsolete!")
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

        k_t = k_kernel + k_cat
        # f, (ax1, ax2, ax3) = plt.subplots(3, 1)
        # plt.title(f"min_kernel = {np.min(k_kernel)}, max_kernel = {np.max(k_kernel)}, \n"
        #             f"min_cat = {np.min(k_cat)}, max_cat = {np.max(k_cat)}")
        # ax1.imshow(k_kernel)
        # ax2.imshow(k_cat)
        # ax3.imshow(k_kernel + k_cat)
        # # f.colorbar()
        # plt.show()
        return k_t

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
        cat_kern_contrib = np.sum(
            self.K_cat(X, X2=X2) * dL_dK) / self.kernel.variance

        self.kernel.variance.gradient = stat_grad + cat_kern_contrib
        # print(f"Updating gradient: "
        #       f"{np.hstack((self.gradient, cat_kern_contrib))}")
