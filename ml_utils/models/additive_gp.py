from pprint import pprint

import GPy
from typing import Optional, List, Union

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


class DeltaKernel(GPy.kern.Kern):
    # TODO
    pass

