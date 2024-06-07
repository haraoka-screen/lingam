"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/view/sshimizu06/lingam
"""

import numpy as np

from .direct_lingam import DirectLiNGAM


class HighDimDirectLiNGAM(DirectLiNGAM):

    def _prepare_to_causal_discovery(self, X):
        self._cov_X = np.cov(X.T)

    def _update_residual(self, X, U, K, m):
        if len(K) == 0:
            return X

        sub_cov = self._cov_X[K][:, K]
        beta = np.linalg.pinv(sub_cov) @ self._cov_X[K, m]
        X[:, m] = X[:, m] - X[:, K] @ beta

        return X
