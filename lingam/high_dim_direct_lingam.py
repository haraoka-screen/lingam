"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/view/sshimizu06/lingam
"""

import numpy as np
from sklearn.utils import check_array

from .direct_lingam import DirectLiNGAM


class HighDimDirectLiNGAM(DirectLiNGAM):

    def fit(self, X):
        """Fit the model to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where ``n_samples`` is the number of samples
            and ``n_features`` is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Check parameters
        X = check_array(X)
        n_features = X.shape[1]

        # Check prior knowledge
        if self._Aknw is not None:
            if (n_features, n_features) != self._Aknw.shape:
                raise ValueError(
                    "The shape of prior knowledge must be (n_features, n_features)"
                )
            else:
                # Extract all partial orders in prior knowledge matrix
                if not self._apply_prior_knowledge_softly:
                    self._partial_orders = self._extract_partial_orders(self._Aknw)

        # Causal discovery
        U = np.arange(n_features)
        K = []
        X_ = np.copy(X)
        if self._measure == "kernel":
            X_ = scale(X_)

        cov_X = np.cov(X.T)

        for _ in range(n_features):
            if self._measure == "kernel":
                m = self._search_causal_order_kernel(X_, U)
            elif self._measure == "pwling_fast":
                m = self._search_causal_order_gpu(X_.astype(np.float64), U.astype(np.int32))
            else:
                m = self._search_causal_order(X_, U)

            if len(K) == 0:
                return X_

            sub_cov = cov_X[K][:, K]
            beta = np.linalg.pinv(sub_cov) @ cov_X[K, m]
            X_[:, m] = X_[:, m] - X_[:, K] @ beta

            K.append(m)
            U = U[U != m]
            # Update partial orders
            if (self._Aknw is not None) and (not self._apply_prior_knowledge_softly):
                self._partial_orders = self._partial_orders[
                    self._partial_orders[:, 0] != m
                ]

        self._causal_order = K
        return self._estimate_adjacency_matrix(X, prior_knowledge=self._Aknw)

