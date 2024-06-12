"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/view/sshimizu06/lingam
"""

import numpy as np
from sklearn.utils import check_array, check_scalar

from .direct_lingam import DirectLiNGAM


class HighDimDirectLiNGAM(DirectLiNGAM):


    def __init__(self, disable_estimate_adj_mat=False, **kwargs):
        """Construct a DirectLiNGAM model.

        Parameters
        ----------
        random_state : int, optional (default=None)
            ``random_state`` is the seed used by the random number generator.
        prior_knowledge : array-like, shape (n_features, n_features), optional (default=None)
            Prior knowledge used for causal discovery, where ``n_features`` is the number of features.

            The elements of prior knowledge matrix are defined as follows [1]_:

            * ``0`` : :math:`x_i` does not have a directed path to :math:`x_j`
            * ``1`` : :math:`x_i` has a directed path to :math:`x_j`
            * ``-1`` : No prior knowledge is available to know if either of the two cases above (0 or 1) is true.
        apply_prior_knowledge_softly : boolean, optional (default=False)
            If True, apply prior knowledge softly.
        measure : {'pwling', 'kernel', 'pwling_fast'}, optional (default='pwling')
            Measure to evaluate independence: 'pwling' [2]_ or 'kernel' [1]_.
            For fast execution with GPU, 'pwling_fast' can be used (culingam is required).
        disable_estimate_adj_mat: bool optional (default=False)
            An adjacency matrix estimation is skipped if it is True; otherwise, it is not.
        """
        disable_estimate_adj_mat = check_scalar(disable_estimate_adj_mat, "disable_estimate_adj_mat", bool)

        super().__init__(kwargs)

        self._disable_estimate_adj_mat = disable_estimate_adj_mat

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

        cov_X_ = np.cov(X_.T)

        for _ in range(n_features):
            if self._measure == "kernel":
                m = self._search_causal_order_kernel(X_, U)
            elif self._measure == "pwling_fast":
                m = self._search_causal_order_gpu(X_.astype(np.float64), U.astype(np.int32))
            else:
                m = self._search_causal_order(X_, U)

            if len(K) != 0:
                sub_cov = cov_X_[K][:, K]
                beta = np.linalg.pinv(sub_cov) @ cov_X_[K, m]
                X_[:, m] = X_[:, m] - X_[:, K] @ beta

            K.append(m)
            U = U[U != m]
            # Update partial orders
            if (self._Aknw is not None) and (not self._apply_prior_knowledge_softly):
                self._partial_orders = self._partial_orders[
                    self._partial_orders[:, 0] != m
                ]

        self._causal_order = K

        if self._disable_estimate_adj_mat:
            self._adjacency_matrix = np.zero((X.shape[1], X.shape[1]))
            return self
        else:
            return self._estimate_adj_mat(X, prior_knowledge=self._Aknw)

