"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/view/sshimizu06/lingam
"""

import numpy as np
from sklearn.preprocessing import scale
from sklearn.linear_model import LassoLarsCV, LinearRegression
from sklearn.preprocessing import StandardScaler
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

        cov_X_ = np.cov(X_.T)

        for _ in range(n_features):
            if self._measure == "kernel":
                m = self._search_causal_order_kernel(X_, U)
            elif self._measure == "pwling_fast":
                m = self._search_causal_order_gpu(X_.astype(np.float64), U.astype(np.int32))
            else:
                m = self._search_causal_order(X_, U)

            #if len(K) != 0:
            #    sub_cov = cov_X_[K][:, K]
            #    beta = np.linalg.pinv(sub_cov) @ cov_X_[K, m]
            #    X_[:, m] = X_[:, m] - X_[:, K] @ beta

            K.append(m)
            U = U[U != m]

            for i in U:
                sub_cov = cov_X_[K][:, K]
                beta = np.linalg.pinv(sub_cov) @ cov_X_[K, i]
                X_[:, i] = X_[:, i] - X_[:, K] @ beta

            # Update partial orders
            if (self._Aknw is not None) and (not self._apply_prior_knowledge_softly):
                self._partial_orders = self._partial_orders[
                    self._partial_orders[:, 0] != m
                ]

        self._causal_order = K
        #return self._estimate_adjacency_matrix(X, prior_knowledge=self._Aknw)
        self._adjacency_matrix = np.zeros((len(X.T), len(X.T)))
        return self

    #def _entropy(self, u):
    #    """Calculate entropy using the maximum entropy approximations."""
    #    k1 = 79.047
    #    k2 = 7.4129
    #    gamma = 0.37457
    #    return (1 + np.log(2 * np.pi)) / 2 - k1 * (np.mean(np.log(np.cosh(u)), axis=1) - gamma) ** 2 - k2 * (np.mean(u * np.exp((-(u ** 2)) / 2), axis=1)) ** 2

    #def _diff_mutual_info(self, xi_std, xj_std, ri_j, rj_i):
    #    """Calculate the difference of the mutual informations."""
    #    entropies = self._entropy(np.array([xj_std, ri_j / np.std(ri_j), xi_std, rj_i / np.std(rj_i)]))
    #    return (entropies[0] + entropies[1]) - (entropies[2] + entropies[3])

    #def _search_causal_order(self, X, U):
    #    """Search the causal ordering."""
    #    Uc, Vj = self._search_candidate(U)
    #    if len(Uc) == 1:
    #        return Uc[0]

    #    # standardize
    #    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    #    # residuals 残差の列
    #    res = X[:, :, None] - np.cov(X.T) * X[:, None, :]

    #    M_list = []
    #    for i in Uc:
    #        M = 0

    #        for j in U:
    #            if i == j:
    #                continue

    #            ri_j = X[:, i] if i in Vj and j in Uc else res[:, i, j]
    #            rj_i = X[:, j] if j in Vj and i in Uc else res[:, j, i]

    #            M += np.min([0, self._diff_mutual_info(X[:, i], X[:, j], ri_j, rj_i)]) ** 2
    #        M_list.append(-1.0 * M)
    #    return Uc[np.argmax(M_list)]

    #def _estimate_adjacency_matrix(self, X, prior_knowledge=None):
    #    """Estimate adjacency matrix by causal order.

    #    Parameters
    #    ----------
    #    X : array-like, shape (n_samples, n_features)
    #        Training data, where n_samples is the number of samples
    #        and n_features is the number of features.
    #    prior_knowledge : array-like, shape (n_variables, n_variables), optional (default=None)
    #        Prior knowledge matrix.

    #    Returns
    #    -------
    #    self : object
    #        Returns the instance itself.
    #    """
    #    if prior_knowledge is not None:
    #        pk = prior_knowledge.copy()
    #        np.fill_diagonal(pk, 0)

    #    B = np.zeros([X.shape[1], X.shape[1]], dtype="float64")
    #    for i in range(1, len(self._causal_order)):
    #        target = self._causal_order[i]
    #        predictors = self._causal_order[:i]

    #        # Exclude variables specified in no_path with prior knowledge
    #        if prior_knowledge is not None:
    #            predictors = [p for p in predictors if pk[target, p] != 0]

    #        # target is exogenous variables if predictors are empty
    #        if len(predictors) == 0:
    #            continue

    #        B[target, predictors] = _predict_adaptive_lasso(X, predictors, target)

    #    self._adjacency_matrix = B
    #    return self

def _predict_adaptive_lasso(X, predictors, target, gamma=1.0):
    """Predict with Adaptive Lasso.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training data, where n_samples is the number of samples
        and n_features is the number of features.
    predictors : array-like, shape (n_predictors)
        Indices of predictor variable.
    target : int
        Index of target variable.

    Returns
    -------
    coef : array-like, shape (n_features)
        Coefficients of predictor variable.
    """
    # Standardize X
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # Pruning with Adaptive Lasso
    lr = LinearRegression()
    lr.fit(X_std[:, predictors], X_std[:, target])
    weight = np.power(np.abs(lr.coef_), gamma)
    reg = LassoLarsCV()
    reg.fit(X_std[:, predictors] * weight, X_std[:, target])
    pruned_idx = np.abs(reg.coef_ * weight) > 0.0

    # Calculate coefficients of the original scale
    coef = np.zeros(reg.coef_.shape)
    if pruned_idx.sum() > 0:
        lr = LinearRegression()
        pred = np.array(predictors)
        lr.fit(X[:, pred[pruned_idx]], X[:, target])
        coef[pruned_idx] = lr.coef_

    return coef
