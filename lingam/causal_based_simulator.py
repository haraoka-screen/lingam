from .direct_lingam import DirectLiNGAM
from .bottom_up_parce_lingam import BottomUpParceLiNGAM
#from lingam import DirectLiNGAM, BottomUpParceLiNGAM

from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd

from sklearn.utils import check_array, check_random_state
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.base import RegressorMixin, ClassifierMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection._search import BaseSearchCV


class CBSImpl(metaclass=ABCMeta):

    def __init__(self, X, causal_graph):
        raise NotImplementedError

    @property
    def endog_names_(self):
        raise NotImplementedError

    @property
    def discrete_endog_names_(self):
        raise NotImplementedError

    @property
    def causal_order_(self):
        raise NotImplementedError

    @property
    def exog_length_(self):
        raise NotImplementedError

    #@abstractmethod
    # 誤差項の扱いが因果探索モデルごとに異なる場合は増えます。
    #def set_exog_data(self, var_name, e):
    #    raise NotImplementedError

    @abstractmethod
    def get_parent_names(self, var_name):
        raise NotImplementedError

    @abstractmethod
    def get_data(self, var_names):
        raise NotImplementedError

    @abstractmethod
    def get_causal_order(self, changing_edges=None):
        raise NotImplementedError


class CBSIDirectLiNGAM(CBSImpl):

    def __init__(self, X, causal_graph, is_discrete=None):
        X_ = check_array(X)

        n_samples, n_features = X_.shape

        causal_graph = check_array(causal_graph)
        if causal_graph.shape != (n_features, n_features):
            raise RuntimeError("The shape of causal_graph must be (n_features, n_features)")

        if isinstance(X, pd.DataFrame):
            endog_names = X.columns.tolist()
        else:
            endog_names = [f"{i:d}" for i in range(n_features)]

        if is_discrete is None:
            is_discrete = [False for _ in range(n_features)]

        discrete_endog_names = np.array(endog_names)[is_discrete].tolist()

        causal_order = self._calc_causal_order(causal_graph)
        causal_order = [endog_names[n] for n in causal_order]

        self._X = X_
        self._exog_length = n_samples
        self._is_discrete = is_discrete
        self._endog_names = endog_names
        self._discrete_endog_names = discrete_endog_names
        self._causal_graph = causal_graph
        self._causal_order = causal_order

    @property
    def causal_order_(self):
        return self._causal_order

    @property
    def endog_names_(self):
        return self._endog_names

    @property
    def discrete_endog_names_(self):
        return self._discrete_endog_names

    @property
    def exog_length_(self):
        return self._exog_length

    def get_parent_names(self, var_name):
        if var_name not in self._endog_names:
            raise ValueError("not exist")

        causal_graph = ~np.isclose(self._causal_graph, 0)

        index = self._endog_names.index(var_name)
        parent_indices = np.argwhere(causal_graph[index, :]).ravel()
        parent_names = np.array(self._endog_names)[parent_indices].tolist()

        return parent_names

    def get_data(self, var_names):
        var_indices = []
        for var_name in var_names:
            index = self._endog_names.index(var_name)
            var_indices.append(index)
        data = self._X[:, var_indices]
        
        return data

    def get_causal_order(self, changing_edges=None):
        if changing_edges is None:
            changing_edges = {}

        causal_graph = self._causal_graph.copy()

        for y_name, X_names in changing_edges.items():
            row = self._endog_names.index(y_name)
            cols = [self._endog_names.index(X_name) for X_name in X_names]

            causal_graph[row, :] = 0
            causal_graph[row, cols] = 1

        causal_order = self._calc_causal_order(causal_graph)
        causal_order = [self._endog_names[n] for n in causal_order]

        return causal_order

    def _calc_causal_order(self, causal_graph):
        """Obtain a causal order from the given causal_graph strictly.

        Parameters
        ----------
        causal_graph : array-like, shape (n_features, n_samples)
            Target causal_graph.

        Return
        ------
        causal_order : array, shape [n_features, ]
            A causal order of the given causal_graph on success, None otherwise.
        """
        causal_order = []

        row_num = causal_graph.shape[0]
        original_index = np.arange(row_num)

        while 0 < len(causal_graph):
            # find a row all of which elements are zero
            row_index_list = np.where(np.sum(np.abs(causal_graph), axis=1) == 0)[0]
            if len(row_index_list) == 0:
                break

            target_index = row_index_list[0]

            # append i to the end of the list
            causal_order.append(original_index[target_index])
            original_index = np.delete(original_index, target_index, axis=0)

            # remove the i-th row and the i-th column from causal_graph
            mask = np.delete(np.arange(len(causal_graph)), target_index, axis=0)
            causal_graph = causal_graph[mask][:, mask]

        if len(causal_order) != row_num:
            causal_order = None

        return causal_order


class CausalBasedSimulator:
    """
    Causal based simulator.

    Attributes
    ----------
    train_result_ : dict of string -> list of namedtuple
        information about trained models.

    simulated_data_ : pandas.DataFrame
        result of simulation.
    """

    def train(self, X, causal_graph, causal_graph_type="DirectLiNGAM", models=None):
        """
        Estimate functional relations between variables and variable
        distributions based on the training data ``X`` and the causal graph
        ``G``. The functional relations represents by
        sklearn.linear_model.LinearRegression if the object variable is
        numeric, and represents by sklearn.linear_model.LogisticRegression
        if the object variable is categorical by default. ``train_result_``
        and ``residual_`` will be exposed after executing train().

        Parameters
        ----------
        X : array-like
            Training data.

        causal_graph : array-like of shape (n_features, _features)
            Causal graph.

        models : dict of string -> object, default=None
            Dictionary about models of variables. Models are cloned internaly
            and are trained to infer functioal relations. Given instances of
            the model are cloned to estimate the functional relation between
            variables.

        Returns
        -------
        self : Object
        """

        if not isinstance(causal_graph_type, str):
            raise TypeError("causal_graph_type must be str.")

        impl_constructor = self._dispatch_impl(causal_graph_type)
        impl = impl_constructor(X, causal_graph)

        train_models = self._check_models(models, impl.endog_names_, impl.discrete_endog_names_)

        train_result = self._train(train_models, causal_graph, impl)

        self._impl = impl
        self._train_result = train_result
        return self

    def run(
        self,
        changing_exog=None,
        changing_models=None,
        shuffle_residual=False,
        random_state=None,
    ):
        """
        Generate simulated data using trained models and the given
        causal graph with given exogenous data and models.
        Specifying environmental changes to ``changing_exog`` or
        specifiyig changes in fucitonal relation to ``change_models``
        effects simulated data.
        Residuals to simulate variables are shuffled using radom_state
        if ``shuffle_residual`` is True. ``simulated_data_`` will be
        expose after excuting train().

        Parameters
        ----------
        changing_exog : dict of string -> array-like, default=None
            Dictioary about exogeous variables which keys are variable
            names and values are data of variables. That variable name
            should be a one of column names of X and the length should
            be same as X.

        changing_model : list of dict, default=None
            List of the changing models which elements are dictionary. that
            keys should be name, condition, and model. For name and
            condition, refer to ``train_result_`` and set the values
            corresponding to the conditions you wish to change. For model,
            you must provide a trained machine learning instance.

        shuffle_residual : bool, default=True
            If True, residuals are shuffled.

        random_state : int, default=None
            If shuffle_residual is True, random_state is used as seed.

        Returns
        -------
        simulated_data : pandas.DataFrame
            simulated data.
        """

        if self._train_result is None:
            raise RuntimeError("run() must be executed after train() is executed")

        random_state = check_random_state(random_state)

        # check inputs
        changing_exog = self._check_changing_exog(
            changing_exog,
            self._impl.exog_length_,
            self._impl.endog_names_,
            self._impl.discrete_endog_names_
        )

        changing_models = self._check_changing_models(
            changing_models,
            self._impl.endog_names_,
            self._impl.discrete_endog_names_
        )

        simulated_data = self._simulate(
            changing_exog,
            changing_models,
            self._impl,
            self._train_result,
            shuffle_residual,
            random_state,
        )

        return simulated_data

    @property
    def train_result_(self):
        return self._train_result

    def _check_model_instance(self, model, var_name, discrete_endog_names):
        if var_name not in discrete_endog_names:
            model_type = RegressorMixin
        else:
            model_type = ClassifierMixin

        if isinstance(model, Pipeline):
            if not isinstance(model.steps[-1][-1], model_type):
                raise RuntimeError(
                    "The last step in Pipeline should be an "
                    + "instance of a regression/classification model."
                )
        elif isinstance(model, BaseSearchCV):
            if not isinstance(model.get_params()["estimator"], model_type):
                raise RuntimeError(
                    "The type of the estimator shall be an "
                    + "instance of a regression/classification model."
                )
        else:
            if not isinstance(model, model_type):
                raise RuntimeError(
                    "The type of the estimator shall be an "
                    + "instance of a regression/classification model."
                )

        if model_type == ClassifierMixin:
            try:
                func = getattr(model, "predict_proba")
                if not callable(func):
                    raise Exception
            except Exception:
                raise RuntimeError(
                    "Classification models shall have " + "predict_proba()."
                )

    def _check_models(self, models, endog_names, discrete_endog_names):
        if models is None:
            return {}

        if not isinstance(models, dict):
            raise RuntimeError("models must be a dictionary.")

        for var_name, model in models.items():
            if var_name not in endog_names:
                raise RuntimeError(f"Unknown variable name ({var_name})")

            self._check_model_instance(model, endog_name, discrete_endog_names)

        return models

    def _check_changing_exog(self, changing_exog, n_samples, endog_names, discrete_endog_names):
        if changing_exog is None:
            return {}

        if not isinstance(changing_exog, dict):
            raise RuntimeError("changing_exog must be a dictionary.")

        changing_exog_ = {}
        for var_name, values in changing_exog.items():
            if var_name not in exog_names:
                raise RuntimeError(f"Unknown key in changing_exog. ({col_name})")

            if var_name in discrete_endog_names:
                raise RuntimeError(
                    f"Category variables shall not be specified. ({col_name})"
                )

            s = check_array(values, ensure_2d=False, dtype=None).ravel()
            if s.shape[0] != len(index):
                raise RuntimeError(f"Wrong length. ({s.shape[0]} != {len(index)})")

            changing_exog_[var_name] = values

        return changing_exog_

    def _check_changing_models(self, changing_models, endog_names, discrete_endog_names):
        if changing_models is None:
            return {}

        if not isinstance(changing_models, dict):
            raise RuntimeError("changing_models shall be list.")

        changing_models_ = {}
        for y_name, model_info in changing_models.items():
            if not isinstance(model_info, dict):
                raise RuntimeError("changing_models shall be list of dictionaries.")

            missing_keys = {"model", "X_names"} - set(model_info.keys())
            if len(missing_keys) > 0:
                raise RuntimeError("Missing key on model_info. " + str(missing_keys))

            if not isinstance(y_name, str):
                raise TypeError("Key of changing_models must be str.")
            if y_name not in endog_names:
                raise RuntimeError(f"Unknown name. ({name})")

            X_names = model_info["X_names"]
            if X_names is None:
                X_names = []
            else:
                if not isinstance(X_names, list):
                    raise TypeError("")
                for X_name in model_info["X_names"]:
                    if X_name not in endog_names:
                        raise RuntimeError(f"Unknown name. ({name})")

            model = model_info["model"]
            if model is not None:
                self._check_model_instance(model, y_name, discrete_endog_names)
            else:
                if len(X_names) > 0:
                    raise ValueError("model is None but X_names is not empty")

            changing_models_[y_name] = {
                "model": model,
                "X_names": X_names,
            }
        return changing_models_

    def _dispatch_impl(self, causal_graph_type):
        if causal_graph_type == DirectLiNGAM.__name__:
            return CBSIDirectLiNGAM
        elif causal_graph_type == BottomUpParseLiNGAM.__name__:
            return CBSIDirectLiNGAM
        else:
            raise ValueError("Unknown")

    def _train(self, models, causal_graph, impl):
        train_result = {}

        for y_name in impl.endog_names_:
            y = impl.get_data(y_name)

            X_names = impl.get_parent_names(y_name)
            if len(X_names) == 0:
                train_result[y_name] = {
                    "model": None,
                    "X_names": [],
                    "y_pred": None,
                    "residual": y.ravel(),
                }
                continue
            X = impl.get_data(X_names)

            is_classifier = y_name in impl.discrete_endog_names_

            # select a model to train
            if y_name in models.keys():
                model = clone(models[y_name])
            else:
                if is_classifier:
                    model = LogisticRegression()
                else:
                    model = LinearRegression()

            model.fit(X, y)
            y_pred = model.predict(X)

            # compute residuals
            if not is_classifier:
                residual = y - y_pred
            else:
                residual = None

            train_result[y_name] = {
                "model": model,
                "X_names": X_names,
                "y_pred": y_pred.ravel(),
                "residual": residual.ravel(),
            }

        return train_result

    def _simulate(
        self,
        changing_exog,
        changing_models,
        impl,
        train_result,
        shuffle_residual,
        random_state,
    ):
        simulated = pd.DataFrame(columns=impl.endog_names_)

        # modify causal order
        changing_edges = {}
        for y_name, info in changing_models.items():
            changing_edges[y_name] = info["X_names"]
        causal_order = self._impl.get_causal_order(changing_edges)

        # predict from upstream to downstream
        for y_name in causal_order:
            # error
            if y_name not in changing_exog.keys():
                error = train_result[y_name]["residual"]
            else:
                error = changing_exog[y_name].ravel()

            # data
            X_names = impl.get_parent_names(y_name)
            if y_name in changing_models.keys():
                X_names_ = changing_models[y_name]["X_names"]
                if X_names_ is not None:
                    X_names = X_names_
 
            if len(X_names) == 0:
                simulated[y_name] = error
                continue

            if shuffle_residual:
                error = random_state.choice(error, size=len(error), replace=False)

            X = simulated[X_names]

            # model
            if y_name not in changing_models.keys():
                model = train_result[y_name]["model"]
            else:
                model = changing_models[y_name]["model"]

            # predict
            y_pred = model.predict(X.values)
            y_pred = y_pred.ravel()
            if y_name not in impl.discrete_endog_names_:
                y_pred = y_pred + error

            simulated[y_name] = y_pred

        return simulated

