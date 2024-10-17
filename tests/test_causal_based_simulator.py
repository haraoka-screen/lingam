import json
import pytest

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.svm import SVR, SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from scipy.special import expit

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lingam.causal_based_simulator import CausalBasedSimulator
from lingam.causal_based_simulator import CBSILiNGAM
from lingam.causal_based_simulator import CBSIUnobsCommonCauseLiNGAM
from lingam.causal_based_simulator import CBSITimeSeriesLiNGAM


@pytest.fixture
def init():
    return lambda :np.random.seed(0)

@pytest.fixture
def test_data():
    np.random.seed(0)

    N = 1000
    causal_graph = np.array([[0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
                  [3.0, 0.0, 2.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 6.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [8.0, 0.0,-1.0, 0.0, 0.0, 0.0],
                  [4.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    e = np.random.uniform(-np.sqrt(3), np.sqrt(3), size=(len(causal_graph), N))
    X = np.linalg.pinv(np.eye(len(causal_graph)) - causal_graph) @ e
    X = X.T

    is_discrete = None

    return X, causal_graph, is_discrete

@pytest.fixture
def test_data_unobs():
    np.random.seed(0)

    N = 1000
    causal_graph = np.array([[0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
                  [3.0, 0.0, 2.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 6.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [8.0, 0.0,-1.0, 0.0, 0.0, 0.0],
                  [4.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    e = np.random.uniform(-np.sqrt(3), np.sqrt(3), size=(len(causal_graph), N))
    X = np.linalg.pinv(np.eye(len(causal_graph)) - causal_graph) @ e
    X = X.T

    delete_index = 3
    causal_graph = np.delete(causal_graph, delete_index, axis=0)
    causal_graph = np.delete(causal_graph, delete_index, axis=1)
    X = np.delete(X, delete_index, axis=1)

    is_discrete = None

    return X, causal_graph, is_discrete

@pytest.fixture
def test_data_ts():
    np.random.seed(0)

    N = 1000
    causal_graph = np.array([[
        [0,-0.12,0,0,0],
        [0,0,0,0,0],
        [-0.41,0.01,0,-0.02,0],
        [0.04,-0.22,0,0,0],
        [0.15,0,-0.03,0,0],
    ], [
        [-0.32,0,0.12,0.32,0],
        [0,-0.35,-0.1,-0.46,0.4],
        [0,0,0.37,0,0.46],
        [-0.38,-0.1,-0.24,0,-0.13],
        [0,0,0,0,0],
    ]])

    def _x_t(X, e, causal_graph):
        _, n_features, _ = causal_graph.shape
        term = np.linalg.pinv(np.eye(n_features) - causal_graph[0])
        term2 = np.hstack(causal_graph[1:]) @ np.hstack(X[:, ::-1][:, :len(causal_graph) - 1]).reshape(n_features, 1) + e
        return term @ term2

    n_lags = len(causal_graph) - 1
    size = N + n_lags
    
    errors = np.random.uniform(-np.sqrt(3), np.sqrt(3), size=(causal_graph.shape[1], size))

    X = errors[:, :n_lags]
    for t in range(n_lags, size):
        new_data = _x_t(X[:, :t], errors[:, [t]], causal_graph)
        X = np.append(X, new_data, axis=1)
    X = X.T

    is_discrete = None

    return X, causal_graph, is_discrete

@pytest.fixture
def test_data_discrete():
    np.random.seed(0)

    N = 1000
    causal_graph = np.array([[0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
                  [3.0, 0.0, 2.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 6.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [8.0, 0.0,-1.0, 0.0, 0.0, 0.0],
                  [4.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    e = np.random.uniform(-np.sqrt(3), np.sqrt(3), size=(len(causal_graph), N))
    X = np.linalg.pinv(np.eye(len(causal_graph)) - causal_graph) @ e
    X = X.T

    discrete_index = 5

    X[:, discrete_index] = (expit(X[:, discrete_index]) > np.random.uniform(size=N)).astype(int)

    is_discrete = [False for i in range(len(causal_graph))]
    is_discrete[discrete_index] = True

    return X, causal_graph, is_discrete

def test_cbs_success(test_data, test_data_unobs, test_data_ts, test_data_discrete):
    models = {"0": LinearRegression()}
    models_ts = {"0[t]": LinearRegression()}
    changing_models = {"0": {"parent_names": []}}
    changing_models_ts = {"0[t]": {"parent_names": []}}
    changing_exog = {"0": np.random.uniform(-10, 10, size=1000)}
    changing_exog_ts = {"0[t]": np.random.uniform(-10, 10, size=1000)}

    sim = CausalBasedSimulator()

    # normal data
    X, causal_graph, _ = test_data
    sim.train(X, causal_graph)
    sim.train(X, causal_graph, models=models)
    sim.run()
    sim.run(changing_models=changing_models, changing_exog=changing_exog)

    # unobserved
    X, causal_graph, _ = test_data_unobs
    sim.train(X, causal_graph, cd_algo_name="BottomUpParceLiNGAM")
    sim.train(X, causal_graph, cd_algo_name="BottomUpParceLiNGAM", models=models)
    sim.run()
    sim.run(changing_models=changing_models, changing_exog=changing_exog)
    
    # time series
    X, causal_graph, _ = test_data_ts
    sim.train(X, causal_graph, cd_algo_name="VARLiNGAM")
    sim.train(X, causal_graph, cd_algo_name="VARLiNGAM", models=models_ts)
    sim.run()
    sim.run(changing_models=changing_models_ts, changing_exog=changing_exog_ts)

    # discrete
    X, causal_graph, is_discrete = test_data_discrete
    sim.train(X, causal_graph, is_discrete=is_discrete)
    sim.train(X, causal_graph, is_discrete=is_discrete, models=models)
    sim.run()
    sim.run(changing_models=changing_models, changing_exog=changing_exog)

def test_cbs_exception(test_data):
    X, causal_graph, _ = test_data

    sim = CausalBasedSimulator()

    # cd_algo_name
    try:
        sim.train(X, causal_graph, cd_algo_name=1234)
    except:
        pass
    else:
        raise AssertionError

    try:
        sim.train(X, causal_graph, cd_algo_name="UnknownAlgoName")
    except:
        pass
    else:
        raise AssertionError

def test_cbsi_lingam_success(test_data):
    X, causal_graph, _ = test_data

    # X
    CBSILiNGAM(X, causal_graph)

    # X is pandas.DataFrame
    X_df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    CBSILiNGAM(X_df, causal_graph)

def test_cbsi_lingam_expcetion(test_data):
    X, causal_graph, _ = test_data

    # causal_graph
    try:
        CBSILiNGAM(X, np.eye(X.shape[1] + 1))
    except:
        pass
    else:
        raise AssertionError

def test_cbsi_unobs_success(test_data_unobs):
    return

def test_cbsi_unobs_expcetion(test_data_unobs):
    return

def test_cbsi_ts_success(test_data_ts):
    return

def test_cbsi_ts_expcetion(test_data_ts):
    return

def test_cbsi_discrete_success(test_data_discrete):
    return

def test_cbsi_discrete_expcetion(test_data_discrete):
    return
