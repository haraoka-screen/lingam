""" ノートブック間で共有している関数
"""

import numpy as np
import pandas as pd
from scipy.special import expit

import matplotlib.pyplot as plt

from sklearn.utils import check_array
from lingam import DirectLiNGAM, BottomUpParceLiNGAM, VARLiNGAM, CausalBasedSimulator


class CausalDigitalTwin:

    def __init__(self, causal_graph, error, data_gen_func, sink_index, is_discrete=None, cd_algo_name="DirectLiNGAM"):
        """
        Arguments
        ---------
        causal_graph : numpy.ndarray, shape=(n_features, n_features)
            因果グラフ。
        error : numpy.ndarray
            誤差項データ。
        data_gen_func : callable
            data_gen_func(causal_graph, e) でデータを生成する。
        sink_index : int
            シンク変数のインデックス。
        is_discrete : list
            各変数が離散であるかどうか。Noneの場合はすべて連続変数として扱う。
        cd_algo_name : str
            データ生成や管理の方法を指定する。CausalBasedSimulator内部で、与えられたデータの扱い方を判断するために使用する。
            DirectLiNGAM、BottomUpParceLinNGAM、VARLiNGAMのいずれかを指定する。
        """
        # 変更前の真のデータ
        X, error = data_gen_func(causal_graph, error)

        # 真のデータを元に因果探索を実行して因果グラフを推定する。
        if cd_algo_name == "DirectLiNGAM":
            cd_model = DirectLiNGAM()
        elif cd_algo_name == "BottomUpParceLiNGAM":
            cd_model = BottomUpParceLiNGAM()
        elif cd_algo_name == "VARLiNGAM":
            cd_model = VARLiNGAM()
        else:
            raise ValueError("Unknown cd_algo_name")

        cd_model.fit(X)

        if cd_algo_name == "DirectLiNGAM" or cd_algo_name == "BottomUpParceLiNGAM":
            est_adj = cd_model.adjacency_matrix_
        elif cd_algo_name == "VARLiNGAM":
            est_adj = cd_model.adjacency_matrices_

        sim = CausalBasedSimulator()
        sim.train(X, est_adj, cd_algo_name=cd_algo_name, is_discrete=is_discrete)

        self._X = X
        self._error = error
        self._sim = sim
        self._causal_graph = causal_graph
        self._error = error
        self._data_gen_func = data_gen_func
        self._sink_index = sink_index
        self._is_discrete = is_discrete
        self._est_adj = est_adj
    
    def run(self, ml_models, eval_funcs, causal_graph=None, error=None, shuffle_residual=False):
        """
        Arguments
        ---------
        ml_models : dict
            機械学習モデルの辞書。キーは変数名、値は機械学習モデルのインスタンス。
        eval_funcs : dict
            評価関数の辞書。
        causal_graph : numpy.ndarray, shape=(n_features, n_features)
            変更後の因果グラフ。
        error : numpy.ndarray, shape=(n_features, n_samples)
            変更後の誤差項データ。
        shuffle_residual : boolean
            誤差項のシャッフルを行うかどうか
        
        Return
        ------
        evaluated
        """
        
        if causal_graph is None:
            changing_models = None
        else:
            # 真のDAGとユーザが真のDAGを見ながら更新したDAGの差分からchanging_modelsを作成する。
            changing_models = self._make_changing_models(self._causal_graph, causal_graph)
        
        if error is None:
            changing_exog = None
        else:
            changing_exog = {i: error[:, i].ravel() for i in range(error.shape[1])}
        
        simulated = self._sim.run(changing_models=changing_models, changing_exog=changing_exog, shuffle_residual=shuffle_residual)
        
        if causal_graph is None:
            causal_graph = self._causal_graph
        if error is None:
            error = self._error

        # 変更後の真のデータ
        X2, _ = self._data_gen_func(causal_graph, error)

        # NaNはBottomUpParceLiNGAMのときのみ
        causal_graph = check_array(self._causal_graph, ensure_2d=False, allow_nd=True)
        causal_graph2 = check_array(causal_graph, ensure_2d=False, allow_nd=True)

        # 因果グラフの形状考慮
        if len(causal_graph.shape) == 3:
            index = (0, self._sink_index)
        else:
            index = (self._sink_index,)
        # 変化前のシンク変数の親
        parent_indices = np.argwhere(~np.isclose(causal_graph[index], 0)).ravel()
        # 変化後のシンク変数の親
        parent_indices2 = np.argwhere(~np.isclose(causal_graph2[index], 0)).ravel()

        evaluates, predicted, configs = self._evaluate(self._X, X2, simulated, self._sink_index, parent_indices, parent_indices2, ml_models, eval_funcs)
        
        ret = {
            "X": self._X,
            "X2": X2,
            "simulated": simulated,
            "evaluates": evaluates,
            "predicted": predicted,
            "configs": configs,
        }
        
        return ret
    
    def _make_changing_models(self, causal_graph, causal_graph2):
        """ グラフ間の差分からchanging_modelsを作成する。"""

        # 未観測共通原因は0にしておく
        causal_graph[np.isnan(causal_graph)] = 0
        causal_graph2[np.isnan(causal_graph2)] = 0
        
        # 変数名のリスト
        if len(causal_graph.shape) == 2:
            var_names = [f"{i}" for i in range(len(causal_graph))]
        else:
            n_dags, n_features, _ = causal_graph.shape
            var_names = []
            for i in range(n_dags):
                if i == 0:
                    var_names += [f"{j}[t]" for j in range(n_features)]
                else:
                    var_names += [f"{j}[t-{i}]" for j in range(n_features)]
        
        # VARの形式の時は隣接行列を変更
        if len(causal_graph.shape) > 2:
            causal_graph = np.concatenate(causal_graph, axis=1)
            causal_graph2 = np.concatenate(causal_graph2, axis=1)
        
        # グラフの構造の変化からchanging_modelsを作成する。
        changed = ~np.isclose(causal_graph - causal_graph2, 0)

        # 説明変数の係数に変化を含む行を対象として処理
        changing_models = {}
        for i, row in enumerate(changed):
            # 変化なしはパス
            if np.isclose(np.sum(row), 0):
                continue

            # 変更後のDAGに従って親変数とその係数値を改めて設定する。
            parent_indices = np.argwhere(~np.isclose(causal_graph2[i], 0)).ravel()
            parent_names = [var_names[j] for j in parent_indices]
            coefs = causal_graph2[i, parent_indices] if len(parent_indices) != 0 else []
            changing_models[var_names[i]] = {"parent_names": parent_names, "coef": coefs}
        if len(changing_models) == 0:
            changing_models = None
        
        return changing_models
    
    def _evaluate(self, X, X2, simulated, sink_index, parent_indices, parent_indices2, ml_models, eval_funcs):    
        """
        X
            変化前の真のデータ
        X2
            変化後の真のデータ(変化前後で誤差項は同じものを使用)
        simulated
            変化後をCausalBasedSimulatorでシミュレーションしたデータ
        sink_index
            シンク変数。機械学習の目的変数になる。
        ml_models
            機械学習モデルの辞書。回帰モデルのみか分類モデルのみにする必要がある。
            機械学習モデルはscikit-learnのインスタンスを使用すること。
        eval_funcs
            評価関数の辞書。ml_modelsが回帰モデルのみであれば、
            評価関数も回帰モデル用のもののみとする必要がある。
            分類モデルのみであれば分類モデル用の評価関数のみとする必要がある。
            評価関数にはscikit-learnの評価関数を使用すること。
        parent_indices
            変化前シンク変数の親変数のインデックス
        parent_indices2
            変化後シンク変数の親変数のインデックス
        """
        # シンク変数を予測する機械学習の訓練の設定
        configs = {
            # 変化前の真のデータで訓練、変化前真データで予測
            "before": {
                # 訓練
                "X_train": X[:, parent_indices],
                "y_train": X[:, sink_index],
                # 予測
                "X_test": X[:, parent_indices],
                # 評価
                "y_true": X[:, sink_index],
            },
            # 変化後のシミュレーションデータで訓練、変化後のシミュレーションデータで予測
            "simulation": {
                "X_train": simulated.iloc[:, parent_indices2],
                "y_train": simulated.iloc[:, sink_index],
                "X_test": simulated.iloc[:, parent_indices2],
                "y_true": X2[:, sink_index],
            },
            # 変化前の真のデータで訓練、変化後の真のデータで予測。親変数は訓練時のものを使用する。
            "before_after": {
                "X_train": X[:, parent_indices],
                "y_train": X[:, sink_index],
                "X_test": X2[:, parent_indices],
                "y_true": X2[:, sink_index],
            }
        }
        
        evaluated = {}
        predicted = {}
        for config_name, config in configs.items():
            for ml_model_name, ml_model in ml_models.items():
                ml_model.fit(config["X_train"], config["y_train"])

                y_pred = ml_model.predict(config["X_test"])
                y_true = config["y_true"]

                predicted[(config_name, ml_model_name)] = y_true, y_pred
                
                for eval_name, eval_func in eval_funcs.items():
                    # XXX: 時系列のシミュレートを行うとラグの分だけ縮んでしまう。
                    value = eval_func(y_true[:len(y_pred)], y_pred)
                    evaluated[(config_name, ml_model_name, eval_name)] = value
                    
        return evaluated, predicted, configs

def generate_test_data(causal_graph, causal_order, ratio_list, size=1000):
    """ 誤差項の分散が指定の割合になるように係数を調整しながらデータを生成する。
    
    Arguments
    ---------
    causal_graph
        因果グラフ
    raito_list
        各変数における誤差項の分散の割合
    causal_order
        因果順序
    e
        誤差項
    Return
    ------
    X
        データ
    causal_graph
        係数を調整した因果グラフ
    """
    n_features = len(causal_graph)
    
    # generate errors
    e = np.empty((n_features, size))
    for i, ratio in enumerate(ratio_list):
        a = np.sqrt(3 * ratio)
        e[i] = np.random.uniform(-a, a, size=size)

    # generate data
    X = np.zeros((n_features, size))
    for no in causal_order:
        if np.all(np.isclose(causal_graph[no], 0)):
            X[no] = e[no]
            continue
            
        # adjust coefs
        var = np.var(causal_graph[no] @ X)
        causal_graph[no] = causal_graph[no] / np.sqrt(var) * np.sqrt(1 - ratio_list[no])
    
        # generate
        X[no] = causal_graph[no] @ X + e[no]
    
    return X.T, causal_graph, e.T

def discretize(X, sink_index):
    """ X[:, sink_index] を離散化する。 対象は常にシンクなので最後に適用すればよい。"""
    prob = expit(X[:, sink_index])
    mask = np.random.uniform(size=len(X))
    X[:, sink_index] = prob > mask
    return X

def draw_hist(n_features, n_patterns, results):
    # シミュレーションの様子
    fig, axes = plt.subplots(n_patterns, n_features, figsize=(n_features*2, n_patterns*1.5))
    count = 0

    for name, result in results.items():
        for name2, result2 in result.items():
            X_true_before = result2["X"]
            X_true_after = result2["X2"]
            X_sim = result2["simulated"]
            evaluates = result2["evaluates"]

            ax = axes[count]
            for i, (true, sim) in enumerate(zip(X_true_before.T, X_sim.values.T)):
                range_ = min(*true, *sim), max(*true, *sim)
                ax[i].hist(true, range=range_, label="true", alpha=0.5)
                ax[i].hist(sim, range=range_, label="sim", alpha=0.5)
                ax[i].set_title(f"x{i}")

                if count == 0:
                    ax[i].set_title(f"x{i}")
                if i == 0:
                    min_, max_ = ax[i].get_xlim()
                    text_pos = min_ - abs(max_ - min_) * 0.4

                    s = "operation: " + name + "\n"
                    s += "sink type: " + name2.split("_")[0] + "\n"
                    s += "shuffle residual: " + ("True" if len(name2.split("_")) == 2 else "False") + "\n"
                    ax[i].text(text_pos, 0, s, ha="right", va="bottom")
                if count == n_patterns - 1 and i == n_features - 1:
                    ax[i].legend(bbox_to_anchor=(1, -0.7), loc="center right")
            count += 1

    plt.tight_layout()
    plt.show()
    
def make_tables(results):
    table_c = []
    columns_c = ["operation", "model", "shuffle error", "MSE(true)", "MSE(sim)", "MSE(est)"]

    table_d = []
    columns_d = ["operation", "model", "shuffle error", "Precision(true)", "Precision(sim)", "Precision(est)", "Recall(true)", "Recall(sim)", "Recall(est)"]

    for name, result in results.items():
        for name2, result2 in result.items():
            is_continuous = name2.split("_")[0] == "continuous"
            is_shuffle = len(name2.split("_")) == 2

            X_true_before = result2["X"]
            X_true_after = result2["X2"]
            X_sim = result2["simulated"]
            evaluates = result2["evaluates"]
            is_shuffle = len(name2.split("_")) > 1

            if is_continuous:
                # lr
                mse_true = evaluates[("before", "lr", "mse")]
                mse_sim = evaluates[("simulation", "lr", "mse")]
                mse_est = evaluates[("before_after", "lr", "mse")]
                table_c.append((name, "LinearRegression", str(is_shuffle), f"{mse_true:.3f}", f"{mse_sim:.3f}", f"{mse_est:.3f}"))

                # rf
                mse_true = evaluates[("before", "rf", "mse")]
                mse_sim = evaluates[("simulation", "rf", "mse")]
                mse_est = evaluates[("before_after", "rf", "mse")]
                table_c.append((name, "RandomForestRegressor", str(is_shuffle), f"{mse_true:.3f}", f"{mse_sim:.3f}", f"{mse_est:.3f}"))
            else:
                # lr
                precision_true = evaluates[("before", "lr", "precision")]
                precision_sim = evaluates[("simulation", 'lr', 'precision')]
                precision_est = evaluates[("before_after", 'lr', 'precision')]
                recall_true = evaluates[("before", 'lr', 'recall')]
                recall_sim = evaluates[("simulation", 'lr', 'recall')]
                recall_est = evaluates[("before_after", 'lr', 'recall')]
                table_d.append(
                    (name, "LinearRegression", str(is_shuffle),
                     f"{precision_true:.3f}", f"{precision_sim:.3f}", f"{precision_est:.3f}", f"{recall_true:.3f}", f"{recall_sim:.3f}", f"{recall_est:.3f}")
                )

                # rf
                precision_true = evaluates[("before", 'rf', 'precision')]
                precision_sim = evaluates[("simulation", 'rf', 'precision')]
                precision_est = evaluates[("before_after", 'rf', 'precision')]
                recall_true = evaluates[("before", 'rf', 'recall')]
                recall_sim = evaluates[("simulation", 'rf', 'recall')]
                recall_est = evaluates[("before_after", 'rf', 'recall')]
                table_d.append(
                    (name, "RandomForestRegressor", str(is_shuffle),
                     f"{precision_true:.3f}", f"{precision_sim:.3f}", f"{precision_est:.3f}", f"{recall_true:.3f}", f"{recall_sim:.3f}", f"{recall_est:.3f}")
                )

    table_c = pd.DataFrame(table_c, columns=columns_c)
    table_d = pd.DataFrame(table_d, columns=columns_d)
    
    return table_c, table_d

def _draw_pred_hist(n_patterns, results, sink_index, discrete_sink=False):
    n_cols = 3
    
    fig, axes = plt.subplots(n_patterns, n_cols, figsize=(n_cols*3, n_patterns*1.5))
    count = 0
    
    for name, result in results.items():
        for name2, result2 in result.items():
            X_true_before = result2["X"]
            X_true_after = result2["X2"]
            X_sim = result2["simulated"]
            evaluates = result2["evaluates"]
            predicted = result2["predicted"]
            configs = result2["configs"]

            is_discrete = name2.split("_")[0] != "continuous"
            if discrete_sink and not is_discrete:
                continue
            elif not discrete_sink and is_discrete:
                continue

            def _draw(ax, a, b, c):
                range_ = min(*a, *b, *c), max(*a, *b, *c)
                range_ = range_[0] - 0.1 * abs(range_[0]), range_[1] + 0.1 * abs(range_[0])
                
                hist, edges = np.histogram(a, range=range_)
                edges =(edges + (edges[1] - edges[0]) / 2)[:-1]
                width = (edges[1] - edges[0]) / 4
                ax.bar(edges-width, hist, width=width, label="true", color="blue")
                
                hist, _ = np.histogram(b, range=range_)
                ax.bar(edges, hist, width=width, label="linear model", color="lime")
                
                hist, _ = np.histogram(c, range=range_)
                ax.bar(edges+width, hist, width=width, label="random forest", color="tomato")
            
            # 変化前真データと、変化前真データを訓練データとした機械学習に変化前真データを与えたときのシンク予測値
            a = X_true_before[:, sink_index]
            # predicted = (y_true, y_pred)
            b = predicted[('before', 'lr')][1]
            c = predicted[('before', 'rf')][1]
            _draw(axes[count, 0], a, b, c)
            
            # 変化後シミュレーションデータと、変化後シミュレーションデータを訓練データとした機械学習に変化前真データを与えたときのシンク予測値
            a = X_true_after[:, sink_index]
            b = predicted[('simulation', 'lr')][1]
            c = predicted[('simulation', 'rf')][1]
            _draw(axes[count, 1], a, b, c)

            # 変化後真データと、変化前データで訓練したモデルによる変化後シミュレーションの予測値
            a = X_true_after[:, sink_index]
            b = predicted[('before_after', 'lr')][1]
            c = predicted[('before_after', 'rf')][1]
            _draw(axes[count, 2], a, b, c)
            
            # 条件名
            s = "operation: " + name + "\n"
            s += "sink type: " + name2.split("_")[0] + "\n"
            s += "shuffle residual: " + ("True" if len(name2.split("_")) == 2 else "False") + "\n"
            xlim = axes[count, 0].get_xlim()
            x_pos = xlim[1] - xlim[0]
            x_pos = xlim[0] - abs(x_pos) * 0.3
            axes[count, 0].text(x_pos, 0, s, ha="right", va="bottom")

            if count == n_patterns - 1:
                axes[count, 2].legend(bbox_to_anchor=(1, -0.7), loc="center right")
                
            if count == 0:
                axes[0, 0].set_title("true")
                axes[0, 1].set_title("sim")
                axes[0, 2].set_title("est")
                
            count += 1

    plt.tight_layout()
    plt.show()

def draw_pred_hist(n_patterns, results, sink_index):
    _draw_pred_hist(n_patterns // 2, results, sink_index, discrete_sink=False)
    _draw_pred_hist(n_patterns // 2, results, sink_index, discrete_sink=True)
