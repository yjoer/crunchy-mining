from __future__ import annotations

import typing

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from tqdm.auto import tqdm
from xgboost import XGBClassifier

from . import mlflow_util
from .util import aggregate_cv_metrics
from .util import custom_predict
from .util import evaluate_classification
from .util import evaluate_roc
from .util import trace_memory

if typing.TYPE_CHECKING:
    from omegaconf import DictConfig


def inspect_holdout_split_size(train_val_sets: dict, target_variable: str):
    df_train, df_test = train_val_sets["testing"]
    df_train_sm, df_val = train_val_sets["validation"]

    return pd.concat(
        [
            df_train[target_variable].value_counts().rename("train"),
            df_train_sm[target_variable].value_counts().rename("train (sm)"),
            df_val[target_variable].value_counts().rename("validation"),
            df_test[target_variable].value_counts().rename("test"),
        ],
        axis=1,
    )


def inspect_cv_split_size(train_val_sets: dict, target_variable: str):
    folds = []

    for name, (df_train, df_val) in train_val_sets.items():
        if not name.startswith("fold_"):
            continue

        i = name.split("_")[-1]

        count_by_split = pd.concat(
            [
                df_train[target_variable].value_counts().rename("train"),
                df_val[target_variable].value_counts().rename("validation"),
            ],
            axis=1,
        )

        count_by_split.reset_index(inplace=True)
        count_by_split.index = [i] * len(count_by_split)
        count_by_split.index.name = "Folds"

        folds.append(count_by_split)

    return pd.concat(folds).pivot(
        columns=target_variable,
        values=["train", "validation"],
    )


def train_knn(X_train, y_train):
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)

    return knn


def validate_knn(cfg: DictConfig, train_val_sets: dict):
    fixed_fpr = cfg.validation.metrics.fixed_fpr
    memory_legacy = cfg.validation.metrics.memory_usage.legacy

    with mlflow.start_run(run_name="KNN"):
        for name, (X_train, y_train, X_val, y_val) in train_val_sets.items():
            if name == "testing":
                continue

            with mlflow.start_run(run_name=name, nested=True):
                with trace_memory(legacy=memory_legacy) as fit_trace:
                    knn = train_knn(X_train, y_train)

                with trace_memory(legacy=memory_legacy) as score_trace:
                    if fixed_fpr:
                        y_knn_p, roc_raw, roc = evaluate_roc(
                            estimator=knn,
                            X_test=X_val,
                            y_test=y_val,
                            fixed_fpr=fixed_fpr,
                        )

                        y_knn = custom_predict(
                            y_prob=y_knn_p,
                            threshold=roc["threshold"],
                        )
                    else:
                        y_knn = knn.predict(X_val)

                if fixed_fpr:
                    mlflow.log_dict(roc_raw, artifact_file="roc.json")
                    mlflow.log_metrics(roc)

                mlflow.log_metrics(evaluate_classification(y_val, y_knn))
                mlflow.log_metric("fit_time", fit_trace["duration"])
                mlflow.log_metric("fit_memory_peak", fit_trace["peak"])
                mlflow.log_metric("score_time", score_trace["duration"])
                mlflow.log_metric("score_memory_peak", score_trace["peak"])
                mlflow.log_params(knn.get_params())

                mlflow.sklearn.log_model(
                    sk_model=knn,
                    artifact_path="model",
                    signature=mlflow.models.infer_signature(X_val, y_knn),
                )

            for v in ["knn", "fit_trace", "y_knn", "y_knn_p", "roc_raw", "roc", "score_trace"]:  # fmt: skip
                if v in locals():
                    del locals()[v]


def tune_knn(cfg: DictConfig, train_val_sets: dict):
    fixed_fpr = cfg.tuning.metrics.fixed_fpr
    memory_legacy = cfg.tuning.metrics.memory_usage.legacy

    param_grid = ParameterGrid(
        {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"],
            "p": [1, 2],
        }
    )

    with mlflow.start_run(run_name="Grid Search"):
        for idx, params in enumerate(param_grid):
            with mlflow.start_run(run_name=f"Combination {idx + 1}", nested=True):
                metrics_list = []
                params_used = {}

                for name, (X_train, y_train, X_val, y_val) in tqdm(train_val_sets.items()):  # fmt: skip
                    if name == "validation" or name == "testing":
                        continue

                    with trace_memory(legacy=memory_legacy) as fit_trace:
                        knn = KNeighborsClassifier(**params)
                        knn.fit(X_train, y_train)

                    with trace_memory(legacy=memory_legacy) as score_trace:
                        if fixed_fpr:
                            y_knn_p, roc_raw, roc = evaluate_roc(
                                estimator=knn,
                                X_test=X_val,
                                y_test=y_val,
                                fixed_fpr=fixed_fpr,
                            )

                            y_knn = custom_predict(
                                y_prob=y_knn_p,
                                threshold=roc["threshold"],
                            )
                        else:
                            y_knn = knn.predict(X_val)

                    metrics = {}

                    if fixed_fpr:
                        metrics.update(roc)

                    metrics.update(evaluate_classification(y_val, y_knn))
                    metrics["fit_time"] = fit_trace["duration"]
                    metrics["fit_memory_peak"] = fit_trace["peak"]
                    metrics["score_time"] = score_trace["duration"]
                    metrics["score_memory_peak"] = score_trace["peak"]
                    metrics_list.append(metrics)
                    params_used = knn.get_params()

                    for v in ["knn", "fit_trace", "y_knn", "y_knn_p", "roc_raw", "roc", "score_trace"]:  # fmt: skip
                        if v in locals():
                            del locals()[v]

                mlflow.log_metrics(aggregate_cv_metrics(metrics_list))
                mlflow.log_params(params_used)


def train_logistic_regression(X_train, y_train):
    params = {
        "random_state": 12345,
        "n_jobs": -1,
    }

    logreg = LogisticRegression(**params)
    logreg.fit(X_train, y_train)

    return logreg


def validate_logistic_regression(cfg: DictConfig, train_val_sets: dict):
    fixed_fpr = cfg.validation.metrics.fixed_fpr
    memory_legacy = cfg.validation.metrics.memory_usage.legacy

    with mlflow.start_run(run_name="Logistic Regression"):
        for name, (X_train, y_train, X_val, y_val) in train_val_sets.items():
            if name == "testing":
                continue

            with mlflow.start_run(run_name=name, nested=True):
                with trace_memory(legacy=memory_legacy) as fit_trace:
                    logreg = train_logistic_regression(X_train, y_train)

                with trace_memory(legacy=memory_legacy) as score_trace:
                    if fixed_fpr:
                        y_logreg_p, roc_raw, roc = evaluate_roc(
                            estimator=logreg,
                            X_test=X_val,
                            y_test=y_val,
                            fixed_fpr=fixed_fpr,
                        )

                        y_logreg = custom_predict(
                            y_prob=y_logreg_p,
                            threshold=roc["threshold"],
                        )
                    else:
                        y_logreg = logreg.predict(X_val)

                if fixed_fpr:
                    mlflow.log_dict(roc_raw, artifact_file="roc.json")
                    mlflow.log_metrics(roc)

                mlflow.log_metrics(evaluate_classification(y_val, y_logreg))
                mlflow.log_metric("fit_time", fit_trace["duration"])
                mlflow.log_metric("fit_memory_peak", fit_trace["peak"])
                mlflow.log_metric("score_time", score_trace["duration"])
                mlflow.log_metric("score_memory_peak", score_trace["peak"])
                mlflow.log_params(logreg.get_params())

                mlflow.sklearn.log_model(
                    sk_model=logreg,
                    artifact_path="model",
                    signature=mlflow.models.infer_signature(X_val, y_logreg),
                )

            for v in ["logreg", "fit_trace", "y_logreg", "y_logreg_p", "roc_raw", "roc", "score_trace"]:  # fmt: skip
                if v in locals():
                    del locals()[v]


def tune_logistic_regression(cfg: DictConfig, train_val_sets: dict):
    fixed_fpr = cfg.tuning.metrics.fixed_fpr
    memory_legacy = cfg.tuning.metrics.memory_usage.legacy

    param_grid = ParameterGrid(
        {
            "penalty": ["l1", "l2", "elasticnet", None],
            "C": np.logspace(start=-2, stop=2, num=5),
            "class_weight": ["balanced", None],
            "solver": ["lbfgs", "newton-cholesky", "newton-cg", "sag", "saga"],
            "max_iter": [100, 1000],
            "l1_ratio": [0.1, 0.5, 0.9, None],
        }
    )

    param_grid = list(param_grid)

    for params in reversed(param_grid):
        if params["solver"] == "lbfgs" and params["penalty"] not in ["l2", None]:
            param_grid.remove(params)
        elif params["solver"] == "newton-cg" and params["penalty"] not in ["l2", None]:
            param_grid.remove(params)
        elif params["solver"] == "newton-cholesky" and params["penalty"] not in ["l2", None]:  # fmt: skip
            param_grid.remove(params)
        elif params["solver"] == "sag" and params["penalty"] not in ["l2", None]:
            param_grid.remove(params)
        elif params["penalty"] == "elasticnet" and params["l1_ratio"] is None:
            param_grid.remove(params)
        elif params["penalty"] != "elasticnet" and params["l1_ratio"] is not None:
            param_grid.remove(params)

    with mlflow.start_run(run_name="Grid Search"):
        for idx, params in enumerate(param_grid):
            params = {
                **params,
                "random_state": 12345,
                "n_jobs": -1,
            }

            with mlflow.start_run(run_name=f"Combination {idx + 1}", nested=True):
                metrics_list = []
                params_used = {}

                for name, (X_train, y_train, X_val, y_val) in tqdm(train_val_sets.items()):  # fmt: skip
                    if name == "validation" or name == "testing":
                        continue

                    with trace_memory(legacy=memory_legacy) as fit_trace:
                        logreg = LogisticRegression(**params)
                        logreg.fit(X_train, y_train)

                    with trace_memory(legacy=memory_legacy) as score_trace:
                        if fixed_fpr:
                            y_logreg_p, roc_raw, roc = evaluate_roc(
                                estimator=logreg,
                                X_test=X_val,
                                y_test=y_val,
                                fixed_fpr=fixed_fpr,
                            )

                            y_logreg = custom_predict(
                                y_prob=y_logreg_p,
                                threshold=roc["threshold"],
                            )
                        else:
                            y_logreg = logreg.predict(X_val)

                    metrics = {}

                    if fixed_fpr:
                        metrics.update(roc)

                    metrics.update(evaluate_classification(y_val, y_logreg))
                    metrics["fit_time"] = fit_trace["duration"]
                    metrics["fit_memory_peak"] = fit_trace["peak"]
                    metrics["score_time"] = score_trace["duration"]
                    metrics["score_memory_peak"] = score_trace["peak"]
                    metrics_list.append(metrics)
                    params_used = logreg.get_params()

                    for v in ["logreg", "fit_trace", "y_logreg", "y_logreg_p", "roc_raw", "roc", "score_trace"]:  # fmt: skip
                        if v in locals():
                            del locals()[v]

                mlflow.log_metrics(aggregate_cv_metrics(metrics_list))
                mlflow.log_params(params_used)


def train_gaussian_nb(X_train, y_train):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    return gnb


def validate_gaussian_nb(cfg: DictConfig, train_val_sets: dict):
    fixed_fpr = cfg.validation.metrics.fixed_fpr
    memory_legacy = cfg.validation.metrics.memory_usage.legacy

    with mlflow.start_run(run_name="Gaussian NB"):
        for name, (X_train, y_train, X_val, y_val) in train_val_sets.items():
            if name == "testing":
                continue

            with mlflow.start_run(run_name=name, nested=True):
                with trace_memory(legacy=memory_legacy) as fit_trace:
                    gnb = train_gaussian_nb(X_train, y_train)

                with trace_memory(legacy=memory_legacy) as score_trace:
                    if fixed_fpr:
                        y_gnb_p, roc_raw, roc = evaluate_roc(
                            estimator=gnb,
                            X_test=X_val,
                            y_test=y_val,
                            fixed_fpr=fixed_fpr,
                        )

                        y_gnb = custom_predict(
                            y_prob=y_gnb_p,
                            threshold=roc["threshold"],
                        )
                    else:
                        y_gnb = gnb.predict(X_val)

                if fixed_fpr:
                    mlflow.log_dict(roc_raw, artifact_file="roc.json")
                    mlflow.log_metrics(roc)

                mlflow.log_metrics(evaluate_classification(y_val, y_gnb))
                mlflow.log_metric("fit_time", fit_trace["duration"])
                mlflow.log_metric("fit_memory_peak", fit_trace["peak"])
                mlflow.log_metric("score_time", score_trace["duration"])
                mlflow.log_metric("score_memory_peak", score_trace["peak"])
                mlflow.log_params(gnb.get_params())

                mlflow.sklearn.log_model(
                    sk_model=gnb,
                    artifact_path="model",
                    signature=mlflow.models.infer_signature(X_val, y_gnb),
                )

            for v in ["gnb", "fit_trace", "y_gnb", "y_gnb_p", "roc_raw", "roc", "score_trace"]:  # fmt: skip
                if v in locals():
                    del locals()[v]


def tune_gaussian_nb(cfg: DictConfig, train_val_sets: dict):
    fixed_fpr = cfg.tuning.metrics.fixed_fpr
    memory_legacy = cfg.tuning.metrics.memory_usage.legacy

    param_grid = ParameterGrid(
        {
            "var_smoothing": np.logspace(start=0, stop=-9, num=10),
        }
    )

    with mlflow.start_run(run_name="Grid Search"):
        for idx, params in enumerate(param_grid):
            with mlflow.start_run(run_name=f"Combination {idx + 1}", nested=True):
                metrics_list = []
                params_used = {}

                for name, (X_train, y_train, X_val, y_val) in tqdm(train_val_sets.items()):  # fmt: skip
                    if name == "validation" or name == "testing":
                        continue

                    with trace_memory(legacy=memory_legacy) as fit_trace:
                        gnb = GaussianNB(**params)
                        gnb.fit(X_train, y_train)

                    with trace_memory(legacy=memory_legacy) as score_trace:
                        if fixed_fpr:
                            y_gnb_p, roc_raw, roc = evaluate_roc(
                                estimator=gnb,
                                X_test=X_val,
                                y_test=y_val,
                                fixed_fpr=fixed_fpr,
                            )

                            y_gnb = custom_predict(
                                y_prob=y_gnb_p,
                                threshold=roc["threshold"],
                            )
                        else:
                            y_gnb = gnb.predict(X_val)

                    metrics = {}

                    if fixed_fpr:
                        metrics.update(roc)

                    metrics.update(evaluate_classification(y_val, y_gnb))
                    metrics["fit_time"] = fit_trace["duration"]
                    metrics["fit_memory_peak"] = fit_trace["peak"]
                    metrics["score_time"] = score_trace["duration"]
                    metrics["score_memory_peak"] = score_trace["peak"]
                    metrics_list.append(metrics)
                    params_used = gnb.get_params()

                    for v in ["gnb", "fit_trace", "y_gnb", "y_gnb_p", "roc_raw", "roc", "score_trace"]:  # fmt: skip
                        if v in locals():
                            del locals()[v]

                mlflow.log_metrics(aggregate_cv_metrics(metrics_list))
                mlflow.log_params(params_used)


def train_linear_svc(X_train, y_train):
    params = {
        "dual": "auto",
        "random_state": 12345,
    }

    svc = LinearSVC(**params)
    svc.fit(X_train, y_train)

    return svc


def train_calibrated_linear_svc(X_train, y_train):
    params = {
        "n_jobs": -1,
    }

    svc = CalibratedClassifierCV(**params)
    svc.fit(X_train, y_train)

    return svc


def validate_linear_svc(cfg: DictConfig, train_val_sets: dict):
    fixed_fpr = cfg.validation.metrics.fixed_fpr
    memory_legacy = cfg.validation.metrics.memory_usage.legacy

    with mlflow.start_run(run_name="Linear SVC"):
        for name, (X_train, y_train, X_val, y_val) in train_val_sets.items():
            if name == "testing":
                continue

            with mlflow.start_run(run_name=name, nested=True):
                with trace_memory(legacy=memory_legacy) as fit_trace:
                    if fixed_fpr:
                        svc = train_calibrated_linear_svc(X_train, y_train)
                    else:
                        svc = train_linear_svc(X_train, y_train)

                with trace_memory(legacy=memory_legacy) as score_trace:
                    if fixed_fpr:
                        y_svc_p, roc_raw, roc = evaluate_roc(
                            estimator=svc,
                            X_test=X_val,
                            y_test=y_val,
                            fixed_fpr=fixed_fpr,
                        )

                        y_svc = custom_predict(
                            y_prob=y_svc_p,
                            threshold=roc["threshold"],
                        )
                    else:
                        y_svc = svc.predict(X_val)

                if fixed_fpr:
                    mlflow.log_dict(roc_raw, artifact_file="roc.json")
                    mlflow.log_metrics(roc)

                mlflow.log_metrics(evaluate_classification(y_val, y_svc))
                mlflow.log_metric("fit_time", fit_trace["duration"])
                mlflow.log_metric("fit_memory_peak", fit_trace["peak"])
                mlflow.log_metric("score_time", score_trace["duration"])
                mlflow.log_metric("score_memory_peak", score_trace["peak"])
                mlflow.log_params(svc.get_params())

                mlflow.sklearn.log_model(
                    sk_model=svc,
                    artifact_path="model",
                    signature=mlflow.models.infer_signature(X_val, y_svc),
                )

            for v in ["svc", "fit_trace", "y_svc", "y_svc_p", "roc_raw", "roc", "score_trace"]:  # fmt: skip
                if v in locals():
                    del locals()[v]


def tune_linear_svc(cfg: DictConfig, train_val_sets: dict):
    fixed_fpr = cfg.tuning.metrics.fixed_fpr
    memory_legacy = cfg.tuning.metrics.memory_usage.legacy

    param_grid = ParameterGrid(
        {
            "penalty": ["l1", "l2"],
            "loss": ["hinge", "squared_hinge"],
            "C": np.logspace(start=-2, stop=2, num=5),
            "class_weight": ["balanced", None],
            "method": ["sigmoid", "isotonic"],
        }
    )

    param_grid = list(param_grid)

    for params in reversed(param_grid):
        if params["penalty"] == "l1" and params["loss"] == "hinge":
            param_grid.remove(params)

    with mlflow.start_run(run_name="Grid Search"):
        for idx, params in enumerate(param_grid):
            params_svc = {
                "method": params["method"],
                "n_jobs": -1,
            }

            del params["method"]

            params_base = {
                **params,
                "dual": "auto",
                "random_state": 12345,
            }

            with mlflow.start_run(run_name=f"Combination {idx + 1}", nested=True):
                metrics_list = []
                params_used = {}

                for name, (X_train, y_train, X_val, y_val) in tqdm(train_val_sets.items()):  # fmt: skip
                    if name == "validation" or name == "testing":
                        continue

                    with trace_memory(legacy=memory_legacy) as fit_trace:
                        base = LinearSVC(**params_base)
                        svc = CalibratedClassifierCV(estimator=base, **params_svc)
                        svc.fit(X_train, y_train)

                    with trace_memory(legacy=memory_legacy) as score_trace:
                        if fixed_fpr:
                            y_svc_p, roc_raw, roc = evaluate_roc(
                                estimator=svc,
                                X_test=X_val,
                                y_test=y_val,
                                fixed_fpr=fixed_fpr,
                            )

                            y_svc = custom_predict(
                                y_prob=y_svc_p,
                                threshold=roc["threshold"],
                            )
                        else:
                            y_svc = svc.predict(X_val)

                    metrics = {}

                    if fixed_fpr:
                        metrics.update(roc)

                    metrics.update(evaluate_classification(y_val, y_svc))
                    metrics["fit_time"] = fit_trace["duration"]
                    metrics["fit_memory_peak"] = fit_trace["peak"]
                    metrics["score_time"] = score_trace["duration"]
                    metrics["score_memory_peak"] = score_trace["peak"]
                    metrics_list.append(metrics)
                    params_used = svc.get_params()

                    for v in ["svc", "fit_trace", "y_svc", "y_svc_p", "roc_raw", "roc", "score_trace"]:  # fmt: skip
                        if v in locals():
                            del locals()[v]

                mlflow.log_metrics(aggregate_cv_metrics(metrics_list))
                mlflow.log_params(params_used)


def train_decision_tree(X_train, y_train):
    params = {
        "random_state": 12345,
    }

    dt = DecisionTreeClassifier(**params)
    dt.fit(X_train, y_train)

    return dt


def validate_decision_tree(cfg: DictConfig, train_val_sets: dict):
    fixed_fpr = cfg.validation.metrics.fixed_fpr
    memory_legacy = cfg.validation.metrics.memory_usage.legacy

    with mlflow.start_run(run_name="Decision Tree"):
        for name, (X_train, y_train, X_val, y_val) in train_val_sets.items():
            if name == "testing":
                continue

            with mlflow.start_run(run_name=name, nested=True):
                with trace_memory(legacy=memory_legacy) as fit_trace:
                    dt = train_decision_tree(X_train, y_train)

                with trace_memory(legacy=memory_legacy) as score_trace:
                    if fixed_fpr:
                        y_dt_p, roc_raw, roc = evaluate_roc(
                            estimator=dt,
                            X_test=X_val,
                            y_test=y_val,
                            fixed_fpr=fixed_fpr,
                        )

                        y_dt = custom_predict(
                            y_prob=y_dt_p,
                            threshold=roc["threshold"],
                        )
                    else:
                        y_dt = dt.predict(X_val)

                if fixed_fpr:
                    mlflow.log_dict(roc_raw, artifact_file="roc.json")
                    mlflow.log_metrics(roc)

                mlflow.log_metrics(evaluate_classification(y_val, y_dt))
                mlflow.log_metric("fit_time", fit_trace["duration"])
                mlflow.log_metric("fit_memory_peak", fit_trace["peak"])
                mlflow.log_metric("score_time", score_trace["duration"])
                mlflow.log_metric("score_memory_peak", score_trace["peak"])
                mlflow.log_params(dt.get_params())

                mlflow.sklearn.log_model(
                    sk_model=dt,
                    artifact_path="model",
                    signature=mlflow.models.infer_signature(X_val, y_dt),
                )

            for v in ["dt", "fit_trace", "y_dt", "y_dt_p", "roc_raw", "roc", "score_trace"]:  # fmt: skip
                if v in locals():
                    del locals()[v]


def tune_decision_tree(cfg: DictConfig, train_val_sets: dict):
    fixed_fpr = cfg.tuning.metrics.fixed_fpr
    memory_legacy = cfg.tuning.metrics.memory_usage.legacy

    param_grid = ParameterGrid(
        {
            "criterion": ["gini", "entropy", "log_loss"],
            "max_depth": [3, 9, 27, None],
            "min_samples_split": [2, 8, 32],
            "min_samples_leaf": [1, 5, 10],
            "max_features": ["sqrt", "log2", None],
            "max_leaf_nodes": [10, 50, 100, None],
            "class_weight": ["balanced", None],
        }
    )

    with mlflow.start_run(run_name="Grid Search"):
        for idx, params in enumerate(param_grid):
            params = {
                **params,
                "random_state": 12345,
            }

            with mlflow.start_run(run_name=f"Combination {idx + 1}", nested=True):
                metrics_list = []
                params_used = {}

                for name, (X_train, y_train, X_val, y_val) in tqdm(train_val_sets.items()):  # fmt: skip
                    if name == "validation" or name == "testing":
                        continue

                    with trace_memory(legacy=memory_legacy) as fit_trace:
                        dt = DecisionTreeClassifier(**params)
                        dt.fit(X_train, y_train)

                    with trace_memory(legacy=memory_legacy) as score_trace:
                        if fixed_fpr:
                            y_dt_p, roc_raw, roc = evaluate_roc(
                                estimator=dt,
                                X_test=X_val,
                                y_test=y_val,
                                fixed_fpr=fixed_fpr,
                            )

                            y_dt = custom_predict(
                                y_prob=y_dt_p,
                                threshold=roc["threshold"],
                            )
                        else:
                            y_dt = dt.predict(X_val)

                    metrics = {}

                    if fixed_fpr:
                        metrics.update(roc)

                    metrics.update(evaluate_classification(y_val, y_dt))
                    metrics["fit_time"] = fit_trace["duration"]
                    metrics["fit_memory_peak"] = fit_trace["peak"]
                    metrics["score_time"] = score_trace["duration"]
                    metrics["score_memory_peak"] = score_trace["peak"]
                    metrics_list.append(metrics)
                    params_used = dt.get_params()

                    for v in ["dt", "fit_trace", "y_dt", "y_dt_p", "roc_raw", "roc", "score_trace"]:  # fmt: skip
                        if v in locals():
                            del locals()[v]

                mlflow.log_metrics(aggregate_cv_metrics(metrics_list))
                mlflow.log_params(params_used)


def train_random_forest(X_train, y_train):
    params = {
        "n_jobs": -1,
        "random_state": 12345,
    }

    rf = RandomForestClassifier(**params)
    rf.fit(X_train, y_train)

    return rf


def validate_random_forest(cfg: DictConfig, train_val_sets: dict):
    fixed_fpr = cfg.validation.metrics.fixed_fpr
    memory_legacy = cfg.validation.metrics.memory_usage.legacy

    with mlflow.start_run(run_name="Random Forest"):
        for name, (X_train, y_train, X_val, y_val) in train_val_sets.items():
            if name == "testing":
                continue

            with mlflow.start_run(run_name=name, nested=True):
                with trace_memory(legacy=memory_legacy) as fit_trace:
                    rf = train_random_forest(X_train, y_train)

                with trace_memory(legacy=memory_legacy) as score_trace:
                    if fixed_fpr:
                        y_rf_p, roc_raw, roc = evaluate_roc(
                            estimator=rf,
                            X_test=X_val,
                            y_test=y_val,
                            fixed_fpr=fixed_fpr,
                        )

                        y_rf = custom_predict(
                            y_prob=y_rf_p,
                            threshold=roc["threshold"],
                        )
                    else:
                        y_rf = rf.predict(X_val)

                if fixed_fpr:
                    mlflow.log_dict(roc_raw, artifact_file="roc.json")
                    mlflow.log_metrics(roc)

                mlflow.log_metrics(evaluate_classification(y_val, y_rf))
                mlflow.log_metric("fit_time", fit_trace["duration"])
                mlflow.log_metric("fit_memory_peak", fit_trace["peak"])
                mlflow.log_metric("score_time", score_trace["duration"])
                mlflow.log_metric("score_memory_peak", score_trace["peak"])
                mlflow.log_params(rf.get_params())

                mlflow.sklearn.log_model(
                    sk_model=rf,
                    artifact_path="model",
                    signature=mlflow.models.infer_signature(X_val, y_rf),
                )

            for v in ["rf", "fit_trace", "y_rf", "y_rf_p", "roc_raw", "roc", "score_trace"]:  # fmt: skip
                if v in locals():
                    del locals()[v]


def tune_random_forest(cfg: DictConfig, train_val_sets: dict):
    fixed_fpr = cfg.tuning.metrics.fixed_fpr
    memory_legacy = cfg.tuning.metrics.memory_usage.legacy

    param_grid = ParameterGrid(
        {
            "n_estimators": [50, 100, 200],
            "criterion": ["gini", "entropy"],
            "max_depth": [3, 30, None],
            "min_samples_split": [2, 32],
            "min_samples_leaf": [1, 10],
            "max_features": ["sqrt", "log2", None],
            "max_leaf_nodes": [10, 100, None],
            "class_weight": [None],
        }
    )

    with mlflow.start_run(run_name="Grid Search"):
        for idx, params in enumerate(param_grid):
            params = {
                **params,
                "n_jobs": -1,
                "random_state": 12345,
            }

            with mlflow.start_run(run_name=f"Combination {idx + 1}", nested=True):
                metrics_list = []
                params_used = {}

                for name, (X_train, y_train, X_val, y_val) in tqdm(train_val_sets.items()):  # fmt: skip
                    if name == "validation" or name == "testing":
                        continue

                    with trace_memory(legacy=memory_legacy) as fit_trace:
                        rf = RandomForestClassifier(**params)
                        rf.fit(X_train, y_train)

                    with trace_memory(legacy=memory_legacy) as score_trace:
                        if fixed_fpr:
                            y_rf_p, roc_raw, roc = evaluate_roc(
                                estimator=rf,
                                X_test=X_val,
                                y_test=y_val,
                                fixed_fpr=fixed_fpr,
                            )

                            y_rf = custom_predict(
                                y_prob=y_rf_p,
                                threshold=roc["threshold"],
                            )
                        else:
                            y_rf = rf.predict(X_val)

                    metrics = {}

                    if fixed_fpr:
                        metrics.update(roc)

                    metrics.update(evaluate_classification(y_val, y_rf))
                    metrics["fit_time"] = fit_trace["duration"]
                    metrics["fit_memory_peak"] = fit_trace["peak"]
                    metrics["score_time"] = score_trace["duration"]
                    metrics["score_memory_peak"] = score_trace["peak"]
                    metrics_list.append(metrics)
                    params_used = rf.get_params()

                    for v in ["rf", "fit_trace", "y_rf", "y_rf_p", "roc_raw", "roc", "score_trace"]:  # fmt: skip
                        if v in locals():
                            del locals()[v]

                mlflow.log_metrics(aggregate_cv_metrics(metrics_list))
                mlflow.log_params(params_used)


def train_adaboost(X_train, y_train):
    params = {
        "algorithm": "SAMME",
        "random_state": 12345,
    }

    ab = AdaBoostClassifier(**params)
    ab.fit(X_train, y_train)

    return ab


def validate_adaboost(cfg: DictConfig, train_val_sets: dict):
    fixed_fpr = cfg.validation.metrics.fixed_fpr
    memory_legacy = cfg.validation.metrics.memory_usage.legacy

    with mlflow.start_run(run_name="AdaBoost"):
        for name, (X_train, y_train, X_val, y_val) in train_val_sets.items():
            if name == "testing":
                continue

            with mlflow.start_run(run_name=name, nested=True):
                with trace_memory(legacy=memory_legacy) as fit_trace:
                    ab = train_adaboost(X_train, y_train)

                with trace_memory(legacy=memory_legacy) as score_trace:
                    if fixed_fpr:
                        y_ab_p, roc_raw, roc = evaluate_roc(
                            estimator=ab,
                            X_test=X_val,
                            y_test=y_val,
                            fixed_fpr=fixed_fpr,
                        )

                        y_ab = custom_predict(
                            y_prob=y_ab_p,
                            threshold=roc["threshold"],
                        )
                    else:
                        y_ab = ab.predict(X_val)

                if fixed_fpr:
                    mlflow.log_dict(roc_raw, artifact_file="roc.json")
                    mlflow.log_metrics(roc)

                mlflow.log_metrics(evaluate_classification(y_val, y_ab))
                mlflow.log_metric("fit_time", fit_trace["duration"])
                mlflow.log_metric("fit_memory_peak", fit_trace["peak"])
                mlflow.log_metric("score_time", score_trace["duration"])
                mlflow.log_metric("score_memory_peak", score_trace["peak"])
                mlflow.log_params(ab.get_params())

                mlflow.sklearn.log_model(
                    sk_model=ab,
                    artifact_path="model",
                    signature=mlflow.models.infer_signature(X_val, y_ab),
                )

            for v in ["ab", "fit_trace", "y_ab", "y_ab_p", "roc_raw", "roc", "score_trace"]:  # fmt: skip
                if v in locals():
                    del locals()[v]


def tune_adaboost(cfg: DictConfig, train_val_sets: dict):
    fixed_fpr = cfg.tuning.metrics.fixed_fpr
    memory_legacy = cfg.tuning.metrics.memory_usage.legacy

    param_grid = ParameterGrid(
        {
            "n_estimators": [50, 100, 200],
            "learning_rate": np.logspace(start=-5, stop=0, num=6),
        }
    )

    with mlflow.start_run(run_name="Grid Search"):
        for idx, params in enumerate(param_grid):
            params = {
                **params,
                "algorithm": "SAMME",
                "random_state": 12345,
            }

            with mlflow.start_run(run_name=f"Combination {idx + 1}", nested=True):
                metrics_list = []
                params_used = {}

                for name, (X_train, y_train, X_val, y_val) in tqdm(train_val_sets.items()):  # fmt: skip
                    if name == "validation" or name == "testing":
                        continue

                    with trace_memory(legacy=memory_legacy) as fit_trace:
                        ab = AdaBoostClassifier(**params)
                        ab.fit(X_train, y_train)

                    with trace_memory(legacy=memory_legacy) as score_trace:
                        if fixed_fpr:
                            y_ab_p, roc_raw, roc = evaluate_roc(
                                estimator=ab,
                                X_test=X_val,
                                y_test=y_val,
                                fixed_fpr=fixed_fpr,
                            )

                            y_ab = custom_predict(
                                y_prob=y_ab_p,
                                threshold=roc["threshold"],
                            )
                        else:
                            y_ab = ab.predict(X_val)

                    metrics = {}

                    if fixed_fpr:
                        metrics.update(roc)

                    metrics.update(evaluate_classification(y_val, y_ab))
                    metrics["fit_time"] = fit_trace["duration"]
                    metrics["fit_memory_peak"] = fit_trace["peak"]
                    metrics["score_time"] = score_trace["duration"]
                    metrics["score_memory_peak"] = score_trace["peak"]
                    metrics_list.append(metrics)
                    params_used = ab.get_params()

                    for v in ["ab", "fit_trace", "y_ab", "y_ab_p", "roc_raw", "roc", "score_trace"]:  # fmt: skip
                        if v in locals():
                            del locals()[v]

                mlflow.log_metrics(aggregate_cv_metrics(metrics_list))
                mlflow.log_params(params_used)


def train_xgboost(X_train, y_train):
    params = {
        "n_jobs": -1,
        "random_state": 12345,
    }

    xgb = XGBClassifier(**params)
    xgb.fit(X_train, y_train)

    return xgb


def validate_xgboost(cfg: DictConfig, train_val_sets: dict):
    fixed_fpr = cfg.validation.metrics.fixed_fpr
    memory_legacy = cfg.validation.metrics.memory_usage.legacy

    with mlflow.start_run(run_name="XGBoost"):
        for name, (X_train, y_train, X_val, y_val) in train_val_sets.items():
            if name == "testing":
                continue

            with mlflow.start_run(run_name=name, nested=True):
                with trace_memory(legacy=memory_legacy) as fit_trace:
                    xgb = train_xgboost(X_train, y_train)

                with trace_memory(legacy=memory_legacy) as score_trace:
                    if fixed_fpr:
                        y_xgb_p, roc_raw, roc = evaluate_roc(
                            estimator=xgb,
                            X_test=X_val,
                            y_test=y_val,
                            fixed_fpr=fixed_fpr,
                        )

                        y_xgb = custom_predict(
                            y_prob=y_xgb_p,
                            threshold=roc["threshold"],
                        )
                    else:
                        y_xgb = xgb.predict(X_val)

                if fixed_fpr:
                    mlflow.log_dict(roc_raw, artifact_file="roc.json")
                    mlflow.log_metrics(roc)

                mlflow.log_metrics(evaluate_classification(y_val, y_xgb))
                mlflow.log_metric("fit_time", fit_trace["duration"])
                mlflow.log_metric("fit_memory_peak", fit_trace["peak"])
                mlflow.log_metric("score_time", score_trace["duration"])
                mlflow.log_metric("score_memory_peak", score_trace["peak"])
                mlflow.log_params(xgb.get_params())

                mlflow.xgboost.log_model(
                    xgb_model=xgb,
                    artifact_path="model",
                    signature=mlflow.models.infer_signature(X_val, y_xgb),
                    model_format="json",
                )

            for v in ["xgb", "fit_trace", "y_xgb", "y_xgb_p", "roc_raw", "roc", "score_trace"]:  # fmt: skip
                if v in locals():
                    del locals()[v]


def tune_xgboost(cfg: DictConfig, train_val_sets: dict):
    fixed_fpr = cfg.tuning.metrics.fixed_fpr
    memory_legacy = cfg.tuning.metrics.memory_usage.legacy

    # https://xgboost.readthedocs.io/en/stable/parameter.html
    # https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html
    param_grid = ParameterGrid(
        {
            "learning_rate": [0.01, 0.1, 0.2, 0.3],
            "min_split_loss": [0, 5, 10],
            "max_depth": [3, 6, 9],
            "min_child_weight": [1, 5, 9],
            "subsample": [0.5, 1],
            "colsample_bytree": [0.5, 1],
            # "reg_alpha": [],
            # "reg_lambda": [],
            # "scale_pos_weight": [],
        }
    )

    with mlflow.start_run(run_name="Grid Search"):
        for idx, params in enumerate(param_grid):
            params = {
                **params,
                "n_jobs": -1,
                "random_state": 12345,
            }

            with mlflow.start_run(run_name=f"Combination {idx + 1}", nested=True):
                metrics_list = []
                params_used = {}

                for name, (X_train, y_train, X_val, y_val) in tqdm(train_val_sets.items()):  # fmt: skip
                    if name == "validation" or name == "testing":
                        continue

                    with trace_memory(legacy=memory_legacy) as fit_trace:
                        xgb = XGBClassifier(**params)
                        xgb.fit(X_train, y_train)

                    with trace_memory(legacy=memory_legacy) as score_trace:
                        if fixed_fpr:
                            y_xgb_p, roc_raw, roc = evaluate_roc(
                                estimator=xgb,
                                X_test=X_val,
                                y_test=y_val,
                                fixed_fpr=fixed_fpr,
                            )

                            y_xgb = custom_predict(
                                y_prob=y_xgb_p,
                                threshold=roc["threshold"],
                            )
                        else:
                            y_xgb = xgb.predict(X_val)

                    metrics = {}

                    if fixed_fpr:
                        metrics.update(roc)

                    metrics.update(evaluate_classification(y_val, y_xgb))
                    metrics["fit_time"] = fit_trace["duration"]
                    metrics["fit_memory_peak"] = fit_trace["peak"]
                    metrics["score_time"] = score_trace["duration"]
                    metrics["score_memory_peak"] = score_trace["peak"]
                    metrics_list.append(metrics)
                    params_used = xgb.get_params()

                    for v in ["xgb", "fit_trace", "y_xgb", "y_xgb_p", "roc_raw", "roc", "score_trace"]:  # fmt: skip
                        if v in locals():
                            del locals()[v]

                mlflow.log_metrics(aggregate_cv_metrics(metrics_list))
                mlflow.log_params(params_used)


def train_lightgbm(X_train, y_train):
    params = {
        "random_state": 12345,
        "n_jobs": -1,
    }

    lgb = LGBMClassifier(**params)
    lgb.fit(X_train, y_train)

    return lgb


def validate_lightgbm(cfg: DictConfig, train_val_sets: dict):
    fixed_fpr = cfg.validation.metrics.fixed_fpr
    memory_legacy = cfg.validation.metrics.memory_usage.legacy

    with mlflow.start_run(run_name="LightGBM"):
        for name, (X_train, y_train, X_val, y_val) in train_val_sets.items():
            if name == "testing":
                continue

            with mlflow.start_run(run_name=name, nested=True):
                with trace_memory(legacy=memory_legacy) as fit_trace:
                    lgb = train_lightgbm(X_train, y_train)

                with trace_memory(legacy=memory_legacy) as score_trace:
                    if fixed_fpr:
                        y_lgb_p, roc_raw, roc = evaluate_roc(
                            estimator=lgb,
                            X_test=X_val,
                            y_test=y_val,
                            fixed_fpr=fixed_fpr,
                        )

                        y_lgb = custom_predict(
                            y_prob=y_lgb_p,
                            threshold=roc["threshold"],
                        )
                    else:
                        y_lgb = lgb.predict(X_val)

                if fixed_fpr:
                    mlflow.log_dict(roc_raw, artifact_file="roc.json")
                    mlflow.log_metrics(roc)

                mlflow.log_metrics(evaluate_classification(y_val, y_lgb))
                mlflow.log_metric("fit_time", fit_trace["duration"])
                mlflow.log_metric("fit_memory_peak", fit_trace["peak"])
                mlflow.log_metric("score_time", score_trace["duration"])
                mlflow.log_metric("score_memory_peak", score_trace["peak"])
                mlflow.log_params(lgb.get_params())

                mlflow.lightgbm.log_model(
                    lgb_model=lgb,
                    artifact_path="model",
                    signature=mlflow.models.infer_signature(X_val, y_lgb),
                )

            for v in ["lgb", "fit_trace", "y_lgb", "y_lgb_p", "roc_raw", "roc", "score_trace"]:  # fmt: skip
                if v in locals():
                    del locals()[v]


def tune_lightgbm(cfg: DictConfig, train_val_sets: dict):
    fixed_fpr = cfg.tuning.metrics.fixed_fpr
    memory_legacy = cfg.tuning.metrics.memory_usage.legacy

    # https://lightgbm.readthedocs.io/en/latest/Parameters.html
    param_grid = ParameterGrid(
        {
            "learning_rate": [0.01, 0.1, 0.2, 0.3],
            "num_leaves": [8, 16, 32],
            "max_depth": [-1, 3, 6, 9],
            "subsample": [0.5, 1],
            "colsample_bytree": [0.5, 1],
            "is_unbalance": [True, False],
        }
    )

    with mlflow.start_run(run_name="Grid Search"):
        for idx, params in enumerate(param_grid):
            params = {
                **params,
                "random_state": 12345,
                "n_jobs": -1,
            }

            with mlflow.start_run(run_name=f"Combination {idx + 1}", nested=True):
                metrics_list = []
                params_used = {}

                for name, (X_train, y_train, X_val, y_val) in tqdm(train_val_sets.items()):  # fmt: skip
                    if name == "validation" or name == "testing":
                        continue

                    with trace_memory(legacy=memory_legacy) as fit_trace:
                        lgb = LGBMClassifier(**params)
                        lgb.fit(X_train, y_train)

                    with trace_memory(legacy=memory_legacy) as score_trace:
                        if fixed_fpr:
                            y_lgb_p, roc_raw, roc = evaluate_roc(
                                estimator=lgb,
                                X_test=X_val,
                                y_test=y_val,
                                fixed_fpr=fixed_fpr,
                            )

                            y_lgb = custom_predict(
                                y_prob=y_lgb_p,
                                threshold=roc["threshold"],
                            )
                        else:
                            y_lgb = lgb.predict(X_val)

                    metrics = {}

                    if fixed_fpr:
                        metrics.update(roc)

                    metrics.update(evaluate_classification(y_val, y_lgb))
                    metrics["fit_time"] = fit_trace["duration"]
                    metrics["fit_memory_peak"] = fit_trace["peak"]
                    metrics["score_time"] = score_trace["duration"]
                    metrics["score_memory_peak"] = score_trace["peak"]
                    metrics_list.append(metrics)
                    params_used = lgb.get_params()

                    for v in ["lgb", "fit_trace", "y_lgb", "y_lgb_p", "roc_raw", "roc", "score_trace"]:  # fmt: skip
                        if v in locals():
                            del locals()[v]

                mlflow.log_metrics(aggregate_cv_metrics(metrics_list))
                mlflow.log_params(params_used)


def train_catboost(X_train, y_train):
    params = {
        "metric_period": 250,
        "random_state": 12345,
    }

    catb = CatBoostClassifier(**params)
    catb.fit(X_train, y_train)

    return catb


def validate_catboost(cfg: DictConfig, train_val_sets: dict):
    fixed_fpr = cfg.validation.metrics.fixed_fpr
    memory_legacy = cfg.validation.metrics.memory_usage.legacy

    with mlflow.start_run(run_name="CatBoost"):
        for name, (X_train, y_train, X_val, y_val) in train_val_sets.items():
            if name == "testing":
                continue

            with mlflow.start_run(run_name=name, nested=True):
                with trace_memory(legacy=memory_legacy) as fit_trace:
                    catb = train_catboost(X_train, y_train)

                with trace_memory(legacy=memory_legacy) as score_trace:
                    if fixed_fpr:
                        y_catb_p, roc_raw, roc = evaluate_roc(
                            estimator=catb,
                            X_test=X_val,
                            y_test=y_val,
                            fixed_fpr=fixed_fpr,
                        )

                        y_catb = custom_predict(
                            y_prob=y_catb_p,
                            threshold=roc["threshold"],
                        )
                    else:
                        y_catb = catb.predict(X_val)

                if fixed_fpr:
                    mlflow.log_dict(roc_raw, artifact_file="roc.json")
                    mlflow.log_metrics(roc)

                mlflow.log_metrics(evaluate_classification(y_val, y_catb))
                mlflow.log_metric("fit_time", fit_trace["duration"])
                mlflow.log_metric("fit_memory_peak", fit_trace["peak"])
                mlflow.log_metric("score_time", score_trace["duration"])
                mlflow.log_metric("score_memory_peak", score_trace["peak"])
                mlflow.log_params(catb.get_params())

                mlflow.catboost.log_model(
                    cb_model=catb,
                    artifact_path="model",
                    signature=mlflow.models.infer_signature(X_val, y_catb),
                )

            for v in ["catb", "fit_trace", "y_catb", "y_catb_p", "roc_raw", "roc", "score_trace"]:  # fmt: skip
                if v in locals():
                    del locals()[v]


def tune_catboost(cfg: DictConfig, train_val_sets: dict):
    fixed_fpr = cfg.tuning.metrics.fixed_fpr
    memory_legacy = cfg.tuning.metrics.memory_usage.legacy

    param_grid = ParameterGrid(
        {
            "learning_rate": [0.001, 0.01, 0.03, 0.1],
            "depth": [3, 6, 9],
        }
    )

    with mlflow.start_run(run_name="Grid Search"):
        for idx, params in enumerate(param_grid):
            params = {
                **params,
                "metric_period": 250,
                "random_state": 12345,
            }

            with mlflow.start_run(run_name=f"Combination {idx + 1}", nested=True):
                metrics_list = []
                params_used = {}

                for name, (X_train, y_train, X_val, y_val) in tqdm(train_val_sets.items()):  # fmt: skip
                    if name == "validation" or name == "testing":
                        continue

                    with trace_memory(legacy=memory_legacy) as fit_trace:
                        catb = CatBoostClassifier(**params)
                        catb.fit(X_train, y_train)

                    with trace_memory(legacy=memory_legacy) as score_trace:
                        if fixed_fpr:
                            y_catb_p, roc_raw, roc = evaluate_roc(
                                estimator=catb,
                                X_test=X_val,
                                y_test=y_val,
                                fixed_fpr=fixed_fpr,
                            )

                            y_catb = custom_predict(
                                y_prob=y_catb_p,
                                threshold=roc["threshold"],
                            )
                        else:
                            y_catb = catb.predict(X_val)

                    metrics = {}

                    if fixed_fpr:
                        metrics.update(roc)

                    metrics.update(evaluate_classification(y_val, y_catb))
                    metrics["fit_time"] = fit_trace["duration"]
                    metrics["fit_memory_peak"] = fit_trace["peak"]
                    metrics["score_time"] = score_trace["duration"]
                    metrics["score_memory_peak"] = score_trace["peak"]
                    metrics_list.append(metrics)
                    params_used = catb.get_params()

                    for v in ["catb", "fit_trace", "y_catb", "y_catb_p", "roc_raw", "roc", "score_trace"]:  # fmt: skip
                        if v in locals():
                            del locals()[v]

                mlflow.log_metrics(aggregate_cv_metrics(metrics_list))
                mlflow.log_params(params_used)


def intrinsic_linear(cfg: DictConfig, train_val_sets: dict, model_name: str):
    feature_names = cfg.vars.categorical + cfg.vars.numerical

    for name, (X_train, _, _, _) in tqdm(train_val_sets.items()):
        parent_run_id = mlflow_util.get_latest_run_id_by_name(model_name)
        run_id = mlflow_util.get_nested_run_ids_by_parent_id(parent_run_id, name=name)

        if not run_id:
            continue

        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

        contributions = np.abs(np.std(X_train, axis=0) * model.coef_[0])
        importances = contributions / np.sum(contributions)

        df = pd.DataFrame({"feature_names": feature_names, "importances": importances})
        mlflow_util.log_table(
            data=df,
            artifact_file="interpretation/intrinsic.json",
            run_id=run_id,
        )


def intrinsic_calibrated_svc(cfg: DictConfig, train_val_sets: dict):
    feature_names = cfg.vars.categorical + cfg.vars.numerical

    for name, (X_train, _, _, _) in tqdm(train_val_sets.items()):
        parent_run_id = mlflow_util.get_latest_run_id_by_name("Linear SVC")
        run_id = mlflow_util.get_nested_run_ids_by_parent_id(parent_run_id, name=name)

        if not run_id:
            continue

        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        coefs = [clf.estimator.coef_ for clf in model.calibrated_classifiers_]
        coef_ = np.mean(coefs, axis=0)

        contributions = np.abs(np.std(X_train, axis=0) * coef_[0])
        importances = contributions / np.sum(contributions)

        df = pd.DataFrame({"feature_names": feature_names, "importances": importances})
        mlflow_util.log_table(
            data=df,
            artifact_file="interpretation/intrinsic.json",
            run_id=run_id,
        )


def intrinsic_trees(cfg: DictConfig, train_val_sets: dict, model_name: str):
    feature_names = cfg.vars.categorical + cfg.vars.numerical

    for name, _ in tqdm(train_val_sets.items()):
        parent_run_id = mlflow_util.get_latest_run_id_by_name(model_name)
        run_id = mlflow_util.get_nested_run_ids_by_parent_id(parent_run_id, name=name)

        if not run_id:
            continue

        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

        importances = model.feature_importances_

        df = pd.DataFrame({"feature_names": feature_names, "importances": importances})
        mlflow_util.log_table(
            data=df,
            artifact_file="interpretation/intrinsic.json",
            run_id=run_id,
        )


def intrinsic_xgboost(cfg: DictConfig, train_val_sets: dict):
    feature_names = cfg.vars.categorical + cfg.vars.numerical

    for name, _ in tqdm(train_val_sets.items()):
        parent_run_id = mlflow_util.get_latest_run_id_by_name("XGBoost")
        run_id = mlflow_util.get_nested_run_ids_by_parent_id(parent_run_id, name=name)

        if not run_id:
            continue

        model = mlflow.xgboost.load_model(f"runs:/{run_id}/model")

        importances = model.feature_importances_

        df = pd.DataFrame({"feature_names": feature_names, "importances": importances})
        mlflow_util.log_table(
            data=df,
            artifact_file="interpretation/intrinsic.json",
            run_id=run_id,
        )


def intrinsic_lightgbm(cfg: DictConfig, train_val_sets: dict):
    feature_names = cfg.vars.categorical + cfg.vars.numerical

    for name, _ in tqdm(train_val_sets.items()):
        parent_run_id = mlflow_util.get_latest_run_id_by_name("LightGBM")
        run_id = mlflow_util.get_nested_run_ids_by_parent_id(parent_run_id, name=name)

        if not run_id:
            continue

        model = mlflow.lightgbm.load_model(f"runs:/{run_id}/model")

        # Use gain as the importance type and normalize to align with XGBoost.
        gains = model.booster_.feature_importance(importance_type="gain")
        importances = gains / np.sum(gains)

        df = pd.DataFrame({"feature_names": feature_names, "importances": importances})
        mlflow_util.log_table(
            data=df,
            artifact_file="interpretation/intrinsic.json",
            run_id=run_id,
        )


def intrinsic_catboost(cfg: DictConfig, train_val_sets: dict):
    feature_names = cfg.vars.categorical + cfg.vars.numerical

    for name, _ in tqdm(train_val_sets.items()):
        parent_run_id = mlflow_util.get_latest_run_id_by_name("CatBoost")
        run_id = mlflow_util.get_nested_run_ids_by_parent_id(parent_run_id, name=name)

        if not run_id:
            continue

        model = mlflow.catboost.load_model(f"runs:/{run_id}/model")

        # https://catboost.ai/en/docs/concepts/fstr#regular-feature-importance
        importances = model.get_feature_importance()

        df = pd.DataFrame({"feature_names": feature_names, "importances": importances})
        mlflow_util.log_table(
            data=df,
            artifact_file="interpretation/intrinsic.json",
            run_id=run_id,
        )


def pimp(train_val_sets: dict, model_name: str):
    for name, (_, _, X_val, y_val) in tqdm(train_val_sets.items()):
        parent_run_id = mlflow_util.get_latest_run_id_by_name(model_name)
        run_id = mlflow_util.get_nested_run_ids_by_parent_id(parent_run_id, name=name)
        model_uri = f"runs:/{run_id}/model"

        match model_name:
            case "XGBoost":
                model = mlflow.xgboost.load_model(model_uri)
            case "LightGBM":
                model = mlflow.lightgbm.load_model(model_uri)
            case "CatBoost":
                model = mlflow.catboost.load_model(model_uri)
            case _:
                model = mlflow.sklearn.load_model(model_uri)

        pimp = permutation_importance(
            estimator=model,
            X=X_val,
            y=y_val,
            n_repeats=10,
            n_jobs=-1,
            random_state=12345,
        )

        mlflow_util.log_pickle(pimp, artifact_file="pimp/pimp.pkl", run_id=run_id)


def pdp(cfg: DictConfig, train_val_sets: dict, model_name: str):
    feature_names = cfg.vars.categorical + cfg.vars.numerical

    for name, (_, _, X_val, _) in tqdm(train_val_sets.items()):
        parent_run_id = mlflow_util.get_latest_run_id_by_name(model_name)
        run_id = mlflow_util.get_nested_run_ids_by_parent_id(parent_run_id, name=name)
        model_uri = f"runs:/{run_id}/model"

        match model_name:
            case "XGBoost":
                model = mlflow.xgboost.load_model(model_uri)
            case "LightGBM":
                model = mlflow.lightgbm.load_model(model_uri)
            case "CatBoost":
                model = mlflow.catboost.load_model(model_uri)
            case _:
                model = mlflow.sklearn.load_model(model_uri)

        for idx in range(len(feature_names)):
            fig = plt.figure()
            ax = fig.gca()

            shap.plots.partial_dependence(
                ind=idx,
                model=model.predict,
                data=X_val,
                feature_names=feature_names,
                model_expected_value=True,
                feature_expected_value=True,
                ice=False,
                ax=ax,
                show=False,
            )

            mlflow_util.log_pickle(fig, artifact_file=f"pdp/{idx}.pkl", run_id=run_id)
            plt.close(fig)
