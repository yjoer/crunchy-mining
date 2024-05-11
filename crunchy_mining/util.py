from __future__ import annotations

import time
import tracemalloc
import typing
from contextlib import contextmanager
from typing import List

import altair as alt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score

if typing.TYPE_CHECKING:
    pass


@contextmanager
def trace_memory():
    # Yield a reference to a dictionary when creating a new context. The
    # dictionary is empty initially until the execution within the context is
    # finished.
    stats = {}
    start = time.perf_counter_ns()

    if not tracemalloc.is_tracing():
        tracemalloc.start()

    try:
        yield stats
    finally:
        current, peak = tracemalloc.get_traced_memory()
        end = time.perf_counter_ns()
        tracemalloc.stop()

        stats["duration"] = end - start
        stats["current"] = current
        stats["peak"] = peak


def evaluate_classification(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    p_pos = tp / (tp + fp)
    r_pos = tp / (tp + fn)
    f1_pos = (2 * p_pos * r_pos) / (p_pos + r_pos)

    p_neg = tn / (tn + fn)
    r_neg = tn / (tn + fp)
    f1_neg = (2 * p_neg * r_neg) / (p_neg + r_neg)

    p_macro = (p_pos + p_neg) / 2
    r_macro = (r_pos + r_neg) / 2
    f1_macro = (f1_pos + f1_neg) / 2

    sup_pos = tp + fn
    sup_neg = tn + fp

    p_weighted = (p_pos * sup_pos + p_neg * sup_neg) / (sup_pos + sup_neg)
    r_weighted = (r_pos * sup_pos + r_neg * sup_neg) / (sup_pos + sup_neg)
    f1_weighted = (f1_pos * sup_pos + f1_neg * sup_neg) / (sup_pos + sup_neg)

    return {
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "precision_1": p_pos,
        "recall_1": r_pos,
        "f1_1": f1_pos,
        "precision_0": p_neg,
        "recall_0": r_neg,
        "f1_0": f1_neg,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f1_macro,
        "precision_weighted": p_weighted,
        "recall_weighted": r_weighted,
        "f1_weighted": f1_weighted,
        "support_1": sup_pos,
        "support_0": sup_neg,
        "roc_auc": roc_auc_score(y_true, y_pred),
    }


def plot_weights_logistic_regression(importances: pd.DataFrame):
    return (
        alt.Chart(importances, title="Feature Importance for Logistic Regression")
        .mark_bar()
        .encode(
            x=alt.X("importances:Q", title="Degree of Importance"),
            y=alt.Y("feature_names:N", title="Feature Names").sort("-x"),
            color=alt.Color("feature_names:N", legend=None, sort="-x"),
            tooltip="importances:Q",
        )
    )


def plot_weights_linear_svc(importances: pd.DataFrame):
    return (
        alt.Chart(importances, title="Feature Importance for Linear SVC")
        .mark_bar()
        .encode(
            x=alt.X("importances:Q", title="Degree of Importance"),
            y=alt.Y("feature_names:N", title="Feature Names").sort("-x"),
            color=alt.Color("feature_names:N", legend=None, sort="-x"),
            tooltip="importances:Q",
        )
    )


def plot_impurity_decision_tree(importances: pd.DataFrame):
    return (
        alt.Chart(importances, title="Feature Importance for Decision Tree")
        .mark_bar()
        .encode(
            x=alt.X("importances:Q", title="Degree of Importance"),
            y=alt.Y("feature_names:N", title="Feature Names").sort("-x"),
            color=alt.Color("feature_names:N", legend=None, sort="-x"),
            tooltip="importances:Q",
        )
    )


def plot_impurity_random_forest(importances: pd.DataFrame):
    return (
        alt.Chart(importances, title="Feature Importance for Random Forest")
        .mark_bar()
        .encode(
            x=alt.X("importances:Q", title="Degree of Importance"),
            y=alt.Y("feature_names:N", title="Feature Names").sort("-x"),
            color=alt.Color("feature_names:N", legend=None, sort="-x"),
            tooltip="importances:Q",
        )
    )


def plot_impurity_adaboost(importances: pd.DataFrame):
    return (
        alt.Chart(importances, title="Feature Importance for AdaBoost")
        .mark_bar()
        .encode(
            x=alt.X("importances:Q", title="Degree of Importance"),
            y=alt.Y("feature_names:N", title="Feature Names").sort("-x"),
            color=alt.Color("feature_names:N", legend=None, sort="-x"),
            tooltip="importances:Q",
        )
    )


def plot_gain_xgboost(importances: pd.DataFrame):
    return (
        alt.Chart(importances, title="Feature Importance for XGBoost")
        .mark_bar()
        .encode(
            x=alt.X("importances:Q", title="Degree of Importance"),
            y=alt.Y("feature_names:N", title="Feature Names").sort("-x"),
            color=alt.Color("feature_names:N", legend=None, sort="-x"),
            tooltip="importances:Q",
        )
    )


def plot_gain_lightgbm(importances: pd.DataFrame):
    return (
        alt.Chart(importances, title="Feature Importance for LightGBM")
        .mark_bar()
        .encode(
            x=alt.X("importances:Q", title="Degree of Importance"),
            y=alt.Y("feature_names:N", title="Feature Names").sort("-x"),
            color=alt.Color("feature_names:N", legend=None, sort="-x"),
            tooltip="importances:Q",
        )
    )


def plot_pvc_catboost(importances: pd.DataFrame):
    return (
        alt.Chart(importances, title="Feature Importance for CatBoost")
        .mark_bar()
        .encode(
            x=alt.X("importances:Q", title="Degree of Importance"),
            y=alt.Y("feature_names:N", title="Feature Names").sort("-x"),
            color=alt.Color("feature_names:N", legend=None, sort="-x"),
            tooltip="importances:Q",
        )
    )


def plot_pimp_mean(feature_names: List[str], pimp):
    df = pd.DataFrame(
        {
            "features": feature_names,
            "importance": pimp["importances_mean"],
        }
    )

    return df, (
        alt.Chart(df, title="Permutation Feature Importance (Mean)")
        .mark_bar()
        .encode(
            x=alt.X("importance:Q", title="Degree of Importance"),
            y=alt.Y("features:N", title="Feature Names").sort("-x"),
            color=alt.Color("features:N", legend=None, sort="-x"),
            tooltip="importance:Q",
        )
    )


def plot_pimp_boxplot(feature_names: List[str], pimp):
    df = pd.concat(
        objs=(
            pd.DataFrame({"features": feature_names}),
            pd.DataFrame(pimp["importances"]),
        ),
        axis=1,
    ).melt(
        id_vars=["features"],
        var_name="repeat_idx",
        value_name="importance",
    )

    return (
        alt.Chart(df, title="Permutation Feature Importance in Repeated Runs")
        .mark_boxplot()
        .encode(
            x=alt.X("importance:Q", title="Degree of Importance"),
            y=alt.Y("features:N", title="Feature Names"),
            color=alt.Color("features:N", legend=None),
        )
    )


def calculate_mape(y_true, y_pred):
    actual = np.array(y_true)
    predicted = np.array(y_pred.flatten())
    return np.mean(np.abs((actual - predicted) / actual)) * 100


def evaluate_regression(y_test, y_pred):
    R2 = r2_score(y_test, y_pred)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    MSE = metrics.mean_squared_error(y_test, y_pred)
    MAE = metrics.mean_absolute_error(y_test, y_pred)

    return {
        "R2": round(R2, 4),
        "RMSE": round(RMSE, 4),
        "MSE": round(MSE, 4),
        "MAE": round(MAE, 4),
    }
