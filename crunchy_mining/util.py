from __future__ import annotations

import tracemalloc
import typing
from contextlib import contextmanager
from typing import List

import altair as alt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

if typing.TYPE_CHECKING:
    from catboost import CatBoostClassifier
    from catboost import CatBoostRegressor
    from lightgbm import LGBMClassifier
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.tree import DecisionTreeRegressor
    from xgboost import XGBClassifier
    from xgboost import XGBRegressor


@contextmanager
def trace_memory():
    # Yield a reference to a dictionary when creating a new context. The
    # dictionary is empty initially until the execution within the context is
    # finished.
    stats = {}

    if not tracemalloc.is_tracing():
        tracemalloc.start()

    try:
        yield stats
    finally:
        current, peak = tracemalloc.get_traced_memory()
        stats["current"] = current
        stats["peak"] = peak

        tracemalloc.stop()


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


def interpret_weights_logistic_regression(model: LogisticRegression, X_train):
    return np.abs(np.std(X_train, axis=0) * model.coef_[0])


def plot_weights_logistic_regression(feature_names: List[str], importance):
    df = pd.DataFrame({"features": feature_names, "importance": importance})

    return (
        alt.Chart(df, title="Feature Importance for Logistic Regression")
        .mark_bar()
        .encode(
            x=alt.X("importance:Q", title="Degree of Importance"),
            y=alt.Y("features:N", title="Feature Names").sort("-x"),
            color=alt.Color("features:N", legend=None, sort="-x"),
            tooltip="importance:Q",
        )
    )


def interpret_weights_linear_svc(model: LinearSVC, X_train):
    return np.abs(np.std(X_train, axis=0) * model.coef_[0])


def plot_weights_linear_svc(feature_names: List[str], importance):
    df = pd.DataFrame({"features": feature_names, "importance": importance})

    return (
        alt.Chart(df, title="Feature Importance for Linear SVC")
        .mark_bar()
        .encode(
            x=alt.X("importance:Q", title="Degree of Importance"),
            y=alt.Y("features:N", title="Feature Names").sort("-x"),
            color=alt.Color("features:N", legend=None, sort="-x"),
            tooltip="importance:Q",
        )
    )


def interpret_impurity_decision_tree(
    model: DecisionTreeClassifier | DecisionTreeRegressor,
):
    return model.feature_importances_


def plot_impurity_decision_tree(feature_names: List[str], importance):
    df = pd.DataFrame({"features": feature_names, "importance": importance})

    return (
        alt.Chart(df, title="Feature Importance for Decision Tree")
        .mark_bar()
        .encode(
            x=alt.X("importance:Q", title="Degree of Importance"),
            y=alt.Y("features:N", title="Feature Names").sort("-x"),
            color=alt.Color("features:N", legend=None, sort="-x"),
            tooltip="importance:Q",
        )
    )


def interpret_impurity_random_forest(
    model: RandomForestClassifier | RandomForestRegressor,
):
    return model.feature_importances_


def plot_impurity_random_forest(feature_names: List[str], importance):
    df = pd.DataFrame({"features": feature_names, "importance": importance})

    return (
        alt.Chart(df, title="Feature Importance for Random Forest")
        .mark_bar()
        .encode(
            x=alt.X("importance:Q", title="Degree of Importance"),
            y=alt.Y("features:N", title="Feature Names").sort("-x"),
            color=alt.Color("features:N", legend=None, sort="-x"),
            tooltip="importance:Q",
        )
    )


def interpret_impurity_adaboost(model: AdaBoostClassifier | AdaBoostRegressor):
    return model.feature_importances_


def plot_impurity_adaboost(feature_names: List[str], importance):
    df = pd.DataFrame({"features": feature_names, "importance": importance})

    return (
        alt.Chart(df, title="Feature Importance for AdaBoost")
        .mark_bar()
        .encode(
            x=alt.X("importance:Q", title="Degree of Importance"),
            y=alt.Y("features:N", title="Feature Names").sort("-x"),
            color=alt.Color("features:N", legend=None, sort="-x"),
            tooltip="importance:Q",
        )
    )


def interpret_gain_xgboost(model: XGBClassifier | XGBRegressor):
    return model.feature_importances_


def plot_gain_xgboost(feature_names: List[str], importance):
    df = pd.DataFrame({"features": feature_names, "importance": importance})

    return (
        alt.Chart(df, title="Feature Importance for XGBoost")
        .mark_bar()
        .encode(
            x=alt.X("importance:Q", title="Degree of Importance"),
            y=alt.Y("features:N", title="Feature Names").sort("-x"),
            color=alt.Color("features:N", legend=None, sort="-x"),
            tooltip="importance:Q",
        )
    )


def interpret_gain_lightgbm(model: LGBMClassifier | LGBMRegressor):
    # Use gain as the importance type and normalize to align with XGBoost.
    gains = model.booster_.feature_importance(importance_type="gain")
    return gains / np.sum(gains)


def plot_gain_lightgbm(feature_names: List[str], importance):
    df = pd.DataFrame({"features": feature_names, "importance": importance})

    return (
        alt.Chart(df, title="Feature Importance for LightGBM")
        .mark_bar()
        .encode(
            x=alt.X("importance:Q", title="Degree of Importance"),
            y=alt.Y("features:N", title="Feature Names").sort("-x"),
            color=alt.Color("features:N", legend=None, sort="-x"),
            tooltip="importance:Q",
        )
    )


def interpret_pvc_catboost(model: CatBoostClassifier | CatBoostRegressor):
    # https://catboost.ai/en/docs/concepts/fstr#regular-feature-importance
    return model.get_feature_importance()


def plot_pvc_catboost(feature_names: List[str], importance):
    df = pd.DataFrame({"features": feature_names, "importance": importance})

    return (
        alt.Chart(df, title="Feature Importance for CatBoost")
        .mark_bar()
        .encode(
            x=alt.X("importance:Q", title="Degree of Importance"),
            y=alt.Y("features:N", title="Feature Names").sort("-x"),
            color=alt.Color("features:N", legend=None, sort="-x"),
            tooltip="importance:Q",
        )
    )
