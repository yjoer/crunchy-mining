from __future__ import annotations

import gc
import os
import platform
import time
import tracemalloc
import typing
from contextlib import contextmanager
from multiprocessing import Manager
from multiprocessing import Process
from typing import List

import altair as alt
import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

if typing.TYPE_CHECKING:
    pass


def set_low_priority(pid: int):
    process = psutil.Process(pid)

    if platform.system() == "Windows":
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    else:
        process.nice(10)


def _get_current_memory(process: psutil.Process, pid_monitor: int = None):
    memory = process.memory_info().rss

    for child_process in process.children(recursive=True):
        if child_process.pid == pid_monitor:
            continue

        memory += child_process.memory_info().rss

    return memory


def _monitor_memory_usage(pid: int, d: dict, resolution: float):
    process = psutil.Process(pid)

    while True:
        memory = _get_current_memory(process, os.getpid())

        if "peak" not in d:
            d["peak"] = memory

        if memory > d["peak"]:
            d["peak"] = memory

        time.sleep(resolution)


@contextmanager
def trace_memory(legacy=True, resolution=0.1):
    # Yield a reference to a dictionary when creating a new context. The
    # dictionary is empty initially until the execution within the context is
    # finished.
    stats = {}
    start = time.perf_counter_ns()

    if legacy:
        if not tracemalloc.is_tracing():
            tracemalloc.start()
    else:
        # GC out-of-scope variables before collecting the memory usage.
        gc.collect()

        process = psutil.Process(os.getpid())
        start_memory = _get_current_memory(process)

        manager = Manager()
        d = manager.dict()
        d["peak"] = 0

        p = Process(target=_monitor_memory_usage, args=(os.getpid(), d, resolution))
        p.start()

    try:
        yield stats
    finally:
        if legacy:
            current, peak = tracemalloc.get_traced_memory()
        else:
            # Use the current memory at this point as the fallback for tasks
            # ending too quickly before the monitoring process is started.
            # Hopefully, the memory is not garbage collected or deallocated yet.
            if d["peak"] == 0:
                end_memory = _get_current_memory(process, p.pid)
            else:
                end_memory = d["peak"]

            current, peak = end_memory - start_memory, end_memory - start_memory

        end = time.perf_counter_ns()

        if legacy:
            tracemalloc.stop()
        else:
            p.terminate()

        stats["duration"] = end - start
        stats["current"] = current
        stats["peak"] = peak


def evaluate_roc(estimator, X_test, y_test, fixed_fpr: float):
    y_prob = estimator.predict_proba(X_test)
    y_score = y_prob[:, 1]
    fprs, tprs, thresholds = roc_curve(y_test, y_score)

    # Find the index of the closest FPR to the fixed FPR.
    idx = np.where(fprs <= fixed_fpr)[-1][-1]
    tpr = tprs[idx]
    fpr = fprs[idx]
    threshold = thresholds[idx]

    return (
        y_prob,
        {
            "true_positive_rates": tprs.tolist(),
            "false_positive_rates": fprs.tolist(),
            "thresholds": thresholds.tolist(),
        },
        {
            "true_positive_rate": tpr,
            "false_positive_rate": fpr,
            "threshold": threshold,
        },
    )


def custom_predict(estimator=None, X_test=None, y_prob=None, threshold=0.5):
    # Allow reusing the probabilities from the preceding steps.
    if estimator is not None and X_test is not None:
        y_prob = estimator.predict_proba(X_test)

    # It is possible to have two true classes if the threshold is low, but we
    # want to predict the instance as positive if the probability of the
    # positive class is higher than the threshold.
    # predictions = np.argmax(y_prob >= threshold, axis=1)
    predictions = (y_prob[:, 1] >= threshold).astype(int)

    return predictions


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

    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

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
        "mcc": mcc,
    }


def evaluate_regression(y_true, y_pred):
    ape = np.abs((y_true - y_pred) / y_true)
    mape = np.mean(ape)
    mdape = np.median(ape)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mape": mape,
        "mdae": median_absolute_error(y_true, y_pred),
        "mdape": mdape,
        "mse": mse,
        "rmse": rmse,
        "r_squared": r2_score(y_true, y_pred),
    }


def summarize_classification(df: pd.DataFrame):
    def create_formatted_values(df: pd.DataFrame, prefix: str):
        v = df[f"{prefix}_mean"].astype(str) + " ± " + df[f"{prefix}_std"].astype(str)
        df["formatted_values"] = v

    def create_experiment_idx(df_sub: pd.DataFrame):
        df_sub["experiment_id_first"] = df_sub["experiment_id_first"].astype(int)
        df_sub.sort_values(by="experiment_id_first", inplace=True)
        df_sub["experiment_idx"] = np.arange(1, len(df_sub) + 1)

    df_f1_macro = df.loc[df.groupby("experiment_id_first")["f1_macro_mean"].idxmax()]
    create_formatted_values(df_f1_macro, prefix="f1_macro")
    create_experiment_idx(df_f1_macro)

    df_f1_macro_min = df.loc[df.groupby("experiment_id_first")["f1_macro_mean"].idxmin()]  # fmt: skip
    create_formatted_values(df_f1_macro_min, prefix="f1_macro")
    create_experiment_idx(df_f1_macro_min)

    df_auc = df.loc[df.groupby("experiment_id_first")["roc_auc_mean"].idxmax()]
    create_formatted_values(df_auc, prefix="roc_auc")
    create_experiment_idx(df_auc)

    df_auc_min = df.loc[df.groupby("experiment_id_first")["roc_auc_mean"].idxmin()]
    create_formatted_values(df_auc_min, prefix="roc_auc")
    create_experiment_idx(df_auc_min)

    outputs = {}

    outputs["f1_macro_chart"] = (
        alt.Chart(
            data=df_f1_macro,
            title=alt.TitleParams(
                text="Best Models by Macro F1 Score Across Experiments",
                anchor="start",
            ),
        )
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("f1_macro_mean:Q", title="Macro F1 Score"),
            color=alt.Color("parent_run_name_first:N", title="Models"),
            tooltip=[
                alt.Tooltip("experiment_idx", title="Experiments"),
                alt.Tooltip("parent_run_name_first", title="Models"),
                alt.Tooltip("formatted_values", title="Macro F1 Score"),
            ],
        )
    )

    outputs["f1_macro_chart"] += (
        alt.Chart(df_f1_macro)
        .mark_errorbar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments"),
            y=alt.Y("f1_macro_mean:Q", title="Macro F1 Score"),
            yError=alt.YError("f1_macro_std:Q"),
        )
    )

    outputs["f1_macro_min_chart"] = (
        alt.Chart(
            data=df_f1_macro_min,
            title=alt.TitleParams(
                text="Worst Models by Macro F1 Score Across Experiments",
                anchor="start",
            ),
        )
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("f1_macro_mean:Q", title="Macro F1 Score"),
            color=alt.Color("parent_run_name_first:N", title="Models"),
            tooltip=[
                alt.Tooltip("experiment_idx", title="Experiments"),
                alt.Tooltip("parent_run_name_first", title="Models"),
                alt.Tooltip("formatted_values", title="Macro F1 Score"),
            ],
        )
    )

    outputs["f1_macro_min_chart"] += (
        alt.Chart(df_f1_macro_min)
        .mark_errorbar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments"),
            y=alt.Y("f1_macro_mean:Q", title="Macro F1 Score"),
            yError=alt.YError("f1_macro_std:Q"),
        )
    )

    outputs["auc_chart"] = (
        alt.Chart(
            data=df_auc,
            title=alt.TitleParams(
                text="Best Models by AUC Score Across Experiments",
                anchor="start",
            ),
        )
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("roc_auc_mean:Q", title="AUC Score"),
            color=alt.Color("parent_run_name_first:N", title="Models"),
            tooltip=[
                alt.Tooltip("experiment_idx", title="Experiments"),
                alt.Tooltip("parent_run_name_first", title="Models"),
                alt.Tooltip("formatted_values", title="AUC Score"),
            ],
        )
    )

    outputs["auc_chart"] += (
        alt.Chart(df_auc)
        .mark_errorbar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments"),
            y=alt.Y("roc_auc_mean:Q", title="AUC Score"),
            yError=alt.YError("roc_auc_std:Q"),
        )
    )

    outputs["auc_min_chart"] = (
        alt.Chart(
            data=df_auc_min,
            title=alt.TitleParams(
                text="Worst Models by AUC Score Across Experiments",
                anchor="start",
            ),
        )
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("roc_auc_mean:Q", title="AUC Score"),
            color=alt.Color("parent_run_name_first:N", title="Models"),
            tooltip=[
                alt.Tooltip("experiment_idx", title="Experiments"),
                alt.Tooltip("parent_run_name_first", title="Models"),
                alt.Tooltip("formatted_values", title="AUC Score"),
            ],
        )
    )

    outputs["auc_min_chart"] += (
        alt.Chart(df_auc_min)
        .mark_errorbar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments"),
            y=alt.Y("roc_auc_mean:Q", title="AUC Score"),
            yError=alt.YError("roc_auc_std:Q"),
        )
    )

    outputs["f1_macro"] = df_f1_macro
    outputs["f1_macro_min"] = df_f1_macro_min
    outputs["auc"] = df_auc
    outputs["auc_min"] = df_auc_min

    return outputs


def summarize_regression(df: pd.DataFrame):
    def create_formatted_values(df: pd.DataFrame, prefix: str):
        v = df[f"{prefix}_mean"].astype(str) + " ± " + df[f"{prefix}_std"].astype(str)
        df["formatted_values"] = v

    def create_experiment_idx(df_sub: pd.DataFrame):
        df_sub["experiment_id_first"] = df_sub["experiment_id_first"].astype(int)
        df_sub.sort_values(by="experiment_id_first", inplace=True)
        df_sub["experiment_idx"] = np.arange(1, len(df_sub) + 1)

    df_mae = df.loc[df.groupby("experiment_id_first")["mae_mean"].idxmin()]
    create_formatted_values(df_mae, prefix="mae")
    create_experiment_idx(df_mae)

    df_mae_max = df.loc[df.groupby("experiment_id_first")["mae_mean"].idxmax()]
    df_mae_max["mae_std_capped"] = np.where(df_mae_max["mae_std"] >= df_mae_max["mae_mean"], 0.99 * df_mae_max["mae_mean"], df_mae_max["mae_std"])  # fmt: skip
    create_formatted_values(df_mae_max, prefix="mae")
    create_experiment_idx(df_mae_max)

    df_mape = df.loc[df.groupby("experiment_id_first")["mape_mean"].idxmin()]
    create_formatted_values(df_mape, prefix="mape")
    create_experiment_idx(df_mape)

    df_mape_max = df.loc[df.groupby("experiment_id_first")["mape_mean"].idxmax()]
    df_mape_max["mape_std_capped"] = np.where(df_mape_max["mape_std"] >= df_mape_max["mape_mean"], 0.99 * df_mape_max["mape_mean"], df_mape_max["mape_std"])  # fmt: skip
    create_formatted_values(df_mape_max, prefix="mape")
    create_experiment_idx(df_mape_max)

    df_rsq = df.loc[df.groupby("experiment_id_first")["r_squared_mean"].idxmax()]
    create_formatted_values(df_rsq, prefix="r_squared")
    create_experiment_idx(df_rsq)

    df_rsq_min = df.loc[df.groupby("experiment_id_first")["r_squared_mean"].idxmin()]
    df_rsq_min["r_squared_capped"] = np.where(df_rsq_min["r_squared_mean"] < -1, -1, df_rsq_min["r_squared_mean"])  # fmt: skip
    df_rsq_min["r_squared_std_capped"] = np.where(np.abs(df_rsq_min["r_squared_std"]) >= np.abs(df_rsq_min["r_squared_capped"]), 0.99 * np.abs(df_rsq_min["r_squared_capped"]), df_rsq_min["r_squared_std"])  # fmt: skip
    create_formatted_values(df_rsq_min, prefix="r_squared")
    create_experiment_idx(df_rsq_min)

    outputs = {}

    outputs["mae_chart"] = (
        alt.Chart(
            data=df_mae,
            title=alt.TitleParams(
                text="Best Models by MAE Across Experiments",
                anchor="start",
            ),
        )
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("mae_mean:Q", title="Mean Absolute Error"),
            color=alt.Color("parent_run_name_first:N", title="Models"),
            tooltip=[
                alt.Tooltip("experiment_idx", title="Experiments"),
                alt.Tooltip("parent_run_name_first", title="Models"),
                alt.Tooltip("formatted_values", title="Mean Absolute Error"),
            ],
        )
    )

    outputs["mae_chart"] += (
        alt.Chart(df_mae)
        .mark_errorbar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments"),
            y=alt.Y("mae_mean:Q", title="Mean Absolute Error"),
            yError=alt.YError("mae_std:Q"),
        )
    )

    outputs["mae_max_chart"] = (
        alt.Chart(
            data=df_mae_max,
            title=alt.TitleParams(
                text="Worst Models by MAE Across Experiments",
                anchor="start",
            ),
        )
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("mae_mean:Q", title="Mean Absolute Error").scale(type="log"),
            color=alt.Color("parent_run_name_first:N", title="Models"),
            tooltip=[
                alt.Tooltip("experiment_idx", title="Experiments"),
                alt.Tooltip("parent_run_name_first", title="Models"),
                alt.Tooltip("formatted_values", title="Mean Absolute Error"),
            ],
        )
    )

    outputs["mae_max_chart"] += (
        alt.Chart(df_mae_max)
        .mark_errorbar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments"),
            y=alt.Y("mae_mean:Q", title="Mean Absolute Error"),
            yError=alt.YError("mae_std_capped:Q"),
        )
    )

    outputs["mape_chart"] = (
        alt.Chart(
            data=df_mape,
            title=alt.TitleParams(
                text="Best Models by MAPE Across Experiments",
                anchor="start",
            ),
        )
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("mape_mean:Q", title="Mean Absolute Percentage Error"),
            color=alt.Color("parent_run_name_first:N", title="Models"),
            tooltip=[
                alt.Tooltip("experiment_idx", title="Experiments"),
                alt.Tooltip("parent_run_name_first", title="Models"),
                alt.Tooltip("formatted_values", title="Mean Absolute Percentage Error"),
            ],
        )
    )

    outputs["mape_chart"] += (
        alt.Chart(df_mape)
        .mark_errorbar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments"),
            y=alt.Y("mape_mean:Q", title="Mean Absolute Percentage Error"),
            yError=alt.YError("mape_std:Q"),
        )
    )

    outputs["mape_max_chart"] = (
        alt.Chart(
            data=df_mape_max,
            title=alt.TitleParams(
                text="Worst Models by MAPE Across Experiments",
                anchor="start",
            ),
        )
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y(
                shorthand="mape_mean:Q",
                title="Mean Absolute Percentage Error",
                scale=alt.Scale(type="log"),
            ),
            color=alt.Color("parent_run_name_first:N", title="Models"),
            tooltip=[
                alt.Tooltip("experiment_idx", title="Experiments"),
                alt.Tooltip("parent_run_name_first", title="Models"),
                alt.Tooltip("formatted_values", title="Mean Absolute Percentage Error"),
            ],
        )
    )

    outputs["mape_max_chart"] += (
        alt.Chart(df_mape_max)
        .mark_errorbar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments"),
            y=alt.Y("mape_mean:Q", title="Mean Absolute Percentage Error"),
            yError=alt.YError("mape_std_capped:Q"),
        )
    )

    outputs["rsq_chart"] = (
        alt.Chart(
            data=df_rsq,
            title=alt.TitleParams(
                text="Best Models by R-Squared Score Across Experiments",
                anchor="start",
            ),
        )
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("r_squared_mean:Q", title="R-Squared"),
            color=alt.Color("parent_run_name_first:N", title="Models"),
            tooltip=[
                alt.Tooltip("experiment_idx", title="Experiments"),
                alt.Tooltip("parent_run_name_first", title="Models"),
                alt.Tooltip("formatted_values", title="R-Squared Score"),
            ],
        )
    )

    outputs["rsq_chart"] += (
        alt.Chart(df_rsq)
        .mark_errorbar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments"),
            y=alt.Y("r_squared_mean:Q", title="R-Squared Score"),
            yError=alt.YError("r_squared_std:Q"),
        )
    )

    outputs["rsq_min_chart"] = (
        alt.Chart(
            data=df_rsq_min,
            title=alt.TitleParams(
                text="Worst Models by R-Squared Score Across Experiments",
                anchor="start",
            ),
        )
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("r_squared_capped:Q", title="R-Squared Score"),
            color=alt.Color("parent_run_name_first:N", title="Models"),
            tooltip=[
                alt.Tooltip("experiment_idx", title="Experiments"),
                alt.Tooltip("parent_run_name_first", title="Models"),
                alt.Tooltip("formatted_values", title="R-Squared Score"),
            ],
        )
    )

    outputs["rsq_min_chart"] += (
        alt.Chart(df_rsq_min)
        .mark_errorbar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments"),
            y=alt.Y("r_squared_capped:Q", title="R-Squared"),
            yError=alt.YError("r_squared_std_capped:Q"),
        )
    )

    outputs["mae"] = df_mae
    outputs["mae_max"] = df_mae_max
    outputs["mape"] = df_mape
    outputs["mape_max"] = df_mape_max
    outputs["rsq"] = df_rsq
    outputs["rsq_min"] = df_rsq_min

    return outputs


def summarize_resource(df: pd.DataFrame):
    def create_formatted_values(df: pd.DataFrame, prefix: str):
        v = df[f"{prefix}_mean"].astype(str) + " ± " + df[f"{prefix}_std"].astype(str)
        df["formatted_values"] = v

    def create_experiment_idx(df_sub: pd.DataFrame):
        df_sub["experiment_id_first"] = df_sub["experiment_id_first"].astype(int)
        df_sub.sort_values(by="experiment_id_first", inplace=True)
        df_sub["experiment_idx"] = np.arange(1, len(df_sub) + 1)

    df_fit_time = df.loc[df.groupby("experiment_id_first")["fit_time_mean"].idxmin()]
    df_fit_time["fit_time_mean"] = df_fit_time["fit_time_mean"] / 1_000_000
    df_fit_time["fit_time_std"] = df_fit_time["fit_time_std"] / 1_000_000
    create_formatted_values(df_fit_time, prefix="fit_time")
    create_experiment_idx(df_fit_time)

    df_fit_time_max = df.loc[df.groupby("experiment_id_first")["fit_time_mean"].idxmax()]  # fmt: skip
    df_fit_time_max["fit_time_mean"] = df_fit_time_max["fit_time_mean"] / 1_000_000
    df_fit_time_max["fit_time_std"] = df_fit_time_max["fit_time_std"] / 1_000_000
    create_formatted_values(df_fit_time_max, prefix="fit_time")
    create_experiment_idx(df_fit_time_max)

    df_score_time = df.loc[df.groupby("experiment_id_first")["score_time_mean"].idxmin()]  # fmt: skip
    df_score_time["score_time_mean"] = df_score_time["score_time_mean"] / 1_000_000
    df_score_time["score_time_std"] = df_score_time["score_time_std"] / 1_000_000
    create_formatted_values(df_score_time, prefix="score_time")
    create_experiment_idx(df_score_time)

    df_score_time_max = df.loc[df.groupby("experiment_id_first")["score_time_mean"].idxmax()]  # fmt: skip
    df_score_time_max["score_time_mean"] = df_score_time_max["score_time_mean"] / 1_000_000  # fmt: skip
    df_score_time_max["score_time_std"] = df_score_time_max["score_time_std"] / 1_000_000  # fmt: skip
    create_formatted_values(df_score_time_max, prefix="score_time")
    create_experiment_idx(df_score_time_max)

    df_fit_memory = df.loc[df.groupby("experiment_id_first")["fit_memory_peak_mean"].idxmin()]  # fmt: skip
    df_fit_memory["fit_memory_peak_mean"] = df_fit_memory["fit_memory_peak_mean"] / 1_000_000  # fmt: skip
    df_fit_memory["fit_memory_peak_std"] = df_fit_memory["fit_memory_peak_std"] / 1_000_000  # fmt: skip
    create_formatted_values(df_fit_memory, prefix="fit_memory_peak")
    create_experiment_idx(df_fit_memory)

    df_fit_memory_max = df.loc[df.groupby("experiment_id_first")["fit_memory_peak_mean"].idxmax()]  # fmt: skip
    df_fit_memory_max["fit_memory_peak_mean"] = df_fit_memory_max["fit_memory_peak_mean"] / 1_000_000  # fmt: skip
    df_fit_memory_max["fit_memory_peak_std"] = df_fit_memory_max["fit_memory_peak_std"] / 1_000_000  # fmt: skip
    create_formatted_values(df_fit_memory_max, prefix="fit_memory_peak")
    create_experiment_idx(df_fit_memory_max)

    df_score_memory = df.loc[df.groupby("experiment_id_first")["score_memory_peak_mean"].idxmin()]  # fmt: skip
    df_score_memory["score_memory_peak_mean"] = df_score_memory["score_memory_peak_mean"] / 1_000_000  # fmt: skip
    df_score_memory["score_memory_peak_std"] = df_score_memory["score_memory_peak_std"] / 1_000_000  # fmt: skip
    create_formatted_values(df_score_memory, prefix="score_memory_peak")
    create_experiment_idx(df_score_memory)

    df_score_memory_max = df.loc[df.groupby("experiment_id_first")["score_memory_peak_mean"].idxmax()]  # fmt: skip
    df_score_memory_max["score_memory_peak_mean"] = df_score_memory_max["score_memory_peak_mean"] / 1_000_000  # fmt: skip
    df_score_memory_max["score_memory_peak_std"] = df_score_memory_max["score_memory_peak_std"] / 1_000_000  # fmt: skip
    create_formatted_values(df_score_memory_max, prefix="score_memory_peak")
    create_experiment_idx(df_score_memory_max)

    outputs = {}

    outputs["fit_time_chart"] = (
        alt.Chart(
            data=df_fit_time,
            title=alt.TitleParams(
                text="Best Models by Fit Time Across Experiments",
                anchor="start",
            ),
        )
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y(
                shorthand="fit_time_mean:Q",
                title="Fit Time (ms)",
                scale=alt.Scale(type="log"),
            ),
            color=alt.Color("parent_run_name_first:N", title="Models"),
            tooltip=[
                alt.Tooltip("experiment_idx", title="Experiments"),
                alt.Tooltip("parent_run_name_first", title="Models"),
                alt.Tooltip("formatted_values", title="Fit Time (ms)"),
            ],
        )
    )

    outputs["fit_time_chart"] += (
        alt.Chart(df_fit_time)
        .mark_errorbar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments"),
            y=alt.Y("fit_time_mean:Q", title="Fit Time (ms)"),
            yError=alt.YError("fit_time_std:Q"),
        )
    )

    outputs["fit_time_max_chart"] = (
        alt.Chart(
            data=df_fit_time_max,
            title=alt.TitleParams(
                text="Worst Models by Fit Time Across Experiments",
                anchor="start",
            ),
        )
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("fit_time_mean:Q", title="Fit Time (ms)"),
            color=alt.Color("parent_run_name_first:N", title="Models"),
            tooltip=[
                alt.Tooltip("experiment_idx", title="Experiments"),
                alt.Tooltip("parent_run_name_first", title="Models"),
                alt.Tooltip("formatted_values", title="Fit Time (ms)"),
            ],
        )
    )

    outputs["fit_time_max_chart"] += (
        alt.Chart(df_fit_time_max)
        .mark_errorbar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments"),
            y=alt.Y("fit_time_mean:Q", title="Fit Time (ms)"),
            yError=alt.YError("fit_time_std:Q"),
        )
    )

    outputs["score_time_chart"] = (
        alt.Chart(
            data=df_score_time,
            title=alt.TitleParams(
                text="Best Models by Score Time Across Experiments",
                anchor="start",
            ),
        )
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("score_time_mean:Q", title="Score Time (ms)"),
            color=alt.Color("parent_run_name_first:N", title="Models"),
            tooltip=[
                alt.Tooltip("experiment_idx", title="Experiments"),
                alt.Tooltip("parent_run_name_first", title="Models"),
                alt.Tooltip("formatted_values", title="Score Time (ms)"),
            ],
        )
    )

    outputs["score_time_chart"] += (
        alt.Chart(df_score_time)
        .mark_errorbar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments"),
            y=alt.Y("score_time_mean:Q", title="Score Time (ms)"),
            yError=alt.YError("score_time_std:Q"),
        )
    )

    outputs["score_time_max_chart"] = (
        alt.Chart(
            data=df_score_time_max,
            title=alt.TitleParams(
                text="Worst Models by Score Time Across Experiments",
                anchor="start",
            ),
        )
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("score_time_mean:Q", title="Score Time (ms)"),
            color=alt.Color("parent_run_name_first:N", title="Models"),
            tooltip=[
                alt.Tooltip("experiment_idx", title="Experiments"),
                alt.Tooltip("parent_run_name_first", title="Models"),
                alt.Tooltip("formatted_values", title="Score Time (ms)"),
            ],
        )
    )

    outputs["score_time_max_chart"] += (
        alt.Chart(df_score_time_max)
        .mark_errorbar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments"),
            y=alt.Y("score_time_mean:Q", title="Score Time (ms)"),
            yError=alt.YError("score_time_std:Q"),
        )
    )

    outputs["fit_memory_chart"] = (
        alt.Chart(
            data=df_fit_memory,
            title=alt.TitleParams(
                text="Best Models by Fit Memory Across Experiments",
                anchor="start",
            ),
        )
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("fit_memory_peak_mean:Q", title="Fit Memory (MB)"),
            color=alt.Color("parent_run_name_first:N", title="Models"),
            tooltip=[
                alt.Tooltip("experiment_idx", title="Experiments"),
                alt.Tooltip("parent_run_name_first", title="Models"),
                alt.Tooltip("formatted_values", title="Fit Memory (MB)"),
            ],
        )
    )

    outputs["fit_memory_chart"] += (
        alt.Chart(df_fit_memory)
        .mark_errorbar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments"),
            y=alt.Y("fit_memory_peak_mean:Q", title="Fit Memory (MB)"),
            yError=alt.YError("fit_memory_peak_std:Q"),
        )
    )

    outputs["fit_memory_max_chart"] = (
        alt.Chart(
            data=df_fit_memory_max,
            title=alt.TitleParams(
                text="Worst Models by Fit Memory Across Experiments",
                anchor="start",
            ),
        )
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("fit_memory_peak_mean:Q", title="Fit Memory (MB)"),
            color=alt.Color("parent_run_name_first:N", title="Models"),
            tooltip=[
                alt.Tooltip("experiment_idx", title="Experiments"),
                alt.Tooltip("parent_run_name_first", title="Models"),
                alt.Tooltip("formatted_values", title="Fit Memory (MB)"),
            ],
        )
    )

    outputs["fit_memory_max_chart"] += (
        alt.Chart(df_fit_memory_max)
        .mark_errorbar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments"),
            y=alt.Y("fit_memory_peak_mean:Q", title="Fit Memory (MB)"),
            yError=alt.YError("fit_memory_peak_std:Q"),
        )
    )

    outputs["score_memory_chart"] = (
        alt.Chart(
            data=df_score_memory,
            title=alt.TitleParams(
                text="Best Models by Score Memory Across Experiments",
                anchor="start",
            ),
        )
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("score_memory_peak_mean:Q", title="Score Memory (MB)"),
            color=alt.Color("parent_run_name_first:N", title="Models"),
            tooltip=[
                alt.Tooltip("experiment_idx", title="Experiments"),
                alt.Tooltip("parent_run_name_first", title="Models"),
                alt.Tooltip("formatted_values", title="Score Memory (MB)"),
            ],
        )
    )

    outputs["score_memory_chart"] += (
        alt.Chart(df_score_memory)
        .mark_errorbar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments"),
            y=alt.Y("score_memory_peak_mean:Q", title="Score Memory (MB)"),
            yError=alt.YError("score_memory_peak_std:Q"),
        )
    )

    outputs["score_memory_max_chart"] = (
        alt.Chart(
            data=df_score_memory_max,
            title=alt.TitleParams(
                text="Worst Models by Score Memory Across Experiments",
                anchor="start",
            ),
        )
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("score_memory_peak_mean:Q", title="Score Memory (MB)"),
            color=alt.Color("parent_run_name_first:N", title="Models"),
            tooltip=[
                alt.Tooltip("experiment_idx", title="Experiments"),
                alt.Tooltip("parent_run_name_first", title="Models"),
                alt.Tooltip("formatted_values", title="Score Memory (MB)"),
            ],
        )
    )

    outputs["score_memory_max_chart"] += (
        alt.Chart(df_score_memory_max)
        .mark_errorbar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments"),
            y=alt.Y("score_memory_peak_mean:Q", title="Score Memory (MB)"),
            yError=alt.YError("score_memory_peak_std:Q"),
        )
    )

    outputs["fit_time"] = df_fit_time
    outputs["fit_time_max"] = df_fit_time_max
    outputs["score_time"] = df_score_time
    outputs["score_time_max"] = df_score_time_max
    outputs["fit_memory"] = df_fit_memory
    outputs["fit_memory_max"] = df_fit_memory_max
    outputs["score_memory"] = df_score_memory
    outputs["score_memory_max"] = df_score_memory_max

    return outputs


def tabulate_classification_report(metrics: dict):
    n_support = int(metrics["support_0"] + metrics["support_1"])

    def fmt(x):
        return f"{np.round(x, 4):.4f}"

    df = pd.DataFrame(
        data={
            "precision": [
                fmt(metrics["precision_0"]),
                fmt(metrics["precision_1"]),
                "",
                "",
                "",
                fmt(metrics["precision_macro"]),
                fmt(metrics["precision_weighted"]),
            ],
            "recall": [
                fmt(metrics["recall_0"]),
                fmt(metrics["recall_1"]),
                "",
                "",
                "",
                fmt(metrics["recall_macro"]),
                fmt(metrics["recall_weighted"]),
            ],
            "f1-score": [
                fmt(metrics["f1_0"]),
                fmt(metrics["f1_1"]),
                "",
                fmt(metrics["accuracy"]),
                fmt(metrics["roc_auc"]),
                fmt(metrics["f1_macro"]),
                fmt(metrics["f1_weighted"]),
            ],
            "support": [
                int(metrics["support_0"]),
                int(metrics["support_1"]),
                "",
                n_support,
                n_support,
                n_support,
                n_support,
            ],
        },
        index=["0", "1", "", "accuracy", "auc", "macro avg", "weighted avg"],
    )

    return df


def tabulate_resource_usage(metrics: dict):
    def fmt_time(x):
        return f"{np.round(x / 1_000_000, 4) :.4f} ms"

    def fmt_memory(x):
        return f"{np.round(x / 1_000_000, 4):.4f} MB"

    df = pd.Series(
        {
            "Fit Time": fmt_time(metrics["fit_time"]),
            "Fit Memory (Peak)": fmt_memory(metrics["fit_memory_peak"]),
            "Score Time": fmt_time(metrics["score_time"]),
            "Score Memory (Peak)": fmt_memory(metrics["score_memory_peak"]),
        },
        name="Measurements",
    )

    return df


def plot_confusion_matrix(metrics: dict):
    df = pd.DataFrame(
        {
            "predicted_label": [0, 0, 1, 1],
            "true_label": [0, 1, 0, 1],
            "v": [
                metrics["true_negative"],
                metrics["false_negative"],
                metrics["false_positive"],
                metrics["true_positive"],
            ],
        }
    )

    p75 = df["v"].quantile(q=0.75)

    base = alt.Chart(df, title="Confusion Matrix").encode(
        x=alt.X(
            shorthand="predicted_label:O",
            title="Predicted Label",
            axis=alt.Axis(
                ticks=True,
                labelAlign="center",
                labelAngle=0,
                labelFontSize=15,
                labelPadding=4,
            ),
        ),
        y=alt.Y(
            shorthand="true_label:O",
            title="True Label",
            axis=alt.Axis(
                ticks=True,
                labelBaseline="middle",
                labelFontSize=15,
                labelPadding=4,
            ),
        ),
    )

    heatmap = (
        base.mark_rect()
        .encode(
            color=alt.Color(
                shorthand="v:Q",
                title="Value",
                legend=alt.Legend(title=None),
            ),
        )
        .properties(height=320)
    )

    text = base.mark_text(baseline="middle", fontSize=14).encode(
        text=alt.Text("v:Q"),
        color=alt.condition(alt.datum.v > p75, alt.value("white"), alt.value("black")),
    )

    return heatmap + text


def plot_evaluation_stability(df: pd.DataFrame):
    return (
        alt.Chart(df, title="Stability of Evaluation Scores Across Folds")
        .mark_bar(width={"band": 1})
        .encode(
            x=alt.X(
                shorthand="metrics:N",
                title="Metrics",
                axis=alt.Axis(labelAngle=0, labelPadding=8),
            ),
            y=alt.Y("value:Q", title="Scores", axis=alt.Axis(labelPadding=8)),
            xOffset=alt.XOffset("folds:N", title="Folds"),
            color=alt.Color("folds:N", title="Folds"),
        )
    )


def plot_resource_stability(df_time: pd.DataFrame, df_memory: pd.DataFrame):
    chart_t = (
        alt.Chart(df_time, title="Stability of Time Taken Across Folds")
        .mark_bar(width={"band": 1})
        .encode(
            x=alt.X(
                shorthand="metrics:N",
                title="Metrics",
                axis=alt.Axis(labelAngle=0, labelPadding=8),
            ),
            y=alt.Y(
                shorthand="value:Q",
                title="Time Taken (ms)",
                axis=alt.Axis(labelPadding=8),
                scale=alt.Scale(type="log"),
            ),
            xOffset=alt.XOffset("folds:N", title="Folds"),
            color=alt.Color("folds:N", title="Folds"),
        )
    )

    chart_m = (
        alt.Chart(df_memory, title="Stability of Memory Usage Across Folds")
        .mark_bar(width={"band": 1})
        .encode(
            x=alt.X(
                shorthand="metrics:N",
                title="Metrics",
                axis=alt.Axis(labelAngle=0, labelPadding=8),
            ),
            y=alt.Y(
                shorthand="value:Q",
                title="Memory Usage (MB)",
                axis=alt.Axis(labelPadding=8),
            ),
            xOffset=alt.XOffset("folds:N", title="Folds"),
            color=alt.Color("folds:N", title="Folds"),
        )
    )

    return chart_t, chart_m


def plot_intrinsic_importances(importances: pd.DataFrame, name: str):
    return (
        alt.Chart(importances, title=f"Feature Importance for {name}")
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
