from __future__ import annotations

import platform
import time
import tracemalloc
import typing
from contextlib import contextmanager
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

if typing.TYPE_CHECKING:
    pass


def set_low_priority(pid: int):
    process = psutil.Process(pid)

    if platform.system() == "Windows":
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    else:
        process.nice(10)


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
    df_f1_macro = df.loc[df.groupby("experiment_id")["f1_macro"].idxmax()]
    df_f1_macro["experiment_id"] = df_f1_macro["experiment_id"].astype(int)
    df_f1_macro.sort_values(by="experiment_id", inplace=True)
    df_f1_macro["experiment_idx"] = np.arange(1, len(df_f1_macro) + 1)

    df_f1_macro_min = df.loc[df.groupby("experiment_id")["f1_macro"].idxmin()]
    df_f1_macro_min["experiment_id"] = df_f1_macro_min["experiment_id"].astype(int)
    df_f1_macro_min.sort_values(by="experiment_id", inplace=True)
    df_f1_macro_min["experiment_idx"] = np.arange(1, len(df_f1_macro_min) + 1)

    df_auc = df.loc[df.groupby("experiment_id")["roc_auc"].idxmax()]
    df_auc["experiment_id"] = df_auc["experiment_id"].astype(int)
    df_auc.sort_values(by="experiment_id", inplace=True)
    df_auc["experiment_idx"] = np.arange(1, len(df_auc) + 1)

    df_auc_min = df.loc[df.groupby("experiment_id")["roc_auc"].idxmin()]
    df_auc_min["experiment_id"] = df_auc_min["experiment_id"].astype(int)
    df_auc_min.sort_values(by="experiment_id", inplace=True)
    df_auc_min["experiment_idx"] = np.arange(1, len(df_auc_min) + 1)

    df_fit_time = df.loc[df.groupby("experiment_id")["fit_time"].idxmin()]
    df_fit_time["experiment_id"] = df_fit_time["experiment_id"].astype(int)
    df_fit_time.sort_values(by="experiment_id", inplace=True)
    df_fit_time["experiment_idx"] = np.arange(1, len(df_fit_time) + 1)
    df_fit_time["fit_time"] = df_fit_time["fit_time"] / 1_000_000

    df_fit_time_max = df.loc[df.groupby("experiment_id")["fit_time"].idxmax()]
    df_fit_time_max["experiment_id"] = df_fit_time_max["experiment_id"].astype(int)
    df_fit_time_max.sort_values(by="experiment_id", inplace=True)
    df_fit_time_max["experiment_idx"] = np.arange(1, len(df_fit_time_max) + 1)
    df_fit_time_max["fit_time"] = df_fit_time_max["fit_time"] / 1_000_000

    df_score_time = df.loc[df.groupby("experiment_id")["score_time"].idxmin()]
    df_score_time["experiment_id"] = df_score_time["experiment_id"].astype(int)
    df_score_time.sort_values(by="experiment_id", inplace=True)
    df_score_time["experiment_idx"] = np.arange(1, len(df_score_time) + 1)
    df_score_time["score_time"] = df_score_time["score_time"] / 1_000_000

    df_score_time_max = df.loc[df.groupby("experiment_id")["score_time"].idxmax()]
    df_score_time_max["experiment_id"] = df_score_time_max["experiment_id"].astype(int)
    df_score_time_max.sort_values(by="experiment_id", inplace=True)
    df_score_time_max["experiment_idx"] = np.arange(1, len(df_score_time_max) + 1)
    df_score_time_max["score_time"] = df_score_time_max["score_time"] / 1_000_000

    df_fit_memory = df.loc[df.groupby("experiment_id")["fit_memory_peak"].idxmin()]
    df_fit_memory["experiment_id"] = df_fit_memory["experiment_id"].astype(int)
    df_fit_memory.sort_values(by="experiment_id", inplace=True)
    df_fit_memory["experiment_idx"] = np.arange(1, len(df_fit_memory) + 1)
    df_fit_memory["fit_memory_peak"] = df_fit_memory["fit_memory_peak"] / 1_000_000

    df_fit_memory_max = df.loc[df.groupby("experiment_id")["fit_memory_peak"].idxmax()]
    df_fit_memory_max["experiment_id"] = df_fit_memory_max["experiment_id"].astype(int)
    df_fit_memory_max.sort_values(by="experiment_id", inplace=True)
    df_fit_memory_max["experiment_idx"] = np.arange(1, len(df_fit_memory_max) + 1)
    df_fit_memory_max["fit_memory_peak"] = df_fit_memory_max["fit_memory_peak"] / 1_000_000  # fmt: skip

    df_score_memory = df.loc[df.groupby("experiment_id")["score_memory_peak"].idxmin()]
    df_score_memory["experiment_id"] = df_score_memory["experiment_id"].astype(int)
    df_score_memory.sort_values(by="experiment_id", inplace=True)
    df_score_memory["experiment_idx"] = np.arange(1, len(df_score_memory) + 1)
    df_score_memory["score_memory_peak"] = df_score_memory["score_memory_peak"] / 1_000_000  # fmt: skip

    df_score_memory_max = df.loc[df.groupby("experiment_id")["score_memory_peak"].idxmax()]  # fmt: skip
    df_score_memory_max["experiment_id"] = df_score_memory_max["experiment_id"].astype(int)  # fmt: skip
    df_score_memory_max.sort_values(by="experiment_id", inplace=True)
    df_score_memory_max["experiment_idx"] = np.arange(1, len(df_score_memory_max) + 1)
    df_score_memory_max["score_memory_peak"] = df_score_memory_max["score_memory_peak"] / 1_000_000  # fmt: skip

    outputs = {}

    outputs["f1_macro_chart"] = (
        alt.Chart(df_f1_macro, title="Best Models by Macro F1 Score Across Experiments")
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("f1_macro:Q", title="Macro F1 Score"),
            color=alt.Color("parent_run_name:N", title="Models"),
        )
    )

    outputs["f1_macro_min_chart"] = (
        alt.Chart(
            data=df_f1_macro_min,
            title="Worst Models by Macro F1 Score Across Experiments",
        )
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("f1_macro:Q", title="Macro F1 Score"),
            color=alt.Color("parent_run_name:N", title="Models"),
        )
    )

    outputs["auc_chart"] = (
        alt.Chart(df_auc, title="Best Models by AUC Score Across Experiments")
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("roc_auc:Q", title="ROC AUC"),
            color=alt.Color("parent_run_name:N", title="Models"),
        )
    )

    outputs["auc_min_chart"] = (
        alt.Chart(df_auc_min, title="Worst Models by AUC Score Across Experiments")
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("roc_auc:Q", title="ROC AUC"),
            color=alt.Color("parent_run_name:N", title="Models"),
        )
    )

    outputs["fit_time_chart"] = (
        alt.Chart(df_fit_time, title="Best Models by Fit Time Across Experiments")
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("fit_time:Q", title="Fit Time (ms)"),
            color=alt.Color("parent_run_name:N", title="Models"),
        )
    )

    outputs["fit_time_max_chart"] = (
        alt.Chart(df_fit_time_max, title="Worst Models by Fit Time Across Experiments")
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("fit_time:Q", title="Fit Time (ms)"),
            color=alt.Color("parent_run_name:N", title="Models"),
        )
    )

    outputs["score_time_chart"] = (
        alt.Chart(df_score_time, title="Best Models by Score Time Across Experiments")
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("score_time:Q", title="Score Time (ms)"),
            color=alt.Color("parent_run_name:N", title="Models"),
        )
    )

    outputs["score_time_max_chart"] = (
        alt.Chart(
            data=df_score_time_max,
            title="Worst Models by Score Time Across Experiments",
        )
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("score_time:Q", title="Score Time (ms)"),
            color=alt.Color("parent_run_name:N", title="Models"),
        )
    )

    outputs["fit_memory_chart"] = (
        alt.Chart(df_fit_memory, title="Best Models by Fit Memory Across Experiments")
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("fit_memory_peak:Q", title="Fit Memory (MB)"),
            color=alt.Color("parent_run_name:N", title="Models"),
        )
    )

    outputs["fit_memory_max_chart"] = (
        alt.Chart(
            data=df_fit_memory_max,
            title="Worst Models by Fit Memory Across Experiments",
        )
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("fit_memory_peak:Q", title="Fit Memory (MB)"),
            color=alt.Color("parent_run_name:N", title="Models"),
        )
    )

    outputs["score_memory_chart"] = (
        alt.Chart(
            data=df_score_memory,
            title="Best Models by Score Memory Across Experiments",
        )
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("score_memory_peak:Q", title="Score Memory (MB)"),
            color=alt.Color("parent_run_name:N", title="Models"),
        )
    )

    outputs["score_memory_max_chart"] = (
        alt.Chart(
            data=df_score_memory_max,
            title="Worst Models by Score Memory Across Experiments",
        )
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("score_memory_peak:Q", title="Score Memory (MB)"),
            color=alt.Color("parent_run_name:N", title="Models"),
        )
    )

    outputs["f1_macro"] = df_f1_macro
    outputs["f1_macro_min"] = df_f1_macro_min
    outputs["auc"] = df_auc
    outputs["auc_min"] = df_auc_min
    outputs["fit_time"] = df_fit_time
    outputs["fit_time_max"] = df_fit_time_max
    outputs["score_time"] = df_score_time
    outputs["score_time_max"] = df_score_time_max
    outputs["fit_memory"] = df_fit_memory
    outputs["fit_memory_max"] = df_fit_memory_max
    outputs["score_memory"] = df_score_memory
    outputs["score_memory_max"] = df_score_memory_max

    return outputs


def summarize_regression(df: pd.DataFrame):
    df_mae = df.loc[df.groupby("experiment_id")["mae"].idxmin()]
    df_mae["experiment_id"] = df_mae["experiment_id"].astype(int)
    df_mae.sort_values(by="experiment_id", inplace=True)
    df_mae["experiment_idx"] = np.arange(1, len(df_mae) + 1)

    df_mae_max = df.loc[df.groupby("experiment_id")["mae"].idxmax()]
    df_mae_max["experiment_id"] = df_mae_max["experiment_id"].astype(int)
    df_mae_max.sort_values(by="experiment_id", inplace=True)
    df_mae_max["experiment_idx"] = np.arange(1, len(df_mae_max) + 1)

    df_mape = df.loc[df.groupby("experiment_id")["mape"].idxmin()]
    df_mape["experiment_id"] = df_mape["experiment_id"].astype(int)
    df_mape.sort_values(by="experiment_id", inplace=True)
    df_mape["experiment_idx"] = np.arange(1, len(df_mape) + 1)

    df_mape_max = df.loc[df.groupby("experiment_id")["mape"].idxmax()]
    df_mape_max["experiment_id"] = df_mape_max["experiment_id"].astype(int)
    df_mape_max.sort_values(by="experiment_id", inplace=True)
    df_mape_max["experiment_idx"] = np.arange(1, len(df_mape_max) + 1)

    df_rsq = df.loc[df.groupby("experiment_id")["r_squared"].idxmax()]
    df_rsq["experiment_id"] = df_rsq["experiment_id"].astype(int)
    df_rsq.sort_values(by="experiment_id", inplace=True)
    df_rsq["experiment_idx"] = np.arange(1, len(df_rsq) + 1)

    df_rsq_min = df.loc[df.groupby("experiment_id")["r_squared"].idxmin()]
    df_rsq_min["experiment_id"] = df_rsq_min["experiment_id"].astype(int)
    df_rsq_min.sort_values(by="experiment_id", inplace=True)
    df_rsq_min["experiment_idx"] = np.arange(1, len(df_rsq_min) + 1)
    df_rsq_min["r_squared_capped"] = np.where(
        df_rsq_min["r_squared"] < -1, -1, df_rsq_min["r_squared"]
    )

    df_fit_time = df.loc[df.groupby("experiment_id")["fit_time"].idxmin()]
    df_fit_time["experiment_id"] = df_fit_time["experiment_id"].astype(int)
    df_fit_time.sort_values(by="experiment_id", inplace=True)
    df_fit_time["experiment_idx"] = np.arange(1, len(df_fit_time) + 1)
    df_fit_time["fit_time"] = df_fit_time["fit_time"] / 1_000_000

    df_fit_time_max = df.loc[df.groupby("experiment_id")["fit_time"].idxmax()]
    df_fit_time_max["experiment_id"] = df_fit_time_max["experiment_id"].astype(int)
    df_fit_time_max.sort_values(by="experiment_id", inplace=True)
    df_fit_time_max["experiment_idx"] = np.arange(1, len(df_fit_time_max) + 1)
    df_fit_time_max["fit_time"] = df_fit_time_max["fit_time"] / 1_000_000

    df_score_time = df.loc[df.groupby("experiment_id")["score_time"].idxmin()]
    df_score_time["experiment_id"] = df_score_time["experiment_id"].astype(int)
    df_score_time.sort_values(by="experiment_id", inplace=True)
    df_score_time["experiment_idx"] = np.arange(1, len(df_score_time) + 1)
    df_score_time["score_time"] = df_score_time["score_time"] / 1_000_000

    df_score_time_max = df.loc[df.groupby("experiment_id")["score_time"].idxmax()]
    df_score_time_max["experiment_id"] = df_score_time_max["experiment_id"].astype(int)
    df_score_time_max.sort_values(by="experiment_id", inplace=True)
    df_score_time_max["experiment_idx"] = np.arange(1, len(df_score_time_max) + 1)
    df_score_time_max["score_time"] = df_score_time_max["score_time"] / 1_000_000

    df_fit_memory = df.loc[df.groupby("experiment_id")["fit_memory_peak"].idxmin()]
    df_fit_memory["experiment_id"] = df_fit_memory["experiment_id"].astype(int)
    df_fit_memory.sort_values(by="experiment_id", inplace=True)
    df_fit_memory["experiment_idx"] = np.arange(1, len(df_fit_memory) + 1)
    df_fit_memory["fit_memory_peak"] = df_fit_memory["fit_memory_peak"] / 1_000_000

    df_fit_memory_max = df.loc[df.groupby("experiment_id")["fit_memory_peak"].idxmax()]
    df_fit_memory_max["experiment_id"] = df_fit_memory_max["experiment_id"].astype(int)
    df_fit_memory_max.sort_values(by="experiment_id", inplace=True)
    df_fit_memory_max["experiment_idx"] = np.arange(1, len(df_fit_memory_max) + 1)
    df_fit_memory_max["fit_memory_peak"] = df_fit_memory_max["fit_memory_peak"] / 1_000_000  # fmt: skip

    df_score_memory = df.loc[df.groupby("experiment_id")["score_memory_peak"].idxmin()]
    df_score_memory["experiment_id"] = df_score_memory["experiment_id"].astype(int)
    df_score_memory.sort_values(by="experiment_id", inplace=True)
    df_score_memory["experiment_idx"] = np.arange(1, len(df_score_memory) + 1)
    df_score_memory["score_memory_peak"] = df_score_memory["score_memory_peak"] / 1_000_000  # fmt: skip

    df_score_memory_max = df.loc[df.groupby("experiment_id")["score_memory_peak"].idxmax()]  # fmt: skip
    df_score_memory_max["experiment_id"] = df_score_memory_max["experiment_id"].astype(int)  # fmt: skip
    df_score_memory_max.sort_values(by="experiment_id", inplace=True)
    df_score_memory_max["experiment_idx"] = np.arange(1, len(df_score_memory_max) + 1)
    df_score_memory_max["score_memory_peak"] = df_score_memory_max["score_memory_peak"] / 1_000_000  # fmt: skip

    outputs = {}

    outputs["mae_chart"] = (
        alt.Chart(df_mae, title="Best Models by MAE Across Experiments")
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("mae:Q", title="Mean Absolute Error"),
            color=alt.Color("parent_run_name:N", title="Models"),
        )
    )

    outputs["mae_max_chart"] = (
        alt.Chart(df_mae_max, title="Worst Models by MAE Across Experiments")
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("mae:Q", title="Mean Absolute Error").scale(type="log"),
            color=alt.Color("parent_run_name:N", title="Models"),
        )
    )

    outputs["mape_chart"] = (
        alt.Chart(df_mape, title="Best Models by MAPE Across Experiments")
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("mape:Q", title="Mean Absolute Percentage Error"),
            color=alt.Color("parent_run_name:N", title="Models"),
        )
    )

    outputs["mape_max_chart"] = (
        alt.Chart(df_mape_max, title="Worst Models by MAPE Across Experiments")
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("mape:Q", title="Mean Absolute Percentage Error").scale(type="log"),
            color=alt.Color("parent_run_name:N", title="Models"),
        )
    )

    outputs["rsq_chart"] = (
        alt.Chart(df_rsq, title="Best Models by R-Squared Score Across Experiments")
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("r_squared:Q", title="R-Squared"),
            color=alt.Color("parent_run_name:N", title="Models"),
        )
    )

    outputs["rsq_min_chart"] = (
        alt.Chart(
            data=df_rsq_min,
            title="Worst Models by R-Squared Score Across Experiments",
        )
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("r_squared_capped:Q", title="R-Squared"),
            color=alt.Color("parent_run_name:N", title="Models"),
        )
    )

    outputs["fit_time_chart"] = (
        alt.Chart(df_fit_time, title="Best Models by Fit Time Across Experiments")
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("fit_time:Q", title="Fit Time (ms)"),
            color=alt.Color("parent_run_name:N", title="Models"),
        )
    )

    outputs["fit_time_max_chart"] = (
        alt.Chart(df_fit_time_max, title="Worst Models by Fit Time Across Experiments")
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("fit_time:Q", title="Fit Time (ms)"),
            color=alt.Color("parent_run_name:N", title="Models"),
        )
    )

    outputs["score_time_chart"] = (
        alt.Chart(df_score_time, title="Best Models by Score Time Across Experiments")
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("score_time:Q", title="Score Time (ms)"),
            color=alt.Color("parent_run_name:N", title="Models"),
        )
    )

    outputs["score_time_max_chart"] = (
        alt.Chart(
            data=df_score_time_max,
            title="Worst Models by Score Time Across Experiments",
        )
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("score_time:Q", title="Score Time (ms)"),
            color=alt.Color("parent_run_name:N", title="Models"),
        )
    )

    outputs["fit_memory_chart"] = (
        alt.Chart(df_fit_memory, title="Best Models by Fit Memory Across Experiments")
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("fit_memory_peak:Q", title="Fit Memory (MB)"),
            color=alt.Color("parent_run_name:N", title="Models"),
        )
    )

    outputs["fit_memory_max_chart"] = (
        alt.Chart(
            data=df_fit_memory_max,
            title="Worst Models by Fit Memory Across Experiments",
        )
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("fit_memory_peak:Q", title="Fit Memory (MB)"),
            color=alt.Color("parent_run_name:N", title="Models"),
        )
    )

    outputs["score_memory_chart"] = (
        alt.Chart(
            data=df_score_memory,
            title="Best Models by Score Memory Across Experiments",
        )
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("score_memory_peak:Q", title="Score Memory (MB)"),
            color=alt.Color("parent_run_name:N", title="Models"),
        )
    )

    outputs["score_memory_max_chart"] = (
        alt.Chart(
            data=df_score_memory_max,
            title="Worst Models by Score Memory Across Experiments",
        )
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("score_memory_peak:Q", title="Score Memory (MB)"),
            color=alt.Color("parent_run_name:N", title="Models"),
        )
    )

    outputs["mae"] = df_mae
    outputs["mae_max"] = df_mae_max
    outputs["mape"] = df_mape
    outputs["mape_max"] = df_mape_max
    outputs["rsq"] = df_rsq
    outputs["rsq_min"] = df_rsq_min
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
