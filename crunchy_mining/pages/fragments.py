import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

task_names = {
    "clf": "Classification",
    "bank": "Loan Amounts",
    "sba": "Loan Guarantees",
}

experiments = [
    "clf/sampling_v1",
    "clf/sampling_v2",
    "clf/preprocessing_v1",
    "clf/preprocessing_v2",
    "clf/preprocessing_v3",
    "clf/preprocessing_v4",
    "clf/preprocessing_v5",
    "clf/preprocessing_v6",
    "clf/preprocessing_v7",
    "clf/resampling_v1",
    "clf/resampling_v2",
    "clf/resampling_v3",
    "clf/resampling_v4",
    "clf/resampling_v5",
    "clf/resampling_v6",
    "clf/resampling_v7",
    "clf/resampling_v8",
    "bank/sampling_v1",
    "bank/sampling_v2",
    "bank/preprocessing_v1",
    "bank/preprocessing_v2",
    "bank/preprocessing_v3",
    "bank/preprocessing_v4",
    "bank/preprocessing_v5",
    "bank/preprocessing_v6",
    "bank/preprocessing_v7",
    "bank/preprocessing_v8",
    "bank/preprocessing_v9",
    "bank/preprocessing_v10",
    "bank/preprocessing_v11",
    "sba/sampling_v1",
    "sba/sampling_v2",
    "sba/preprocessing_v1",
    "sba/preprocessing_v2",
    "sba/preprocessing_v3",
    "sba/preprocessing_v4",
    "sba/preprocessing_v5",
    "sba/preprocessing_v6",
    "sba/preprocessing_v7",
    "sba/preprocessing_v8",
    "sba/preprocessing_v9",
    "sba/preprocessing_v10",
    "sba/preprocessing_v11",
]

model_names = {
    "clf": [
        "KNN",
        "Logistic Regression",
        "Gaussian NB",
        "Linear SVC",
        "Decision Tree",
        "Random Forest",
        "AdaBoost",
        "XGBoost",
        "LightGBM",
        "CatBoost",
    ],
    "bank": [
        "Linear Regression",
        "Lasso Regression",
        "Ridge Regression",
        "Elastic Net",
        "Decision Tree",
        "Random Forest",
        "AdaBoost",
        "XGBoost",
        "LightGBM",
        "CatBoost",
    ],
    "sba": [
        "Linear Regression",
        "Lasso Regression",
        "Ridge Regression",
        "Elastic Net",
        "Decision Tree",
        "Random Forest",
        "AdaBoost",
        "XGBoost",
        "LightGBM",
        "CatBoost",
    ],
}

folds = {
    "validation": "Validation",
    "fold_1": "Fold 1",
    "fold_2": "Fold 2",
    "fold_3": "Fold 3",
    "fold_4": "Fold 4",
    "fold_5": "Fold 5",
}


def task_model_selector(task_names, model_names):
    cols = st.columns([1, 1, 1])

    task = cols[0].selectbox(
        label="Tasks",
        options=task_names.keys(),
        format_func=lambda x: task_names[x],
    )

    model = cols[1].selectbox(label="Models", options=model_names[task])

    return task, model


def experiment_selector(experiments):
    cols = st.columns([1, 1, 1])
    return cols[0].selectbox(label="Experiments", options=experiments, key=0)


def experiment_model_selector(experiments, model_names):
    cols = st.columns([1, 1, 1])
    experiment = cols[0].selectbox(label="Experiments", options=experiments, key=1)
    task_name = experiment.split("/")[0]

    model_names = model_names[task_name]
    model = cols[1].selectbox(label="Models", options=model_names, key=2)

    return experiment, model


def fold_selector(experiments, model_names, folds):
    cols = st.columns([1, 1, 1])
    experiment = cols[0].selectbox(label="Experiments", options=experiments, key=3)
    task_name = experiment.split("/")[0]

    model_names = model_names[task_name]
    model = cols[1].selectbox(label="Models", options=model_names, key=4)

    fold = cols[2].selectbox(
        label="Folds",
        options=folds.keys(),
        format_func=lambda x: folds[x],
    )

    return experiment, model, fold


def create_task_model_selector():
    return task_model_selector(task_names, model_names)


def create_experiment_selector():
    return experiment_selector(experiments)


def create_experiment_model_selector():
    return experiment_model_selector(experiments, model_names)


def create_fold_selector():
    return fold_selector(experiments, model_names, folds)


def _create_formatted_values(df: pd.DataFrame, prefix: str):
    v = df[f"{prefix}_mean"].astype(str) + " ± " + df[f"{prefix}_std"].astype(str)
    df["formatted_values"] = v


def plot_f_score_by_experiments(df_cv_metrics: pd.DataFrame):
    _create_formatted_values(df_cv_metrics, prefix="f1_macro")

    return st.altair_chart(
        alt.Chart(
            data=df_cv_metrics,
            title=alt.TitleParams(
                text="Macro F1 Score Across Experiments",
                anchor="start",
            ),
        )
        .mark_bar()
        .encode(
            x=alt.X(
                shorthand="experiment_idx:N",
                title="Experiments",
                axis=alt.Axis(labelAngle=0),
                sort=alt.Sort("x"),
            ),
            y=alt.Y("f1_macro_mean:Q", title="Macro F1 Score"),
            tooltip=[
                alt.Tooltip("experiment_idx", title="Experiments"),
                alt.Tooltip("formatted_values", title="Macro F1 Score"),
            ],
        )
        + (
            alt.Chart(df_cv_metrics)
            .mark_errorbar()
            .encode(
                x=alt.X("experiment_idx:N", title="Experiments"),
                y=alt.Y("f1_macro_mean:Q", title="Macro F1 Score"),
                yError=alt.YError("f1_macro_std:Q"),
            )
        ),
        use_container_width=True,
        theme=None,
    )


def plot_auc_score_by_experiments(df_cv_metrics: pd.DataFrame):
    _create_formatted_values(df_cv_metrics, prefix="roc_auc")

    return st.altair_chart(
        alt.Chart(
            data=df_cv_metrics,
            title=alt.TitleParams(
                text="AUC Score Across Experiments",
                anchor="start",
            ),
        )
        .mark_bar()
        .encode(
            x=alt.X(
                shorthand="experiment_idx:N",
                title="Experiments",
                axis=alt.Axis(labelAngle=0),
                sort=alt.Sort("x"),
            ),
            y=alt.Y("roc_auc_mean:Q", title="AUC Score"),
            tooltip=[
                alt.Tooltip("experiment_idx", title="Experiments"),
                alt.Tooltip("formatted_values", title="AUC Score"),
            ],
        )
        + (
            alt.Chart(df_cv_metrics)
            .mark_errorbar()
            .encode(
                x=alt.X("experiment_idx:N", title="Experiments"),
                y=alt.Y("roc_auc_mean:Q", title="AUC Score"),
                yError=alt.YError("roc_auc_std:Q"),
            )
        ),
        use_container_width=True,
        theme=None,
    )


def plot_mae_by_experiments(df_cv_metrics: pd.DataFrame):
    _create_formatted_values(df_cv_metrics, prefix="mae")
    df_cv_metrics["mae_std_capped"] = np.where(df_cv_metrics["mae_std"] > df_cv_metrics["mae_mean"], 0.99 * df_cv_metrics["mae_mean"], df_cv_metrics["mae_std"])  # fmt: skip

    return st.altair_chart(
        alt.Chart(
            data=df_cv_metrics,
            title=alt.TitleParams(
                text="MAE Across Experiments",
                anchor="start",
            ),
        )
        .mark_bar()
        .encode(
            x=alt.X(
                shorthand="experiment_idx:N",
                title="Experiments",
                axis=alt.Axis(labelAngle=0),
                sort=alt.Sort("x"),
            ),
            y=alt.Y(
                shorthand="mae_mean:Q",
                title="Mean Absolute Error",
                scale=alt.Scale(type="log"),
            ),
            tooltip=[
                alt.Tooltip("experiment_idx", title="Experiments"),
                alt.Tooltip("formatted_values", title="Mean Absolute Error"),
            ],
        )
        + (
            alt.Chart(df_cv_metrics)
            .mark_errorbar()
            .encode(
                x=alt.X("experiment_idx:N", title="Experiments"),
                y=alt.Y("mae_mean:Q", title="Mean Absolute Error"),
                yError=alt.YError("mae_std_capped:Q"),
            )
        ),
        use_container_width=True,
        theme=None,
    )


def plot_mape_by_experiments(df_cv_metrics: pd.DataFrame):
    _create_formatted_values(df_cv_metrics, prefix="mape")
    df_cv_metrics["mape_std_capped"] = np.where(df_cv_metrics["mape_std"] > df_cv_metrics["mape_mean"], 0.99 * df_cv_metrics["mape_mean"], df_cv_metrics["mape_std"])  # fmt: skip

    return st.altair_chart(
        alt.Chart(
            data=df_cv_metrics,
            title=alt.TitleParams(
                text="MAPE Across Experiments",
                anchor="start",
            ),
        )
        .mark_bar()
        .encode(
            x=alt.X(
                shorthand="experiment_idx:N",
                title="Experiments",
                axis=alt.Axis(labelAngle=0),
                sort=alt.Sort("x"),
            ),
            y=alt.Y(
                shorthand="mape_mean:Q",
                title="Mean Absolute Percentage Error",
                scale=alt.Scale(type="log"),
            ),
            tooltip=[
                alt.Tooltip("experiment_idx", title="Experiments"),
                alt.Tooltip("formatted_values", title="Mean Absolute Percentage Error"),
            ],
        )
        + (
            alt.Chart(df_cv_metrics)
            .mark_errorbar()
            .encode(
                x=alt.X("experiment_idx:N", title="Experiments"),
                y=alt.Y("mape_mean:Q", title="Mean Absolute Percentage Error"),
                yError=alt.YError("mape_std_capped:Q"),
            )
        ),
        use_container_width=True,
        theme=None,
    )


def plot_r_squared_score_by_experiments(df_cv_metrics: pd.DataFrame):
    _create_formatted_values(df_cv_metrics, prefix="r_squared")
    df_cv_metrics["r_squared_capped"] = np.where(df_cv_metrics["r_squared_mean"] < -1, -1, df_cv_metrics["r_squared_mean"])  # fmt: skip
    df_cv_metrics["r_squared_std_capped"] = np.where(np.abs(df_cv_metrics["r_squared_std"]) > np.abs(df_cv_metrics["r_squared_capped"]), 0.99 * np.abs(df_cv_metrics["r_squared_capped"]), df_cv_metrics["r_squared_std"])  # fmt: skip

    st.altair_chart(
        alt.Chart(
            data=df_cv_metrics,
            title=alt.TitleParams(
                text="R-Squared Score Across Experiments",
                anchor="start",
            ),
        )
        .mark_bar()
        .encode(
            x=alt.X(
                shorthand="experiment_idx:N",
                title="Experiments",
                axis=alt.Axis(labelAngle=0),
                sort=alt.Sort("x"),
            ),
            y=alt.Y("r_squared_capped:Q", title="R-Squared Score"),
            tooltip=[
                alt.Tooltip("experiment_idx", title="Experiments"),
                alt.Tooltip("formatted_values", title="R-Squared Score"),
            ],
        )
        + (
            alt.Chart(df_cv_metrics)
            .mark_errorbar()
            .encode(
                x=alt.X("experiment_idx:N", title="Experiments"),
                y=alt.Y("r_squared_capped:Q", title="R-Squared Score"),
                yError=alt.YError("r_squared_std_capped:Q"),
            )
        ),
        use_container_width=True,
        theme=None,
    )


def plot_time_by_experiments(df_cv_metrics: pd.DataFrame):
    df_cv_metrics["fit_time_std_capped"] = np.where(df_cv_metrics["fit_time_std"] >= df_cv_metrics["fit_time_mean"], 0.99 * df_cv_metrics["fit_time_mean"], df_cv_metrics["fit_time_std"])  # fmt: skip
    df_cv_metrics["score_time_std_capped"] = np.where(df_cv_metrics["score_time_std"] >= df_cv_metrics["score_time_mean"], 0.99 * df_cv_metrics["score_time_mean"], df_cv_metrics["score_time_std"])  # fmt: skip

    cols_mean = ["experiment_idx", "fit_time_mean", "score_time_mean"]
    cols_std = ["experiment_idx", "fit_time_std_capped", "score_time_std_capped"]

    to_replace = {
        "fit_time_mean": "fit_time",
        "score_time_mean": "score_time",
        "fit_time_std_capped": "fit_time",
        "score_time_std_capped": "score_time",
    }

    df_cv_mean = (
        df_cv_metrics[cols_mean]
        .melt(id_vars="experiment_idx", var_name="metrics", value_name="mean")
        .replace(to_replace)
        .assign(mean=lambda x: x["mean"] / 1_000_000)
    )

    df_cv_std = (
        df_cv_metrics[cols_std]
        .melt(id_vars="experiment_idx", var_name="metrics", value_name="std")
        .replace(to_replace)
        .assign(std=lambda x: x["std"] / 1_000_000)
    )

    df_cv = df_cv_mean.merge(df_cv_std, on=["experiment_idx", "metrics"])

    return st.altair_chart(
        alt.Chart(
            data=df_cv,
            title=alt.TitleParams(
                text="Training and Scoring Time Across Experiments",
                anchor="start",
            ),
        )
        .mark_bar(width={"band": 1})
        .encode(
            x=alt.X(
                shorthand="experiment_idx:N",
                title="Experiments",
                axis=alt.Axis(labelAngle=0),
                sort=alt.Sort("x"),
            ),
            y=alt.Y("mean:Q", title="Time Taken (ms)", scale=alt.Scale(type="log")),
            xOffset=alt.XOffset("metrics:N", title="Metrics"),
            color=alt.Color("metrics:N", title="Metrics"),
        )
        + (
            alt.Chart(df_cv)
            .mark_errorbar()
            .encode(
                x=alt.X("experiment_idx:N", title="Experiments"),
                xOffset=alt.XOffset("metrics:N", title="Metrics"),
                y=alt.Y("mean:Q", title="Time Taken (ms)"),
                yError=alt.YError("std:Q"),
            )
        ),
        use_container_width=True,
        theme=None,
    )


def plot_memory_by_experiments(df_cv_metrics: pd.DataFrame):
    df_cv_metrics["fit_memory_peak_std_capped"] = np.where(df_cv_metrics["fit_memory_peak_std"] >= df_cv_metrics["fit_memory_peak_mean"], 0.99 * df_cv_metrics["fit_memory_peak_mean"], df_cv_metrics["fit_memory_peak_std"])  # fmt: skip
    df_cv_metrics["score_memory_peak_std_capped"] = np.where(df_cv_metrics["score_memory_peak_std"] >= df_cv_metrics["score_memory_peak_mean"], 0.99 * df_cv_metrics["score_memory_peak_mean"], df_cv_metrics["score_memory_peak_std"])  # fmt: skip

    cols_mean = ["experiment_idx", "fit_memory_peak_mean", "score_memory_peak_mean"]
    cols_std = ["experiment_idx", "fit_memory_peak_std_capped", "score_memory_peak_std_capped"]  # fmt: skip

    to_replace = {
        "fit_memory_peak_mean": "fit_memory_peak",
        "score_memory_peak_mean": "score_memory_peak",
        "fit_memory_peak_std_capped": "fit_memory_peak",
        "score_memory_peak_std_capped": "score_memory_peak",
    }

    df_cv_mean = (
        df_cv_metrics[cols_mean]
        .melt(id_vars="experiment_idx", var_name="metrics", value_name="mean")
        .replace(to_replace)
        .assign(mean=lambda x: x["mean"] / 1_000_000)
    )

    df_cv_std = (
        df_cv_metrics[cols_std]
        .melt(id_vars="experiment_idx", var_name="metrics", value_name="std")
        .replace(to_replace)
        .assign(std=lambda x: x["std"] / 1_000_000)
    )

    df_cv = df_cv_mean.merge(df_cv_std, on=["experiment_idx", "metrics"])

    return st.altair_chart(
        alt.Chart(
            data=df_cv,
            title=alt.TitleParams(
                text="Training and Scoring Memory Usage Across Experiments",
                anchor="start",
            ),
        )
        .mark_bar(width={"band": 1})
        .encode(
            x=alt.X(
                shorthand="experiment_idx:N",
                title="Experiments",
                axis=alt.Axis(labelAngle=0),
                sort=alt.Sort("x"),
            ),
            xOffset=alt.XOffset("metrics:N", title="Metrics"),
            y=alt.Y("mean:Q", title="Memory Usage (MB)"),
            color=alt.Color("metrics:N", title="Metrics"),
        )
        + (
            alt.Chart(df_cv)
            .mark_errorbar()
            .encode(
                x=alt.X("experiment_idx:N", title="Experiments"),
                xOffset=alt.XOffset("metrics:N", title="Metrics"),
                y=alt.Y("mean:Q", title="Memory Usage (MB)"),
                yError=alt.YError("std:Q"),
            )
        ),
        use_container_width=True,
        theme=None,
    )
