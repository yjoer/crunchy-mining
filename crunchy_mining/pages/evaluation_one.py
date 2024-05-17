import altair as alt
import mlflow
import numpy as np
import streamlit as st

from crunchy_mining.mlflow_util import get_cv_metrics_by_model
from crunchy_mining.pages.fragments import create_task_model_selector

st.set_page_config(layout="wide")
mlflow.set_tracking_uri("http://localhost:5001")

task, model = create_task_model_selector()

with st.spinner("Fetching experiment data..."):
    df_cv_metrics = get_cv_metrics_by_model(task, model)
    df_cv_metrics["experiment_id"] = df_cv_metrics["experiment_id"].astype(int)
    df_cv_metrics.sort_values(by="experiment_id", inplace=True)
    df_cv_metrics["experiment_idx"] = np.arange(1, len(df_cv_metrics) + 1)

if task == "clf":
    st.altair_chart(
        alt.Chart(
            data=df_cv_metrics,
            title="Sampling and Preprocessing Techniques by Macro F1 Score Across Experiments",
        )
        .mark_bar()
        .encode(
            x=alt.X(
                shorthand="experiment_idx:N",
                title="Experiments",
                axis=alt.Axis(labelAngle=0, labelPadding=8),
                sort=alt.Sort("x"),
            ),
            y=alt.Y("f1_macro:Q", title="Macro F1 Score"),
        ),
        use_container_width=True,
    )

    st.altair_chart(
        alt.Chart(
            data=df_cv_metrics,
            title="Sampling and Preprocessing Techniques by AUC Score Across Experiments",
        )
        .mark_bar()
        .encode(
            x=alt.X(
                shorthand="experiment_idx:N",
                title="Experiments",
                axis=alt.Axis(labelAngle=0, labelPadding=8),
                sort=alt.Sort("x"),
            ),
            y=alt.Y("roc_auc:Q", title="AUC Score"),
        ),
        use_container_width=True,
    )
elif task == "bank" or task == "sba":
    df_cv_metrics["r_squared_capped"] = np.where(
        df_cv_metrics["r_squared"] < -1, -1, df_cv_metrics["r_squared"]
    )

    st.altair_chart(
        alt.Chart(
            data=df_cv_metrics,
            title="Sampling and Preprocessing Techniques by MAE Across Experiments",
        )
        .mark_bar()
        .encode(
            x=alt.X(
                shorthand="experiment_idx:N",
                title="Experiments",
                axis=alt.Axis(labelAngle=0, labelPadding=8),
                sort=alt.Sort("x"),
            ),
            y=alt.Y("mae:Q", title="Mean Absolute Error", scale=alt.Scale(type="log")),
        ),
        use_container_width=True,
    )

    st.altair_chart(
        alt.Chart(
            data=df_cv_metrics,
            title="Sampling and Preprocessing Techniques by MAPE Across Experiments",
        )
        .mark_bar()
        .encode(
            x=alt.X(
                shorthand="experiment_idx:N",
                title="Experiments",
                axis=alt.Axis(labelAngle=0, labelPadding=8),
                sort=alt.Sort("x"),
            ),
            y=alt.Y(
                shorthand="mape:Q",
                title="Mean Absolute Percentage Error",
                scale=alt.Scale(type="log"),
            ),
        ),
        use_container_width=True,
    )

    st.altair_chart(
        alt.Chart(
            data=df_cv_metrics,
            title="Sampling and Preprocessing Techniques by R-Squared Score Across Experiments",
        )
        .mark_bar()
        .encode(
            x=alt.X(
                shorthand="experiment_idx:N",
                title="Experiments",
                axis=alt.Axis(labelAngle=0, labelPadding=8),
                sort=alt.Sort("x"),
            ),
            y=alt.Y("r_squared_capped:Q", title="R-Squared Score"),
        ),
        use_container_width=True,
    )

df_cv_time = (
    df_cv_metrics[["experiment_idx", "fit_time", "score_time"]]
    .melt(id_vars="experiment_idx", var_name="metrics", value_name="value")
    .assign(value=lambda x: x["value"] / 1_000_000)
)

df_cv_memory = (
    df_cv_metrics[["experiment_idx", "fit_memory_peak", "score_memory_peak"]]
    .rename(lambda x: x.replace("_peak", ""), axis=1)
    .melt(id_vars="experiment_idx", var_name="metrics", value_name="value")
    .assign(value=lambda x: x["value"] / 1_000_000)
)

st.altair_chart(
    alt.Chart(
        data=df_cv_time,
        title="Sampling and Preprocessing Techniques by Training and Scoring Time Across Experiments",
    )
    .mark_bar(width={"band": 1})
    .encode(
        x=alt.X(
            shorthand="experiment_idx:N",
            title="Experiments",
            axis=alt.Axis(labelAngle=0, labelPadding=8),
            sort=alt.Sort("x"),
        ),
        y=alt.Y("value:Q", title="Time Taken (ms)", scale=alt.Scale(type="log")),
        xOffset=alt.XOffset("metrics:N", title="Metrics"),
        color=alt.Color("metrics:N", title="Metrics"),
    ),
    use_container_width=True,
)

st.altair_chart(
    alt.Chart(
        data=df_cv_memory,
        title="Sampling and Preprocessing Techniques by Training and Scoring Memory Usage Across Experiments",
    )
    .mark_bar(width={"band": 1})
    .encode(
        x=alt.X(
            "experiment_idx:N",
            title="Experiments",
            axis=alt.Axis(labelAngle=0, labelPadding=8),
            sort=alt.Sort("x"),
        ),
        y=alt.Y("value:Q", title="Memory Usage (MB)"),
        xOffset=alt.XOffset("metrics:N", title="Metrics"),
        color=alt.Color("metrics:N", title="Metrics"),
    ),
    use_container_width=True,
)

st.dataframe(df_cv_metrics)
