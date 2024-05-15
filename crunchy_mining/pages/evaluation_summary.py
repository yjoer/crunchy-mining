import altair as alt
import mlflow
import numpy as np
import streamlit as st

from crunchy_mining import mlflow_util

st.set_page_config(layout="wide")
mlflow.set_tracking_uri("http://localhost:5001")

tabs = st.tabs(["Classification", "Loan Amount", "Loan Guarantees"])

with tabs[0]:
    with st.spinner("Fetching experiment data..."):
        df = mlflow_util.get_cv_metrics_by_task(task_name="clf")

    df_f1_macro = df.loc[df.groupby("experiment_id")["metrics.f1_macro"].idxmax()]
    df_f1_macro["experiment_id"] = df_f1_macro["experiment_id"].astype(int)
    df_f1_macro.sort_values(by="experiment_id", inplace=True)
    df_f1_macro["experiment_idx"] = np.arange(1, len(df_f1_macro) + 1)

    df_f1_macro_min = df.loc[df.groupby("experiment_id")["metrics.f1_macro"].idxmin()]
    df_f1_macro_min["experiment_id"] = df_f1_macro_min["experiment_id"].astype(int)
    df_f1_macro_min.sort_values(by="experiment_id", inplace=True)
    df_f1_macro_min["experiment_idx"] = np.arange(1, len(df_f1_macro_min) + 1)

    df_auc = df.loc[df.groupby("experiment_id")["metrics.roc_auc"].idxmax()]
    df_auc["experiment_id"] = df_auc["experiment_id"].astype(int)
    df_auc.sort_values(by="experiment_id", inplace=True)
    df_auc["experiment_idx"] = np.arange(1, len(df_auc) + 1)

    df_auc_min = df.loc[df.groupby("experiment_id")["metrics.roc_auc"].idxmin()]
    df_auc_min["experiment_id"] = df_auc_min["experiment_id"].astype(int)
    df_auc_min.sort_values(by="experiment_id", inplace=True)
    df_auc_min["experiment_idx"] = np.arange(1, len(df_auc_min) + 1)

    df_fit_time = df.loc[df.groupby("experiment_id")["metrics.fit_time"].idxmin()]
    df_fit_time["experiment_id"] = df_fit_time["experiment_id"].astype(int)
    df_fit_time.sort_values(by="experiment_id", inplace=True)
    df_fit_time["experiment_idx"] = np.arange(1, len(df_fit_time) + 1)
    df_fit_time["metrics.fit_time"] = df_fit_time["metrics.fit_time"] / 1_000_000

    df_fit_time_max = df.loc[df.groupby("experiment_id")["metrics.fit_time"].idxmax()]
    df_fit_time_max["experiment_id"] = df_fit_time_max["experiment_id"].astype(int)
    df_fit_time_max.sort_values(by="experiment_id", inplace=True)
    df_fit_time_max["experiment_idx"] = np.arange(1, len(df_fit_time_max) + 1)
    df_fit_time_max["metrics.fit_time"] = df_fit_time_max["metrics.fit_time"] / 1_000_000  # fmt: skip

    df_score_time = df.loc[df.groupby("experiment_id")["metrics.score_time"].idxmin()]
    df_score_time["experiment_id"] = df_score_time["experiment_id"].astype(int)
    df_score_time.sort_values(by="experiment_id", inplace=True)
    df_score_time["experiment_idx"] = np.arange(1, len(df_score_time) + 1)
    df_score_time["metrics.score_time"] = df_score_time["metrics.score_time"] / 1_000_000  # fmt: skip

    df_score_time_max = df.loc[df.groupby("experiment_id")["metrics.score_time"].idxmax()]  # fmt: skip
    df_score_time_max["experiment_id"] = df_score_time_max["experiment_id"].astype(int)
    df_score_time_max.sort_values(by="experiment_id", inplace=True)
    df_score_time_max["experiment_idx"] = np.arange(1, len(df_score_time_max) + 1)
    df_score_time_max["metrics.score_time"] = df_score_time_max["metrics.score_time"] / 1_000_000  # fmt: skip

    df_fit_memory = df.loc[df.groupby("experiment_id")["metrics.fit_memory_peak"].idxmin()]  # fmt: skip
    df_fit_memory["experiment_id"] = df_fit_memory["experiment_id"].astype(int)
    df_fit_memory.sort_values(by="experiment_id", inplace=True)
    df_fit_memory["experiment_idx"] = np.arange(1, len(df_fit_memory) + 1)
    df_fit_memory["metrics.fit_memory_peak"] = df_fit_memory["metrics.fit_memory_peak"] / 1_000_000  # fmt: skip

    df_fit_memory_max = df.loc[df.groupby("experiment_id")["metrics.fit_memory_peak"].idxmax()]  # fmt: skip
    df_fit_memory_max["experiment_id"] = df_fit_memory_max["experiment_id"].astype(int)
    df_fit_memory_max.sort_values(by="experiment_id", inplace=True)
    df_fit_memory_max["experiment_idx"] = np.arange(1, len(df_fit_memory_max) + 1)
    df_fit_memory_max["metrics.fit_memory_peak"] = df_fit_memory_max["metrics.fit_memory_peak"] / 1_000_000  # fmt: skip

    df_score_memory = df.loc[df.groupby("experiment_id")["metrics.score_memory_peak"].idxmin()]  # fmt: skip
    df_score_memory["experiment_id"] = df_score_memory["experiment_id"].astype(int)
    df_score_memory.sort_values(by="experiment_id", inplace=True)
    df_score_memory["experiment_idx"] = np.arange(1, len(df_score_memory) + 1)
    df_score_memory["metrics.score_memory_peak"] = df_score_memory["metrics.score_memory_peak"] / 1_000_000  # fmt: skip

    df_score_memory_max = df.loc[df.groupby("experiment_id")["metrics.score_memory_peak"].idxmax()]  # fmt: skip
    df_score_memory_max["experiment_id"] = df_score_memory_max["experiment_id"].astype(int)  # fmt: skip
    df_score_memory_max.sort_values(by="experiment_id", inplace=True)
    df_score_memory_max["experiment_idx"] = np.arange(1, len(df_score_memory_max) + 1)
    df_score_memory_max["metrics.score_memory_peak"] = df_score_memory_max["metrics.score_memory_peak"] / 1_000_000  # fmt: skip

    st.altair_chart(
        alt.Chart(df_f1_macro, title="Best Models by Macro F1 Score Across Experiments")
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("metrics\.f1_macro:Q", title="Macro F1 Score"),
            color=alt.Color("parent_run_name:N", title="Models"),
        ),
        use_container_width=True,
    )

    st.dataframe(df_f1_macro, use_container_width=True)

    st.altair_chart(
        alt.Chart(
            df_f1_macro_min, title="Worst Models by Macro F1 Score Across Experiments"
        )
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("metrics\.f1_macro:Q", title="Macro F1 Score"),
            color=alt.Color("parent_run_name:N", title="Models"),
        ),
        use_container_width=True,
    )

    st.dataframe(df_f1_macro_min, use_container_width=True)

    st.altair_chart(
        alt.Chart(df_auc, title="Best Models by AUC Score Across Experiments")
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("metrics\.roc_auc:Q", title="ROC AUC"),
            color=alt.Color("parent_run_name:N", title="Models"),
        ),
        use_container_width=True,
    )

    st.dataframe(df_auc, use_container_width=True)

    st.altair_chart(
        alt.Chart(df_auc_min, title="Worst Models by AUC Score Across Experiments")
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("metrics\.roc_auc:Q", title="ROC AUC"),
            color=alt.Color("parent_run_name:N", title="Models"),
        ),
        use_container_width=True,
    )

    st.dataframe(df_auc_min, use_container_width=True)

    st.altair_chart(
        alt.Chart(df_fit_time, title="Best Models by Fit Time Across Experiments")
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("metrics\.fit_time:Q", title="Fit Time (ms)"),
            color=alt.Color("parent_run_name:N", title="Models"),
        ),
        use_container_width=True,
    )

    st.dataframe(df_fit_time, use_container_width=True)

    st.altair_chart(
        alt.Chart(df_fit_time_max, title="Worst Models by Fit Time Across Experiments")
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("metrics\.fit_time:Q", title="Fit Time (ms)"),
            color=alt.Color("parent_run_name:N", title="Models"),
        ),
        use_container_width=True,
    )

    st.dataframe(df_fit_time_max, use_container_width=True)

    st.altair_chart(
        alt.Chart(df_score_time, title="Best Models by Score Time Across Experiments")
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("metrics\.score_time:Q", title="Score Time (ms)"),
            color=alt.Color("parent_run_name:N", title="Models"),
        ),
        use_container_width=True,
    )

    st.dataframe(df_score_time, use_container_width=True)

    st.altair_chart(
        alt.Chart(
            df_score_time_max, title="Worst Models by Score Time Across Experiments"
        )
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("metrics\.score_time:Q", title="Score Time (ms)"),
            color=alt.Color("parent_run_name:N", title="Models"),
        ),
        use_container_width=True,
    )

    st.dataframe(df_score_time_max, use_container_width=True)

    st.altair_chart(
        alt.Chart(df_fit_memory, title="Best Models by Fit Memory Across Experiments")
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("metrics\.fit_memory_peak:Q", title="Fit Memory (MB)"),
            color=alt.Color("parent_run_name:N", title="Models"),
        ),
        use_container_width=True,
    )

    st.dataframe(df_fit_memory, use_container_width=True)

    st.altair_chart(
        alt.Chart(
            df_fit_memory_max, title="Worst Models by Fit Memory Across Experiments"
        )
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("metrics\.fit_memory_peak:Q", title="Fit Memory (MB)"),
            color=alt.Color("parent_run_name:N", title="Models"),
        ),
        use_container_width=True,
    )

    st.dataframe(df_fit_memory_max, use_container_width=True)

    st.altair_chart(
        alt.Chart(
            df_score_memory, title="Best Models by Score Memory Across Experiments"
        )
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("metrics\.score_memory_peak:Q", title="Score Memory (MB)"),
            color=alt.Color("parent_run_name:N", title="Models"),
        ),
        use_container_width=True,
    )

    st.dataframe(df_score_memory, use_container_width=True)

    st.altair_chart(
        alt.Chart(
            df_score_memory_max, title="Worst Models by Score Memory Across Experiments"
        )
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("metrics\.score_memory_peak:Q", title="Score Memory (MB)"),
            color=alt.Color("parent_run_name:N", title="Models"),
        ),
        use_container_width=True,
    )

    st.dataframe(df_score_memory_max, use_container_width=True)
