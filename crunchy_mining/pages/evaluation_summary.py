import mlflow
import streamlit as st

from crunchy_mining import mlflow_util
from crunchy_mining.util import summarize_classification
from crunchy_mining.util import summarize_regression

st.set_page_config(layout="wide")
mlflow.set_tracking_uri("http://localhost:5001")

tabs = st.tabs(["Classification", "Loan Amounts", "Loan Guarantees"])

with tabs[0]:
    with st.spinner("Fetching experiment data..."):
        df = mlflow_util.get_cv_metrics_by_task(task_name="clf")

    outputs = summarize_classification(df)
    chart_conf = {"use_container_width": True, "theme": None}

    st.altair_chart(outputs["f1_macro_chart"], **chart_conf)
    st.dataframe(outputs["f1_macro"], use_container_width=True)

    st.altair_chart(outputs["f1_macro_min_chart"], **chart_conf)
    st.dataframe(outputs["f1_macro_min"], use_container_width=True)

    st.altair_chart(outputs["auc_chart"], **chart_conf)
    st.dataframe(outputs["auc"], use_container_width=True)

    st.altair_chart(outputs["auc_min_chart"], **chart_conf)
    st.dataframe(outputs["auc_min"], use_container_width=True)

    st.altair_chart(outputs["fit_time_chart"], **chart_conf)
    st.dataframe(outputs["fit_time"], use_container_width=True)

    st.altair_chart(outputs["fit_time_max_chart"], **chart_conf)
    st.dataframe(outputs["fit_time_max"], use_container_width=True)

    st.altair_chart(outputs["score_time_chart"], **chart_conf)
    st.dataframe(outputs["score_time"], use_container_width=True)

    st.altair_chart(outputs["score_time_max_chart"], **chart_conf)
    st.dataframe(outputs["score_time_max"], use_container_width=True)

    st.altair_chart(outputs["fit_memory_chart"], **chart_conf)
    st.dataframe(outputs["fit_memory"], use_container_width=True)

    st.altair_chart(outputs["fit_memory_max_chart"], **chart_conf)
    st.dataframe(outputs["fit_memory_max"], use_container_width=True)

    st.altair_chart(outputs["score_memory_chart"], **chart_conf)
    st.dataframe(outputs["score_memory"], use_container_width=True)

    st.altair_chart(outputs["score_memory_max_chart"], **chart_conf)
    st.dataframe(outputs["score_memory_max"], use_container_width=True)

with tabs[1]:
    with st.spinner("Fetching experiment data..."):
        df = mlflow_util.get_cv_metrics_by_task(task_name="bank")

    outputs = summarize_regression(df)

    st.altair_chart(outputs["mae_chart"], **chart_conf)
    st.dataframe(outputs["mae"], use_container_width=True)

    st.altair_chart(outputs["mae_max_chart"], **chart_conf)
    st.dataframe(outputs["mae_max"], use_container_width=True)

    st.altair_chart(outputs["mape_chart"], **chart_conf)
    st.dataframe(outputs["mape"], use_container_width=True)

    st.altair_chart(outputs["mape_max_chart"], **chart_conf)
    st.dataframe(outputs["mape_max"], use_container_width=True)

    st.altair_chart(outputs["rsq_chart"], **chart_conf)
    st.dataframe(outputs["rsq"], use_container_width=True)

    st.altair_chart(outputs["rsq_min_chart"], **chart_conf)
    st.dataframe(outputs["rsq_min"], use_container_width=True)

    st.altair_chart(outputs["fit_time_chart"], **chart_conf)
    st.dataframe(outputs["fit_time"], use_container_width=True)

    st.altair_chart(outputs["fit_time_max_chart"], **chart_conf)
    st.dataframe(outputs["fit_time_max"], use_container_width=True)

    st.altair_chart(outputs["score_time_chart"], **chart_conf)
    st.dataframe(outputs["score_time"], use_container_width=True)

    st.altair_chart(outputs["score_time_max_chart"], **chart_conf)
    st.dataframe(outputs["score_time_max"], use_container_width=True)

    st.altair_chart(outputs["fit_memory_chart"], **chart_conf)
    st.dataframe(outputs["fit_memory"], use_container_width=True)

    st.altair_chart(outputs["fit_memory_max_chart"], **chart_conf)
    st.dataframe(outputs["fit_memory_max"], use_container_width=True)

    st.altair_chart(outputs["score_memory_chart"], **chart_conf)
    st.dataframe(outputs["score_memory"], use_container_width=True)

    st.altair_chart(outputs["score_memory_max_chart"], **chart_conf)
    st.dataframe(outputs["score_memory_max"], use_container_width=True)

with tabs[2]:
    with st.spinner("Fetching experiment data..."):
        df = mlflow_util.get_cv_metrics_by_task(task_name="sba")

    outputs = summarize_regression(df)

    st.altair_chart(outputs["mae_chart"], **chart_conf)
    st.dataframe(outputs["mae"], use_container_width=True)

    st.altair_chart(outputs["mae_max_chart"], **chart_conf)
    st.dataframe(outputs["mae_max"], use_container_width=True)

    st.altair_chart(outputs["mape_chart"], **chart_conf)
    st.dataframe(outputs["mape"], use_container_width=True)

    st.altair_chart(outputs["mape_max_chart"], **chart_conf)
    st.dataframe(outputs["mape_max"], use_container_width=True)

    st.altair_chart(outputs["rsq_chart"], **chart_conf)
    st.dataframe(outputs["rsq"], use_container_width=True)

    st.altair_chart(outputs["rsq_min_chart"], **chart_conf)
    st.dataframe(outputs["rsq_min"], use_container_width=True)

    st.altair_chart(outputs["fit_time_chart"], **chart_conf)
    st.dataframe(outputs["fit_time"], use_container_width=True)

    st.altair_chart(outputs["fit_time_max_chart"], **chart_conf)
    st.dataframe(outputs["fit_time_max"], use_container_width=True)

    st.altair_chart(outputs["score_time_chart"], **chart_conf)
    st.dataframe(outputs["score_time"], use_container_width=True)

    st.altair_chart(outputs["score_time_max_chart"], **chart_conf)
    st.dataframe(outputs["score_time_max"], use_container_width=True)

    st.altair_chart(outputs["fit_memory_chart"], **chart_conf)
    st.dataframe(outputs["fit_memory"], use_container_width=True)

    st.altair_chart(outputs["fit_memory_max_chart"], **chart_conf)
    st.dataframe(outputs["fit_memory_max"], use_container_width=True)

    st.altair_chart(outputs["score_memory_chart"], **chart_conf)
    st.dataframe(outputs["score_memory"], use_container_width=True)

    st.altair_chart(outputs["score_memory_max_chart"], **chart_conf)
    st.dataframe(outputs["score_memory_max"], use_container_width=True)
