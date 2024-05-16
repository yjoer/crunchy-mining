import mlflow
import streamlit as st

from crunchy_mining import mlflow_util
from crunchy_mining.util import summarize_classification

st.set_page_config(layout="wide")
mlflow.set_tracking_uri("http://localhost:5001")

tabs = st.tabs(["Classification", "Loan Amount", "Loan Guarantees"])

with tabs[0]:
    with st.spinner("Fetching experiment data..."):
        df = mlflow_util.get_cv_metrics_by_task(task_name="clf")

    outputs = summarize_classification(df)

    st.altair_chart(outputs["f1_macro_chart"], use_container_width=True)
    st.dataframe(outputs["f1_macro"], use_container_width=True)

    st.altair_chart(outputs["f1_macro_min_chart"], use_container_width=True)
    st.dataframe(outputs["f1_macro_min"], use_container_width=True)

    st.altair_chart(outputs["auc_chart"], use_container_width=True)
    st.dataframe(outputs["auc"], use_container_width=True)

    st.altair_chart(outputs["auc_min_chart"], use_container_width=True)
    st.dataframe(outputs["auc_min"], use_container_width=True)

    st.altair_chart(outputs["fit_time_chart"], use_container_width=True)
    st.dataframe(outputs["fit_time"], use_container_width=True)

    st.altair_chart(outputs["fit_time_max_chart"], use_container_width=True)
    st.dataframe(outputs["fit_time_max"], use_container_width=True)

    st.altair_chart(outputs["score_time_chart"], use_container_width=True)
    st.dataframe(outputs["score_time"], use_container_width=True)

    st.altair_chart(outputs["score_time_max_chart"], use_container_width=True)
    st.dataframe(outputs["score_time_max"], use_container_width=True)

    st.altair_chart(outputs["fit_memory_chart"], use_container_width=True)
    st.dataframe(outputs["fit_memory"], use_container_width=True)

    st.altair_chart(outputs["fit_memory_max_chart"], use_container_width=True)
    st.dataframe(outputs["fit_memory_max"], use_container_width=True)

    st.altair_chart(outputs["score_memory_chart"], use_container_width=True)
    st.dataframe(outputs["score_memory"], use_container_width=True)

    st.altair_chart(outputs["score_memory_max_chart"], use_container_width=True)
    st.dataframe(outputs["score_memory_max"], use_container_width=True)