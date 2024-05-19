import mlflow
import numpy as np
import streamlit as st

from crunchy_mining.mlflow_util import get_cv_metrics_by_model
from crunchy_mining.pages.fragments import create_task_model_selector
from crunchy_mining.pages.fragments import plot_auc_score_by_experiments
from crunchy_mining.pages.fragments import plot_f_score_by_experiments
from crunchy_mining.pages.fragments import plot_mae_by_experiments
from crunchy_mining.pages.fragments import plot_mape_by_experiments
from crunchy_mining.pages.fragments import plot_memory_by_experiments
from crunchy_mining.pages.fragments import plot_r_squared_score_by_experiments
from crunchy_mining.pages.fragments import plot_time_by_experiments

st.set_page_config(layout="wide")
mlflow.set_tracking_uri("http://localhost:5001")

task, model = create_task_model_selector()

with st.spinner("Fetching experiment data..."):
    df_cv_metrics = get_cv_metrics_by_model(task, model)
    df_cv_metrics["experiment_id"] = df_cv_metrics["experiment_id"].astype(int)
    df_cv_metrics.sort_values(by="experiment_id", inplace=True)
    df_cv_metrics["experiment_idx"] = np.arange(1, len(df_cv_metrics) + 1)

if task == "clf":
    plot_f_score_by_experiments(df_cv_metrics)
    plot_auc_score_by_experiments(df_cv_metrics)

elif task == "bank" or task == "sba":
    plot_mae_by_experiments(df_cv_metrics)
    plot_mape_by_experiments(df_cv_metrics)
    plot_r_squared_score_by_experiments(df_cv_metrics)

plot_time_by_experiments(df_cv_metrics)
plot_memory_by_experiments(df_cv_metrics)

st.dataframe(df_cv_metrics)
