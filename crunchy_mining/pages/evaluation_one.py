import mlflow
import streamlit as st
from hydra import compose
from hydra import initialize

from crunchy_mining import mlflow_util
from crunchy_mining.pages.fragments import create_model_selector
from crunchy_mining.util import plot_confusion_matrix
from crunchy_mining.util import tabulate_classification_report

st.set_page_config(layout="wide")
mlflow.set_tracking_uri("http://localhost:5001")

experiment, model = create_model_selector()

task_name, experiment_file = experiment.split("/")[:2]
task_suffix = "" if task_name == "clf" else f"_{task_name}"

with initialize(version_base=None, config_path="../../conf"):
    cfg = compose(overrides=[f"+experiment{task_suffix}={experiment_file}"])

mlflow.set_experiment(cfg.mlflow.experiment_name)

parent_run_id = mlflow_util.get_latest_run_id_by_name(model)

validation_run = mlflow_util.get_nested_runs_by_parent_id(
    parent_run_id,
    filter_string="run_name = 'validation'",
)

cv_runs = mlflow_util.get_nested_runs_by_parent_id(
    parent_run_id,
    filter_string="run_name LIKE 'fold%'",
)

if validation_run is None or cv_runs is None:
    run_id_warn = "Model training is required. Please train the model before proceeding."  # fmt: skip
    st.warning(run_id_warn)
    st.stop()

val_metrics = (
    validation_run.filter(like="metrics", axis=1)
    .iloc[0]
    .rename(lambda x: x.replace("metrics.", ""))
    .to_dict()
)

cv_metrics = (
    cv_runs.filter(like="metrics", axis=1)
    .mean()
    .rename(lambda x: x.replace("metrics.", ""))
    .to_dict()
)

if task_name == "clf":
    st.markdown("**Validation**")
    cols = st.columns([1.25, 1])

    cols[0].markdown("**Classification Report**")
    cols[0].table(tabulate_classification_report(val_metrics))

    cols[1].altair_chart(plot_confusion_matrix(val_metrics), use_container_width=True)

    st.markdown("**Cross-Validation**")
    cols = st.columns([1.25, 1])

    cols[0].markdown("**Classification Report**")
    cols[0].table(tabulate_classification_report(cv_metrics))

    cols[1].altair_chart(plot_confusion_matrix(cv_metrics), use_container_width=True)
