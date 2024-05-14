import mlflow
import streamlit as st
from hydra import compose
from hydra import initialize
from mlflow import MlflowClient

from crunchy_mining import mlflow_util
from crunchy_mining.pages.fragments import create_fold_selector
from crunchy_mining.util import plot_confusion_matrix
from crunchy_mining.util import tabulate_classification_report

st.set_page_config(layout="wide")
mlflow.set_tracking_uri("http://localhost:5001")

experiment, model, fold = create_fold_selector()

task_name, experiment_file = experiment.split("/")[:2]
task_suffix = "" if task_name == "clf" else f"_{task_name}"

with initialize(version_base=None, config_path="../../conf"):
    cfg = compose(overrides=[f"+experiment{task_suffix}={experiment_file}"])

mlflow.set_experiment(cfg.mlflow.experiment_name)

parent_run_id = mlflow_util.get_latest_run_id_by_name(model)
run_id = mlflow_util.get_nested_run_ids_by_parent_id(parent_run_id, name=fold)

if not run_id:
    run_id_warn = "Model training is required. Please train the model before proceeding."  # fmt: skip
    st.warning(run_id_warn)
    st.stop()

client = MlflowClient()
run = client.get_run(run_id)
metrics = run.data.metrics

if task_name == "clf":
    cols = st.columns([1.25, 1])

    cols[0].markdown("**Classification Report**")
    cols[0].table(tabulate_classification_report(metrics))

    cols[1].altair_chart(plot_confusion_matrix(metrics), use_container_width=True)
