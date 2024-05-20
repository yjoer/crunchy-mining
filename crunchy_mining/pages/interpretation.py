import mlflow
import streamlit as st
from hydra import compose
from hydra import initialize

from crunchy_mining import mlflow_util
from crunchy_mining.pages.fragments import create_fold_selector
from crunchy_mining.util import plot_intrinsic_importances
from crunchy_mining.util import plot_pimp_boxplot
from crunchy_mining.util import plot_pimp_mean

st.set_page_config(layout="wide")
mlflow.set_tracking_uri("http://localhost:5001")

experiment, model, fold = create_fold_selector()

task_name, experiment_file = experiment.split("/")[:2]
task_suffix = "" if task_name == "clf" else f"_{task_name}"

with initialize(version_base=None, config_path="../../conf"):
    cfg = compose(overrides=[f"+experiment{task_suffix}={experiment_file}"])

mlflow.set_experiment(cfg.mlflow.experiment_name)

feature_names = cfg.vars.categorical + cfg.vars.numerical

parent_run_id = mlflow_util.get_latest_run_id_by_name(model)
run_id = mlflow_util.get_nested_run_ids_by_parent_id(parent_run_id, name=fold)

if not run_id:
    st.warning("Model training is required. Please train the model before proceeding.")
    st.stop()

st.markdown("**Intrinsic and Model Specific**")

artifact_uri = f"runs:/{run_id}/interpretation/intrinsic.json"
importances = mlflow_util.load_table(artifact_uri)

if importances is None:
    st.warning("Hit a roadblock! Consider running the function to generate feature importance.")  # fmt: skip
    st.stop()

chart = plot_intrinsic_importances(importances, name=model)

cols = st.columns([1, 1])
cols[0].altair_chart(chart, use_container_width=True, theme=None)
cols[1].dataframe(
    importances.sort_values(by="importances", ascending=False).set_index(
        keys="feature_names"
    ),
    use_container_width=True,
)

st.markdown("**Post Hoc and Model Agnostic**")

pimp = mlflow_util.load_pickle(f"runs:/{run_id}/pimp/pimp.pkl")

if not pimp:
    st.text("Encountered a snag! Consider running permutation importance.")
    st.stop()

pimp_mean_table, pimp_mean_chart = plot_pimp_mean(feature_names, pimp)

cols = st.columns([1, 1])
cols[0].altair_chart(pimp_mean_chart, use_container_width=True)
cols[1].table(
    pimp_mean_table.sort_values(
        by="importance",
        ascending=False,
        ignore_index=True,
    )
)

st.altair_chart(plot_pimp_boxplot(feature_names, pimp), use_container_width=True)

st.markdown("**Partial Dependence Plot**")

for left, right in ((x, x + 1) for x in range(0, len(feature_names), 2)):
    fig_1 = mlflow_util.load_pickle(f"runs:/{run_id}/pdp/{left}.pkl")
    fig_2 = mlflow_util.load_pickle(f"runs:/{run_id}/pdp/{right}.pkl")

    cols = st.columns([1, 1])

    if fig_1:
        cols[0].pyplot(fig_1)

    if fig_2:
        cols[1].pyplot(fig_2)
