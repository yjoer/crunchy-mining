import mlflow
import streamlit as st
from hydra import compose
from hydra import initialize

from crunchy_mining import mlflow_util
from crunchy_mining.util import plot_intrinsic_importances
from crunchy_mining.util import plot_pimp_boxplot
from crunchy_mining.util import plot_pimp_mean

st.set_page_config(layout="wide")
mlflow.set_tracking_uri("http://localhost:5001")

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
]

folds = {
    "validation": "Validation",
    "fold_1": "Fold 1",
    "fold_2": "Fold 2",
    "fold_3": "Fold 3",
    "fold_4": "Fold 4",
    "fold_5": "Fold 5",
}

cols = st.columns([1, 1, 1])
experiment = cols[0].selectbox(label="Experiments", options=experiments)

if experiment.startswith("clf"):
    model_names = [
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
    ]
elif experiment.startswith("bank") or experiment.startswith("sba"):
    model_names = [
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
    ]

model = cols[1].selectbox(label="Models", options=model_names)
fold = cols[2].selectbox(
    label="Folds",
    options=folds.keys(),
    format_func=lambda x: folds[x],
)

task, experiment_file = experiment.split("/")[:2]

if task == "clf":
    task_name = ""
else:
    task_name = f"_{task}"

with initialize(version_base=None, config_path="../../conf"):
    cfg = compose(overrides=[f"+experiment{task_name}={experiment_file}"])

mlflow.set_experiment(cfg.mlflow.experiment_name)

feature_names = cfg.vars.categorical + cfg.vars.numerical

parent_run_id = mlflow_util.get_latest_run_id_by_name(model)
run_id = mlflow_util.get_nested_run_ids_by_parent_id(parent_run_id, name=fold)

if not run_id:
    st.text("Model training is required. Please train the model before proceeding.")
    st.stop()

st.markdown("**Intrinsic and Model Specific**")

artifact_uri = f"runs:/{run_id}/interpretation/intrinsic.json"
importances = mlflow_util.load_table(artifact_uri)

if importances is None:
    st.text("Hit a roadblock! Consider running the function to generate feature importance.")  # fmt: skip
    st.stop()

chart = plot_intrinsic_importances(importances, name=model)

cols = st.columns([1, 1])
cols[0].altair_chart(chart, use_container_width=True)

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
