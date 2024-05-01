import mlflow
import streamlit as st

from crunchy_mining.mlflow_util import get_latest_run_id_by_name
from crunchy_mining.mlflow_util import get_nested_run_ids_by_parent_id
from crunchy_mining.pipeline import get_variables
from crunchy_mining.preprocessing.preprocessors import GenericPreprocessor
from crunchy_mining.util import interpret_weights_logistic_regression
from crunchy_mining.util import plot_weights_logistic_regression

st.set_page_config(layout="wide")
mlflow.set_tracking_uri("http://localhost:5001")

experiments = ["Default"]

model_names = ["Logistic Regression"]

folds = {
    "validation": "Validation",
    "fold_1": "Fold 1",
    "fold_2": "Fold 2",
    "fold_3": "Fold 3",
    "fold_4": "Fold 4",
    "fold_5": "Fold 5",
}

cols = st.columns([1, 1, 1])
selected_experiment = cols[0].selectbox(label="Experiments", options=experiments)
selected_model = cols[1].selectbox(label="Models", options=model_names)

selected_fold = cols[2].selectbox(
    label="Folds",
    options=folds.keys(),
    format_func=lambda x: folds[x],
)

mlflow.set_experiment(selected_experiment)

parent_run_id = get_latest_run_id_by_name(selected_model)
run_id = get_nested_run_ids_by_parent_id(parent_run_id, name=selected_fold)

if not run_id:
    st.text("Model training is required. Please train the model before proceeding.")
    st.stop()

model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

st.markdown("**Model-Specific**")

variables = get_variables()
feature_names = variables["categorical"] + variables["numerical"]

gp = GenericPreprocessor(selected_experiment, variables)
gp.load_train_val_sets()
X_train, _, _, _ = gp.get_train_val_sets()[selected_fold]

if selected_model == "Logistic Regression":
    cols = st.columns([1, 1])

    importance = interpret_weights_logistic_regression(model, X_train)
    chart = plot_weights_logistic_regression(feature_names, importance)
    cols[0].altair_chart(chart)

st.markdown("**Model-Agnostic**")
