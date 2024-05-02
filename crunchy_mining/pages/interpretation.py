import mlflow
import streamlit as st

from crunchy_mining import mlflow_util
from crunchy_mining.pipeline import get_variables
from crunchy_mining.preprocessing.preprocessors import GenericPreprocessor
from crunchy_mining.util import interpret_gain_lightgbm
from crunchy_mining.util import interpret_gain_xgboost
from crunchy_mining.util import interpret_impurity_adaboost
from crunchy_mining.util import interpret_impurity_decision_tree
from crunchy_mining.util import interpret_impurity_random_forest
from crunchy_mining.util import interpret_pvc_catboost
from crunchy_mining.util import interpret_weights_linear_svc
from crunchy_mining.util import interpret_weights_logistic_regression
from crunchy_mining.util import plot_gain_lightgbm
from crunchy_mining.util import plot_gain_xgboost
from crunchy_mining.util import plot_impurity_adaboost
from crunchy_mining.util import plot_impurity_decision_tree
from crunchy_mining.util import plot_impurity_random_forest
from crunchy_mining.util import plot_pimp_boxplot
from crunchy_mining.util import plot_pimp_mean
from crunchy_mining.util import plot_pvc_catboost
from crunchy_mining.util import plot_weights_linear_svc
from crunchy_mining.util import plot_weights_logistic_regression

st.set_page_config(layout="wide")
mlflow.set_tracking_uri("http://localhost:5001")

experiments = ["Default"]

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

parent_run_id = mlflow_util.get_latest_run_id_by_name(selected_model)
run_id = mlflow_util.get_nested_run_ids_by_parent_id(parent_run_id, name=selected_fold)

if not run_id:
    st.text("Model training is required. Please train the model before proceeding.")
    st.stop()

st.markdown("**Intrinsic and Model Specific**")

variables = get_variables()
feature_names = variables["categorical"] + variables["numerical"]

gp = GenericPreprocessor(selected_experiment, variables)
gp.load_train_val_sets()
X_train, _, _, _ = gp.get_train_val_sets()[selected_fold]

match selected_model:
    case "KNN":
        st.text("N/A")
    case "Logistic Regression":
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        importance = interpret_weights_logistic_regression(model, X_train)
        chart = plot_weights_logistic_regression(feature_names, importance)

        cols = st.columns([1, 1])
        cols[0].altair_chart(chart, use_container_width=True)
    case "Gaussian NB":
        st.text("N/A")
    case "Linear SVC":
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        importance = interpret_weights_linear_svc(model, X_train)
        chart = plot_weights_linear_svc(feature_names, importance)

        cols = st.columns([1, 1])
        cols[0].altair_chart(chart, use_container_width=True)
    case "Decision Tree":
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        importance = interpret_impurity_decision_tree(model)
        chart = plot_impurity_decision_tree(feature_names, importance)

        cols = st.columns([1, 1])
        cols[0].altair_chart(chart, use_container_width=True)
    case "Random Forest":
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        importance = interpret_impurity_random_forest(model)
        chart = plot_impurity_random_forest(feature_names, importance)

        cols = st.columns([1, 1])
        cols[0].altair_chart(chart, use_container_width=True)
    case "AdaBoost":
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        importance = interpret_impurity_adaboost(model)
        chart = plot_impurity_adaboost(feature_names, importance)

        cols = st.columns([1, 1])
        cols[0].altair_chart(chart, use_container_width=True)
    case "XGBoost":
        model = mlflow.xgboost.load_model(f"runs:/{run_id}/model")
        importance = interpret_gain_xgboost(model)
        chart = plot_gain_xgboost(feature_names, importance)

        cols = st.columns([1, 1])
        cols[0].altair_chart(chart, use_container_width=True)
    case "LightGBM":
        model = mlflow.lightgbm.load_model(f"runs:/{run_id}/model")
        importance = interpret_gain_lightgbm(model)
        chart = plot_gain_lightgbm(feature_names, importance)

        cols = st.columns([1, 1])
        cols[0].altair_chart(chart, use_container_width=True)
    case "CatBoost":
        model = mlflow.catboost.load_model(f"runs:/{run_id}/model")
        importance = interpret_pvc_catboost(model)
        chart = plot_pvc_catboost(feature_names, importance)

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
