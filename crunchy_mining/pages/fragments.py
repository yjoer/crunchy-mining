import streamlit as st

task_names = {
    "clf": "Classification",
    "bank": "Loan Amounts",
    "sba": "Loan Guarantees",
}

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
    "bank/preprocessing_v1",
    "bank/preprocessing_v2",
    "bank/preprocessing_v3",
    "bank/preprocessing_v4",
    "bank/preprocessing_v5",
    "bank/preprocessing_v6",
    "bank/preprocessing_v7",
    "bank/preprocessing_v8",
    "bank/preprocessing_v9",
    "bank/preprocessing_v10",
    "bank/preprocessing_v11",
    "sba/sampling_v1",
    "sba/sampling_v2",
    "sba/preprocessing_v1",
    "sba/preprocessing_v2",
    "sba/preprocessing_v3",
    "sba/preprocessing_v4",
    "sba/preprocessing_v5",
    "sba/preprocessing_v6",
    "sba/preprocessing_v7",
    "sba/preprocessing_v8",
    "sba/preprocessing_v9",
    "sba/preprocessing_v10",
    "sba/preprocessing_v11",
]

model_names = {
    "clf": [
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
    ],
    "bank": [
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
    ],
    "sba": [
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
    ],
}

folds = {
    "validation": "Validation",
    "fold_1": "Fold 1",
    "fold_2": "Fold 2",
    "fold_3": "Fold 3",
    "fold_4": "Fold 4",
    "fold_5": "Fold 5",
}


def task_model_selector(task_names, model_names):
    cols = st.columns([1, 1, 1])

    task = cols[0].selectbox(
        label="Tasks",
        options=task_names.keys(),
        format_func=lambda x: task_names[x],
    )

    model = cols[1].selectbox(label="Models", options=model_names[task])

    return task, model


def experiment_selector(experiments):
    cols = st.columns([1, 1, 1])
    return cols[0].selectbox(label="Experiments", options=experiments, key=0)


def fold_selector(experiments, model_names, folds):
    cols = st.columns([1, 1, 1])
    experiment = cols[0].selectbox(label="Experiments", options=experiments, key=1)
    task_name = experiment.split("/")[0]

    model_names = model_names[task_name]
    model = cols[1].selectbox(label="Models", options=model_names, key=2)

    fold = cols[2].selectbox(
        label="Folds",
        options=folds.keys(),
        format_func=lambda x: folds[x],
    )

    return experiment, model, fold


def model_selector(experiments, model_names):
    cols = st.columns([1, 1, 1])
    experiment = cols[0].selectbox(label="Experiments", options=experiments, key=3)
    task_name = experiment.split("/")[0]

    model_names = model_names[task_name]
    model = cols[1].selectbox(label="Models", options=model_names, key=4)

    return experiment, model


def create_task_model_selector():
    return task_model_selector(task_names, model_names)


def create_experiment_selector():
    return experiment_selector(experiments)


def create_fold_selector():
    return fold_selector(experiments, model_names, folds)


def create_model_selector():
    return model_selector(experiments, model_names)
