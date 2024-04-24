from typing import List
from typing import Tuple

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.util import evaluate_classification
from src.util import trace_memory


def inspect_holdout_split_size(
    df_train: pd.DataFrame,
    df_train_sm: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    target_variable: str,
):
    return pd.concat(
        [
            df_train[target_variable].value_counts().rename("train"),
            df_train_sm[target_variable].value_counts().rename("train (sm)"),
            df_val[target_variable].value_counts().rename("validation"),
            df_test[target_variable].value_counts().rename("test"),
        ],
        axis=1,
    )


def inspect_cv_split_size(
    df_train: pd.DataFrame,
    indices: List[Tuple],
    target_variable: str,
):
    folds = []

    for i, (train_index, val_index) in enumerate(indices):
        count_by_split = pd.concat(
            [
                df_train.iloc[train_index][target_variable]
                .value_counts()
                .rename("train"),
                df_train.iloc[val_index][target_variable]
                .value_counts()
                .rename("validation"),
            ],
            axis=1,
        )

        count_by_split.reset_index(inplace=True)
        count_by_split.index = [i] * len(count_by_split)
        count_by_split.index.name = "Folds"

        folds.append(count_by_split)

    return pd.concat(folds).pivot(
        columns=target_variable,
        values=["train", "validation"],
    )


def preprocessing_v1(df_train: pd.DataFrame, df_test: pd.DataFrame, variables: dict):
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train_cat = oe.fit_transform(df_train[variables["categorical"]])
    X_test_cat = oe.transform(df_test[variables["categorical"]])

    mm = MinMaxScaler()
    X_train_scaled = mm.fit_transform(X_train_cat)
    X_test_scaled = mm.transform(X_test_cat)

    ss = StandardScaler()
    X_train_num = ss.fit_transform(df_train[variables["numerical"]])
    X_test_num = ss.transform(df_test[variables["numerical"]])

    X_train = np.hstack((X_train_scaled, X_train_num))
    X_test = np.hstack((X_test_scaled, X_test_num))

    le = LabelEncoder()
    y_train = le.fit_transform(df_train[variables["target"]])
    y_test = le.transform(df_test[variables["target"]])

    return X_train, y_train, X_test, y_test


def train_knn(X_train, y_train):
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)

    return knn


def validate_knn(train_val_sets: dict):
    with mlflow.start_run(run_name="KNN"):
        for name, (X_train, y_train, X_val, y_val) in train_val_sets.items():
            with mlflow.start_run(run_name=name, nested=True):
                with trace_memory() as trace:
                    knn = train_knn(X_train, y_train)

                y_knn = knn.predict(X_val)
                mlflow.log_metrics(evaluate_classification(y_val, y_knn))
                mlflow.log_metric("peak_memory_usage", trace["peak"])
                mlflow.log_params(knn.get_params())

                mlflow.sklearn.log_model(
                    sk_model=knn,
                    artifact_path="model",
                    signature=mlflow.models.infer_signature(X_val, y_knn),
                )


def train_logistic_regression(X_train, y_train):
    params = {
        "random_state": 12345,
        "n_jobs": -1,
    }

    logreg = LogisticRegression(**params)
    logreg.fit(X_train, y_train)

    return logreg


def validate_logistic_regression(train_val_sets: dict):
    with mlflow.start_run(run_name="Logistic Regression"):
        for name, (X_train, y_train, X_val, y_val) in train_val_sets.items():
            with mlflow.start_run(run_name=name, nested=True):
                with trace_memory() as trace:
                    logreg = train_logistic_regression(X_train, y_train)

                y_logreg = logreg.predict(X_val)
                mlflow.log_metrics(evaluate_classification(y_val, y_logreg))
                mlflow.log_metric("peak_memory_usage", trace["peak"])
                mlflow.log_params(logreg.get_params())

                mlflow.sklearn.log_model(
                    sk_model=logreg,
                    artifact_path="model",
                    signature=mlflow.models.infer_signature(X_val, y_logreg),
                )


def train_gaussian_nb(X_train, y_train):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    return gnb


def validate_gaussian_nb(train_val_sets: dict):
    with mlflow.start_run(run_name="Gaussian NB"):
        for name, (X_train, y_train, X_val, y_val) in train_val_sets.items():
            with mlflow.start_run(run_name=name, nested=True):
                with trace_memory() as trace:
                    gnb = train_gaussian_nb(X_train, y_train)

                y_gnb = gnb.predict(X_val)
                mlflow.log_metrics(evaluate_classification(y_val, y_gnb))
                mlflow.log_metric("peak_memory_usage", trace["peak"])
                mlflow.log_params(gnb.get_params())

                mlflow.sklearn.log_model(
                    sk_model=gnb,
                    artifact_path="model",
                    signature=mlflow.models.infer_signature(X_val, y_gnb),
                )


def train_linear_svc(X_train, y_train):
    params = {
        "dual": "auto",
        "random_state": 12345,
    }

    svc = LinearSVC(**params)
    svc.fit(X_train, y_train)

    return svc


def validate_linear_svc(train_val_sets: dict):
    with mlflow.start_run(run_name="Linear SVC"):
        for name, (X_train, y_train, X_val, y_val) in train_val_sets.items():
            with mlflow.start_run(run_name=name, nested=True):
                with trace_memory() as trace:
                    svc = train_linear_svc(X_train, y_train)

                y_svc = svc.predict(X_val)
                mlflow.log_metrics(evaluate_classification(y_val, y_svc))
                mlflow.log_metric("peak_memory_usage", trace["peak"])
                mlflow.log_params(svc.get_params())

                mlflow.sklearn.log_model(
                    sk_model=svc,
                    artifact_path="model",
                    signature=mlflow.models.infer_signature(X_val, y_svc),
                )


def train_decision_tree(X_train, y_train):
    params = {
        "random_state": 12345,
    }

    dt = DecisionTreeClassifier(**params)
    dt.fit(X_train, y_train)

    return dt


def validate_decision_tree(train_val_sets: dict):
    with mlflow.start_run(run_name="Decision Tree"):
        for name, (X_train, y_train, X_val, y_val) in train_val_sets.items():
            with mlflow.start_run(run_name=name, nested=True):
                with trace_memory() as trace:
                    dt = train_decision_tree(X_train, y_train)

                y_dt = dt.predict(X_val)
                mlflow.log_metrics(evaluate_classification(y_val, y_dt))
                mlflow.log_metric("peak_memory_usage", trace["peak"])
                mlflow.log_params(dt.get_params())

                mlflow.sklearn.log_model(
                    sk_model=dt,
                    artifact_path="model",
                    signature=mlflow.models.infer_signature(X_val, y_dt),
                )


def train_random_forest(X_train, y_train):
    params = {
        "n_jobs": -1,
        "random_state": 12345,
    }

    rf = RandomForestClassifier(**params)
    rf.fit(X_train, y_train)

    return rf


def validate_random_forest(train_val_sets: dict):
    with mlflow.start_run(run_name="Random Forest"):
        for name, (X_train, y_train, X_val, y_val) in train_val_sets.items():
            with mlflow.start_run(run_name=name, nested=True):
                with trace_memory() as trace:
                    rf = train_random_forest(X_train, y_train)

                y_rf = rf.predict(X_val)
                mlflow.log_metrics(evaluate_classification(y_val, y_rf))
                mlflow.log_metric("peak_memory_usage", trace["peak"])
                mlflow.log_params(rf.get_params())

                mlflow.sklearn.log_model(
                    sk_model=rf,
                    artifact_path="model",
                    signature=mlflow.models.infer_signature(X_val, y_rf),
                )


def train_adaboost(X_train, y_train):
    params = {
        "algorithm": "SAMME",
        "random_state": 12345,
    }

    ab = AdaBoostClassifier(**params)
    ab.fit(X_train, y_train)

    return ab


def validate_adaboost(train_val_sets: dict):
    with mlflow.start_run(run_name="AdaBoost"):
        for name, (X_train, y_train, X_val, y_val) in train_val_sets.items():
            with mlflow.start_run(run_name=name, nested=True):
                with trace_memory() as trace:
                    ab = train_adaboost(X_train, y_train)

                y_ab = ab.predict(X_val)
                mlflow.log_metrics(evaluate_classification(y_val, y_ab))
                mlflow.log_metric("peak_memory_usage", trace["peak"])
                mlflow.log_params(ab.get_params())

                mlflow.sklearn.log_model(
                    sk_model=ab,
                    artifact_path="model",
                    signature=mlflow.models.infer_signature(X_val, y_ab),
                )


def train_xgboost(X_train, y_train):
    params = {
        "n_jobs": -1,
        "random_state": 12345,
    }

    xgb = XGBClassifier(**params)
    xgb.fit(X_train, y_train)

    return xgb


def validate_xgboost(train_val_sets: dict):
    with mlflow.start_run(run_name="XGBoost"):
        for name, (X_train, y_train, X_val, y_val) in train_val_sets.items():
            with mlflow.start_run(run_name=name, nested=True):
                with trace_memory() as trace:
                    xgb = train_xgboost(X_train, y_train)

                y_xgb = xgb.predict(X_val)
                mlflow.log_metrics(evaluate_classification(y_val, y_xgb))
                mlflow.log_metric("peak_memory_usage", trace["peak"])
                mlflow.log_params(xgb.get_params())

                mlflow.xgboost.log_model(
                    xgb_model=xgb,
                    artifact_path="model",
                    signature=mlflow.models.infer_signature(X_val, y_xgb),
                )


def train_lightgbm(X_train, y_train):
    params = {
        "random_state": 12345,
        "n_jobs": -1,
    }

    lgb = LGBMClassifier(**params)
    lgb.fit(X_train, y_train)

    return lgb


def validate_lightgbm(train_val_sets: dict):
    with mlflow.start_run(run_name="LightGBM"):
        for name, (X_train, y_train, X_val, y_val) in train_val_sets.items():
            with mlflow.start_run(run_name=name, nested=True):
                with trace_memory() as trace:
                    lgb = train_lightgbm(X_train, y_train)

                y_lgb = lgb.predict(X_val)
                mlflow.log_metrics(evaluate_classification(y_val, y_lgb))
                mlflow.log_metric("peak_memory_usage", trace["peak"])
                mlflow.log_params(lgb.get_params())

                mlflow.lightgbm.log_model(
                    lgb_model=lgb,
                    artifact_path="model",
                    signature=mlflow.models.infer_signature(X_val, y_lgb),
                )


def train_catboost(X_train, y_train):
    params = {
        "metric_period": 250,
        "random_state": 12345,
    }

    catb = CatBoostClassifier(**params)
    catb.fit(X_train, y_train)

    return catb


def validate_catboost(train_val_sets: dict):
    with mlflow.start_run(run_name="CatBoost"):
        for name, (X_train, y_train, X_val, y_val) in train_val_sets.items():
            with mlflow.start_run(run_name=name, nested=True):
                with trace_memory() as trace:
                    catb = train_catboost(X_train, y_train)

                y_catb = catb.predict(X_val)
                mlflow.log_metrics(evaluate_classification(y_val, y_catb))
                mlflow.log_metric("peak_memory_usage", trace["peak"])
                mlflow.log_params(catb.get_params())

                mlflow.catboost.log_model(
                    cb_model=catb,
                    artifact_path="model",
                    signature=mlflow.models.infer_signature(X_val, y_catb),
                )
