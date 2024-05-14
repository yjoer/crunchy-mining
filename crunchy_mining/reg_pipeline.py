import mlflow
import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from .preprocessing.base_preprocessor import BasePreprocessor
from .util import evaluate_regression
from .util import trace_memory


def train_linear_regression(X_train, y_train):
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    return lr


def validate_linear_regression(preprocessor: BasePreprocessor):
    encoders = preprocessor.get_encoders()
    train_val_sets = preprocessor.get_train_val_sets()

    with mlflow.start_run(run_name="Linear Regression"):
        for name, (X_train, y_train, X_val, y_val) in train_val_sets.items():
            if name == "testing":
                continue

            with mlflow.start_run(run_name=name, nested=True):
                with trace_memory() as fit_trace:
                    lr = train_linear_regression(X_train, y_train)

                with trace_memory() as score_trace:
                    y_lr = lr.predict(X_val)

                    if "y_min_max" in encoders:
                        y_lr = (
                            encoders["y_min_max"]
                            .inverse_transform(y_lr.reshape(-1, 1))
                            .ravel()
                        )
                    elif "y_log" in encoders:
                        y_lr = np.expm1(y_lr)

                mlflow.log_metrics(evaluate_regression(y_val, y_lr))
                mlflow.log_metric("fit_time", fit_trace["duration"])
                mlflow.log_metric("fit_memory_peak", fit_trace["peak"])
                mlflow.log_metric("score_time", score_trace["duration"])
                mlflow.log_metric("score_memory_peak", score_trace["peak"])
                mlflow.log_params(lr.get_params())

                mlflow.sklearn.log_model(
                    sk_model=lr,
                    artifact_path="model",
                    signature=mlflow.models.infer_signature(X_train, y_lr),
                )


def train_lasso(X_train, y_train):
    lasso = Lasso()
    lasso.fit(X_train, y_train)

    return lasso


def validate_lasso(preprocessor: BasePreprocessor):
    encoders = preprocessor.get_encoders()
    train_val_sets = preprocessor.get_train_val_sets()

    with mlflow.start_run(run_name="Lasso Regression"):
        for name, (X_train, y_train, X_val, y_val) in train_val_sets.items():
            if name == "testing":
                continue

            with mlflow.start_run(run_name=name, nested=True):
                with trace_memory() as fit_trace:
                    lasso = train_lasso(X_train, y_train)

                with trace_memory() as score_trace:
                    y_lasso = lasso.predict(X_val)

                    if "y_min_max" in encoders:
                        y_lasso = (
                            encoders["y_min_max"]
                            .inverse_transform(y_lasso.reshape(-1, 1))
                            .ravel()
                        )
                    elif "y_log" in encoders:
                        y_lasso = np.expm1(y_lasso)

                mlflow.log_metrics(evaluate_regression(y_val, y_lasso))
                mlflow.log_metric("fit_time", fit_trace["duration"])
                mlflow.log_metric("fit_memory_peak", fit_trace["peak"])
                mlflow.log_metric("score_time", score_trace["duration"])
                mlflow.log_metric("score_memory_peak", score_trace["peak"])
                mlflow.log_params(lasso.get_params())

                mlflow.sklearn.log_model(
                    sk_model=lasso,
                    artifact_path="model",
                    signature=mlflow.models.infer_signature(X_train, y_lasso),
                )


def train_ridge(X_train, y_train):
    ridge = Ridge()
    ridge.fit(X_train, y_train)

    return ridge


def validate_ridge(preprocessor: BasePreprocessor):
    encoders = preprocessor.get_encoders()
    train_val_sets = preprocessor.get_train_val_sets()

    with mlflow.start_run(run_name="Ridge Regression"):
        for name, (X_train, y_train, X_val, y_val) in train_val_sets.items():
            if name == "testing":
                continue

            with mlflow.start_run(run_name=name, nested=True):
                with trace_memory() as fit_trace:
                    ridge = train_ridge(X_train, y_train)

                with trace_memory() as score_trace:
                    y_ridge = ridge.predict(X_val)

                    if "y_min_max" in encoders:
                        y_ridge = (
                            encoders["y_min_max"]
                            .inverse_transform(y_ridge.reshape(-1, 1))
                            .ravel()
                        )
                    elif "y_log" in encoders:
                        y_ridge = np.expm1(y_ridge)

                mlflow.log_metrics(evaluate_regression(y_val, y_ridge))
                mlflow.log_metric("fit_time", fit_trace["duration"])
                mlflow.log_metric("fit_memory_peak", fit_trace["peak"])
                mlflow.log_metric("score_time", score_trace["duration"])
                mlflow.log_metric("score_memory_peak", score_trace["peak"])
                mlflow.log_params(ridge.get_params())

                mlflow.sklearn.log_model(
                    sk_model=ridge,
                    artifact_path="model",
                    signature=mlflow.models.infer_signature(X_train, y_ridge),
                )


def train_elastic_net(X_train, y_train):
    en = ElasticNet()
    en.fit(X_train, y_train)

    return en


def validate_elastic_net(preprocessor: BasePreprocessor):
    encoders = preprocessor.get_encoders()
    train_val_sets = preprocessor.get_train_val_sets()

    with mlflow.start_run(run_name="Elastic Net"):
        for name, (X_train, y_train, X_val, y_val) in train_val_sets.items():
            if name == "testing":
                continue

            with mlflow.start_run(run_name=name, nested=True):
                with trace_memory() as fit_trace:
                    en = train_elastic_net(X_train, y_train)

                with trace_memory() as score_trace:
                    y_en = en.predict(X_val)

                    if "y_min_max" in encoders:
                        y_en = (
                            encoders["y_min_max"]
                            .inverse_transform(y_en.reshape(-1, 1))
                            .ravel()
                        )
                    elif "y_log" in encoders:
                        y_en = np.expm1(y_en)

                mlflow.log_metrics(evaluate_regression(y_val, y_en))
                mlflow.log_metric("fit_time", fit_trace["duration"])
                mlflow.log_metric("fit_memory_peak", fit_trace["peak"])
                mlflow.log_metric("score_time", score_trace["duration"])
                mlflow.log_metric("score_memory_peak", score_trace["peak"])
                mlflow.log_params(en.get_params())

                mlflow.sklearn.log_model(
                    sk_model=en,
                    artifact_path="model",
                    signature=mlflow.models.infer_signature(X_train, y_en),
                )


def train_decision_tree(X_train, y_train):
    params = {
        "random_state": 12345,
    }

    dt = DecisionTreeRegressor(**params)
    dt.fit(X_train, y_train)

    return dt


def validate_decision_tree(preprocessor: BasePreprocessor):
    encoders = preprocessor.get_encoders()
    train_val_sets = preprocessor.get_train_val_sets()

    with mlflow.start_run(run_name="Decision Tree"):
        for name, (X_train, y_train, X_val, y_val) in train_val_sets.items():
            if name == "testing":
                continue

            with mlflow.start_run(run_name=name, nested=True):
                with trace_memory() as fit_trace:
                    dt = train_decision_tree(X_train, y_train)

                with trace_memory() as score_trace:
                    y_dt = dt.predict(X_val)

                    if "y_min_max" in encoders:
                        y_dt = (
                            encoders["y_min_max"]
                            .inverse_transform(y_dt.reshape(-1, 1))
                            .ravel()
                        )
                    elif "y_log" in encoders:
                        y_dt = np.expm1(y_dt)

                mlflow.log_metrics(evaluate_regression(y_val, y_dt))
                mlflow.log_metric("fit_time", fit_trace["duration"])
                mlflow.log_metric("fit_memory_peak", fit_trace["peak"])
                mlflow.log_metric("score_time", score_trace["duration"])
                mlflow.log_metric("score_memory_peak", score_trace["peak"])
                mlflow.log_params(dt.get_params())

                mlflow.sklearn.log_model(
                    sk_model=dt,
                    artifact_path="model",
                    signature=mlflow.models.infer_signature(X_train, y_dt),
                )


def train_random_forest(X_train, y_train):
    params = {
        "n_jobs": -1,
        "random_state": 12345,
    }

    rf = RandomForestRegressor(**params)
    rf.fit(X_train, y_train)

    return rf


def validate_random_forest(preprocessor: BasePreprocessor):
    encoders = preprocessor.get_encoders()
    train_val_sets = preprocessor.get_train_val_sets()

    with mlflow.start_run(run_name="Random Forest"):
        for name, (X_train, y_train, X_val, y_val) in train_val_sets.items():
            if name == "testing":
                continue

            with mlflow.start_run(run_name=name, nested=True):
                with trace_memory() as fit_trace:
                    rf = train_random_forest(X_train, y_train)

                with trace_memory() as score_trace:
                    y_rf = rf.predict(X_val)

                    if "y_min_max" in encoders:
                        y_rf = (
                            encoders["y_min_max"]
                            .inverse_transform(y_rf.reshape(-1, 1))
                            .ravel()
                        )
                    elif "y_log" in encoders:
                        y_rf = np.expm1(y_rf)

                mlflow.log_metrics(evaluate_regression(y_val, y_rf))
                mlflow.log_metric("fit_time", fit_trace["duration"])
                mlflow.log_metric("fit_memory_peak", fit_trace["peak"])
                mlflow.log_metric("score_time", score_trace["duration"])
                mlflow.log_metric("score_memory_peak", score_trace["peak"])
                mlflow.log_params(rf.get_params())

                mlflow.sklearn.log_model(
                    sk_model=rf,
                    artifact_path="model",
                    signature=mlflow.models.infer_signature(X_train, y_rf),
                )


def train_adaboost(X_train, y_train):
    params = {
        "random_state": 12345,
    }

    ab = AdaBoostRegressor(**params)
    ab.fit(X_train, y_train)

    return ab


def validate_adaboost(preprocessor: BasePreprocessor):
    encoders = preprocessor.get_encoders()
    train_val_sets = preprocessor.get_train_val_sets()

    with mlflow.start_run(run_name="AdaBoost"):
        for name, (X_train, y_train, X_val, y_val) in train_val_sets.items():
            if name == "testing":
                continue

            with mlflow.start_run(run_name=name, nested=True):
                with trace_memory() as fit_trace:
                    ab = train_adaboost(X_train, y_train)

                with trace_memory() as score_trace:
                    y_ab = ab.predict(X_val)

                    if "y_min_max" in encoders:
                        y_ab = (
                            encoders["y_min_max"]
                            .inverse_transform(y_ab.reshape(-1, 1))
                            .ravel()
                        )
                    elif "y_log" in encoders:
                        y_ab = np.expm1(y_ab)

                mlflow.log_metrics(evaluate_regression(y_val, y_ab))
                mlflow.log_metric("fit_time", fit_trace["duration"])
                mlflow.log_metric("fit_memory_peak", fit_trace["peak"])
                mlflow.log_metric("score_time", score_trace["duration"])
                mlflow.log_metric("score_memory_peak", score_trace["peak"])
                mlflow.log_params(ab.get_params())

                mlflow.sklearn.log_model(
                    sk_model=ab,
                    artifact_path="model",
                    signature=mlflow.models.infer_signature(X_train, y_ab),
                )


def train_xgboost(X_train, y_train):
    params = {
        "n_jobs": -1,
        "random_state": 12345,
    }

    xgb = XGBRegressor(**params)
    xgb.fit(X_train, y_train)

    return xgb


def validate_xgboost(preprocessor: BasePreprocessor):
    encoders = preprocessor.get_encoders()
    train_val_sets = preprocessor.get_train_val_sets()

    with mlflow.start_run(run_name="XGBoost"):
        for name, (X_train, y_train, X_val, y_val) in train_val_sets.items():
            if name == "testing":
                continue

            with mlflow.start_run(run_name=name, nested=True):
                with trace_memory() as fit_trace:
                    xgb = train_xgboost(X_train, y_train)

                with trace_memory() as score_trace:
                    y_xgb = xgb.predict(X_val)

                    if "y_min_max" in encoders:
                        y_xgb = (
                            encoders["y_min_max"]
                            .inverse_transform(y_xgb.reshape(-1, 1))
                            .ravel()
                        )
                    elif "y_log" in encoders:
                        y_xgb = np.expm1(y_xgb)

                mlflow.log_metrics(evaluate_regression(y_val, y_xgb))
                mlflow.log_metric("fit_time", fit_trace["duration"])
                mlflow.log_metric("fit_memory_peak", fit_trace["peak"])
                mlflow.log_metric("score_time", score_trace["duration"])
                mlflow.log_metric("score_memory_peak", score_trace["peak"])
                mlflow.log_params(xgb.get_params())

                mlflow.xgboost.log_model(
                    xgb_model=xgb,
                    artifact_path="model",
                    signature=mlflow.models.infer_signature(X_train, y_xgb),
                    model_format="json",
                )


def train_lightgbm(X_train, y_train):
    params = {
        "random_state": 12345,
        "n_jobs": -1,
    }

    lgb = LGBMRegressor(**params)
    lgb.fit(X_train, y_train)

    return lgb


def validate_lightgbm(preprocessor: BasePreprocessor):
    encoders = preprocessor.get_encoders()
    train_val_sets = preprocessor.get_train_val_sets()

    with mlflow.start_run(run_name="LightGBM"):
        for name, (X_train, y_train, X_val, y_val) in train_val_sets.items():
            if name == "testing":
                continue

            with mlflow.start_run(run_name=name, nested=True):
                with trace_memory() as fit_trace:
                    lgb = train_lightgbm(X_train, y_train)

                with trace_memory() as score_trace:
                    y_lgb = lgb.predict(X_val)

                    if "y_min_max" in encoders:
                        y_lgb = (
                            encoders["y_min_max"]
                            .inverse_transform(y_lgb.reshape(-1, 1))
                            .ravel()
                        )
                    elif "y_log" in encoders:
                        y_lgb = np.expm1(y_lgb)

                mlflow.log_metrics(evaluate_regression(y_val, y_lgb))
                mlflow.log_metric("fit_time", fit_trace["duration"])
                mlflow.log_metric("fit_memory_peak", fit_trace["peak"])
                mlflow.log_metric("score_time", score_trace["duration"])
                mlflow.log_metric("score_memory_peak", score_trace["peak"])
                mlflow.log_params(lgb.get_params())

                mlflow.lightgbm.log_model(
                    lgb_model=lgb,
                    artifact_path="model",
                    signature=mlflow.models.infer_signature(X_train, y_lgb),
                )


def train_catboost(X_train, y_train):
    params = {
        "metric_period": 250,
        "random_state": 12345,
    }

    catb = CatBoostRegressor(**params)
    catb.fit(X_train, y_train)

    return catb


def validate_catboost(preprocessor: BasePreprocessor):
    encoders = preprocessor.get_encoders()
    train_val_sets = preprocessor.get_train_val_sets()

    with mlflow.start_run(run_name="CatBoost"):
        for name, (X_train, y_train, X_val, y_val) in train_val_sets.items():
            if name == "testing":
                continue

            with mlflow.start_run(run_name=name, nested=True):
                with trace_memory() as fit_trace:
                    catb = train_catboost(X_train, y_train)

                with trace_memory() as score_trace:
                    y_catb = catb.predict(X_val)

                    if "y_min_max" in encoders:
                        y_catb = (
                            encoders["y_min_max"]
                            .inverse_transform(y_catb.reshape(-1, 1))
                            .ravel()
                        )
                    elif "y_log" in encoders:
                        y_catb = np.expm1(y_catb)

                mlflow.log_metrics(evaluate_regression(y_val, y_catb))
                mlflow.log_metric("fit_time", fit_trace["duration"])
                mlflow.log_metric("fit_memory_peak", fit_trace["peak"])
                mlflow.log_metric("score_time", score_trace["duration"])
                mlflow.log_metric("score_memory_peak", score_trace["peak"])
                mlflow.log_params(catb.get_params())

                mlflow.catboost.log_model(
                    cb_model=catb,
                    artifact_path="model",
                    signature=mlflow.models.infer_signature(X_train, y_catb),
                )
