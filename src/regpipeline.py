# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import mlflow
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

from src.util import evaluate_regression
from src.util import trace_memory


def train_lm(X_train, y_train):
    lm = LinearRegression()
    lm.fit(X_train, y_train)

    return lm


def validate_lm(X_train,y_train,X_test,y_test):
    x_scaler = MinMaxScaler()
    x_train_scaler = x_scaler.fit_transform(X_train)
    x_test_scaler =  x_scaler.transform(X_test)

    y_scaler = MinMaxScaler()
    y_train_scaler = y_scaler.fit_transform(y_train.to_numpy().reshape(-1,1))
    y_test_scaler =  y_scaler.transform(y_test.to_numpy().reshape(-1,1))
    
    with mlflow.start_run(run_name="Linear Regression"):
        with trace_memory() as trace:
            lm = train_lm(x_train_scaler, y_train_scaler)

        y_pred = lm.predict(x_test_scaler)
        result = y_scaler.inverse_transform(y_pred)
        mlflow.log_metrics(evaluate_regression(lm,x_train_scaler,y_train_scaler,x_test_scaler,y_test,result))
        mlflow.log_metric("peak_memory_usage", trace["peak"])
        mlflow.log_params(lm.get_params())
    
        mlflow.sklearn.log_model(
                sk_model=lm,
                artifact_path="model",
                signature=mlflow.models.infer_signature(x_test_scaler, y_pred),
        )


