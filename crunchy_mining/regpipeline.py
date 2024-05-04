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
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

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


def train_random_forest(X_train, y_train):
    rf_reg = RandomForestRegressor(n_estimators = 10, random_state = 0)
    rf_reg.fit(X_train, y_train)

    return rf_reg


def validate_random_forest(X_train,y_train,X_test,y_test):
    x_scaler = MinMaxScaler()
    x_train_scaler = x_scaler.fit_transform(X_train)
    x_test_scaler =  x_scaler.transform(X_test)

    y_scaler = MinMaxScaler()
    y_train_scaler = y_scaler.fit_transform(y_train.to_numpy().reshape(-1,1))
    y_test_scaler =  y_scaler.transform(y_test.to_numpy().reshape(-1,1))
    
    with mlflow.start_run(run_name="Random Forest"):
        with trace_memory() as trace:
            rf_reg = train_random_forest(x_train_scaler, y_train_scaler)
        
        rf_y_pred = rf_reg.predict(x_test_scaler)
        rf_result = y_scaler.inverse_transform(rf_y_pred.reshape(-1,1))
        mlflow.log_metrics(evaluate_regression(rf_reg,x_train_scaler,y_train_scaler,x_test_scaler,y_test,rf_result))
        mlflow.log_metric("peak_memory_usage", trace["peak"])
        mlflow.log_params(rf_reg.get_params())
    
        mlflow.sklearn.log_model(
                sk_model=rf_reg,
                artifact_path="model",
                signature=mlflow.models.infer_signature(x_test_scaler, rf_y_pred),
        )


def train_decision_tree(X_train, y_train):
    dt_reg = DecisionTreeRegressor()
    dt_reg.fit(X_train, y_train)

    return dt_reg


def validate_decision_tree(X_train,y_train,X_test,y_test):
    x_scaler = MinMaxScaler()
    x_train_scaler = x_scaler.fit_transform(X_train)
    x_test_scaler =  x_scaler.transform(X_test)

    y_scaler = MinMaxScaler()
    y_train_scaler = y_scaler.fit_transform(y_train.to_numpy().reshape(-1,1))
    y_test_scaler =  y_scaler.transform(y_test.to_numpy().reshape(-1,1))
    
    with mlflow.start_run(run_name="Decision Tree"):
        with trace_memory() as trace:
            dt_reg = train_decision_tree(x_train_scaler, y_train_scaler)
        
        dt_y_pred = dt_reg.predict(x_test_scaler)
        dt_result = y_scaler.inverse_transform(dt_y_pred.reshape(-1,1))
        mlflow.log_metrics(evaluate_regression(dt_reg,x_train_scaler,y_train_scaler,x_test_scaler,y_test,dt_result))
        mlflow.log_metric("peak_memory_usage", trace["peak"])
        mlflow.log_params(dt_reg.get_params())
    
        mlflow.sklearn.log_model(
                sk_model=dt_reg,
                artifact_path="model",
                signature=mlflow.models.infer_signature(x_test_scaler, dt_y_pred),
        )


def train_ridge_reg(X_train, y_train):
    ridge_reg = Ridge(alpha=3, solver="cholesky")
    ridge_reg.fit(X_train, y_train)

    return ridge_reg


def validate_ridge_regression(X_train,y_train,X_test,y_test):
    x_scaler = MinMaxScaler()
    x_train_scaler = x_scaler.fit_transform(X_train)
    x_test_scaler =  x_scaler.transform(X_test)

    y_scaler = MinMaxScaler()
    y_train_scaler = y_scaler.fit_transform(y_train.to_numpy().reshape(-1,1))
    y_test_scaler =  y_scaler.transform(y_test.to_numpy().reshape(-1,1))
    
    with mlflow.start_run(run_name="Ridge Regression"):
        with trace_memory() as trace:
            ridge_reg = train_ridge_reg(x_train_scaler, y_train_scaler)
            
        ridge_y_pred = ridge_reg.predict(x_test_scaler)
        ridge_result = y_scaler.inverse_transform(ridge_y_pred)
        mlflow.log_metrics(evaluate_regression(ridge_reg,x_train_scaler,y_train_scaler,x_test_scaler,y_test,ridge_result))
        mlflow.log_metric("peak_memory_usage", trace["peak"])
        mlflow.log_params(ridge_reg.get_params())
    
        mlflow.sklearn.log_model(
                sk_model=ridge_reg,
                artifact_path="model",
                signature=mlflow.models.infer_signature(x_test_scaler, ridge_y_pred),
        )


def train_xg_boost(X_train, y_train):
    xg_boost = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.8, colsample_bytree=0.8)
    xg_boost.fit(X_train, y_train)

    return xg_boost


def validate_xg_boost(X_train,y_train,X_test,y_test):
    x_scaler = MinMaxScaler()
    x_train_scaler = x_scaler.fit_transform(X_train)
    x_test_scaler =  x_scaler.transform(X_test)

    y_scaler = MinMaxScaler()
    y_train_scaler = y_scaler.fit_transform(y_train.to_numpy().reshape(-1,1))
    y_test_scaler =  y_scaler.transform(y_test.to_numpy().reshape(-1,1))
    
    with mlflow.start_run(run_name="XG_Boost"):
        with trace_memory() as trace:
            xg_boost = train_xg_boost(x_train_scaler, y_train_scaler)
            
        xg_boost_y_pred = xg_boost.predict(x_test_scaler)
        xg_boost_result = y_scaler.inverse_transform(xg_boost_y_pred.reshape(-1,1))
        mlflow.log_metrics(evaluate_regression(xg_boost,x_train_scaler,y_train_scaler,x_test_scaler,y_test,xg_boost_result))
        mlflow.log_metric("peak_memory_usage", trace["peak"])
        mlflow.log_params(xg_boost.get_params())
    
        mlflow.sklearn.log_model(
                sk_model=xg_boost,
                artifact_path="model",
                signature=mlflow.models.infer_signature(x_test_scaler, xg_boost_y_pred),
        )
