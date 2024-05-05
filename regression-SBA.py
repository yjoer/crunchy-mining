# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% editable=true slideshow={"slide_type": ""}
import warnings
import mlflow
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from crunchy_mining.pipeline import inspect_holdout_split_size
from crunchy_mining.pipeline import inspect_cv_split_size
from crunchy_mining.preprocessing.preprocessors import PreprocessorReg


mlflow.set_tracking_uri("http://localhost:5001")

experiment_name = "SBA Guarantee"
mlflow.set_experiment(experiment_name)

warnings.filterwarnings(
    action="ignore",
    message=".*Distutils was imported before Setuptools.*",
)

warnings.filterwarnings(
    action="ignore",
    message=".*Setuptools is replacing distutils.*",
)

warnings.filterwarnings(
    action="ignore",
    message=".*Attempting to set identical low and high xlims.*",
)

# %%
from crunchy_mining.regpipeline import (
    validate_lm,
    validate_random_forest,
    validate_decision_tree,
    validate_ridge_regression,
    validate_xg_boost,
)

# %%
df = pd.read_parquet("data/output.parquet")

# %%
df.info()

# %%
variables = {
    "categorical": [
        "City",
        "State",
        "BankState",
        "ApprovalDate",
        "DisbursementDate",
        "Industry",
        "Active"
    ],
    "numerical": [
        "ApprovalFY",
        "Term",
        "NoEmp",
        "CreateJob",
        "RetainedJob",
        "FranchiseCode",
        "UrbanRural",
        "RevLineCr",
        "LowDoc",
        "DisbursementGross",
        "BalanceGross",
        "ChgOffPrinGr",
        "GrAppv"
        "DisbursementFY",
        "Is_Franchised",
        "Is_CreatedJob",
        "Is_RetainedJob",
        "RealEstate",
        "DaysTerm",
        "Recession",
        "DaysToDisbursement",
        "StateSame",
        "SBA_AppvPct",
        "AppvDisbursed",
        "Is_Existing",
        "MIS_Status"
    ],
    "target": "SBA_Appv",
}

# %%
df_train, df_test = train_test_split(
    df,
    test_size=0.15,
    random_state=12345,
)

# %%
# 1. Holdout method
df_train_sm, df_val = train_test_split(
    df_train,
    test_size=0.15 / 0.85,
    random_state=12345,
)

# %%
inspect_holdout_split_size(df_train, df_train_sm, df_val, df_test, variables["target"])

# %%
# 2. Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=12345)
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=12345)
kf_indices = list(kf.split(np.zeros(len(df_train)), df_train[variables["target"]]))

# %%
inspect_cv_split_size(df_train, kf_indices, variables["target"])

# %%
preprocessor = PreprocessorReg(experiment_name, variables)
preprocessor.fit(df_train_sm, df_val, name="validation")
preprocessor.fit(df_train, df_test, name="testing")

for i, (train_index, val_index) in enumerate(kf_indices):
    preprocessor.fit(
        df_train.iloc[train_index],
        df_train.iloc[val_index],
        name=f"fold_{i + 1}",
    )

preprocessor.save_train_val_sets()
train_val_sets = preprocessor.get_train_val_sets()
encoders = preprocessor.get_encoders()

# %%
print(encoders)

# %% [markdown]
# Modelling

# %%
validate_lm(train_val_sets, encoders)

# %%
validate_random_forest(train_val_sets, encoders)

# %%
validate_decision_tree(train_val_sets, encoders)

# %%
validate_ridge_regression(train_val_sets, encoders)

# %%
validate_xg_boost(train_val_sets, encoders)

# %% [markdown]
# Old Modelling

# %%
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(x_train_scaler, y_train_scaler)

# Model making a prediction on test data
y_pred = lm.predict(x_test_scaler)

# %%
print(y_pred)

# %%
result = y_scaler.inverse_transform(y_pred)

# %%
calculate_mape(y_test, result)

# %%
ndf = [Reg_Models_Evaluation_Metrics(lm,x_train_scaler,y_train_scaler,x_test_scaler,y_test,result)]

lm_score = pd.DataFrame(data = ndf, columns=['R2 Score','Adjusted R2 Score','Cross Validated R2 Score','RMSE', "MSE", "MAE", "MAPE"])
lm_score.insert(0, 'Model', 'Linear Regression')
lm_score

# %% editable=true slideshow={"slide_type": ""}
plt.figure(figsize = (10,5))
sns.regplot(x=y_test,y=result)
plt.title('Linear regression', fontsize = 20)

# %% [markdown]
# Random Forest

# %%
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators = 10, random_state = 0)
rf_reg.fit(x_train_scaler, y_train_scaler)

# Model making a prediction on test data
rf_y_pred = rf_reg.predict(x_test_scaler)

# %%
print(rf_y_pred)

# %%
rf_result = y_scaler.inverse_transform(rf_y_pred.reshape(-1,1))

# %%
print(rf_result)

# %%
ndf = [Reg_Models_Evaluation_Metrics(rf_reg,x_train_scaler,y_train_scaler,x_test_scaler,y_test,rf_result)]

rf_score = pd.DataFrame(data = ndf, columns=['R2 Score','Adjusted R2 Score','Cross Validated R2 Score','RMSE', "MSE", "MAE"])
rf_score.insert(0, 'Model', 'Random Forest')
rf_score

# %% [markdown]
# Decision Tree

# %%
from sklearn.tree import DecisionTreeRegressor

dt_reg = DecisionTreeRegressor()
dt_reg.fit(x_train_scaler, y_train_scaler)
dt_y_pred = dt_reg.predict(x_test_scaler)

# %%
print(dt_y_pred)

# %%
dt_result = y_scaler.inverse_transform(dt_y_pred.reshape(-1,1))

# %%
ndf = [Reg_Models_Evaluation_Metrics(rf_reg,x_train_scaler,y_train_scaler,x_test_scaler,y_test,dt_result)]

dt_score = pd.DataFrame(data = ndf, columns=['R2 Score','Adjusted R2 Score','Cross Validated R2 Score','RMSE', "MSE", "MAE"])
dt_score.insert(0, 'Model', 'Decision Tree')
dt_score

# %% [markdown]
# Ridge Regression

# %%
from sklearn.linear_model import Ridge

ridge_reg = Ridge(alpha=3, solver="cholesky")
ridge_reg.fit(x_train_scaler, y_train_scaler)

ridge_y_pred = ridge_reg.predict(x_test_scaler)

# %%
print(ridge_y_pred)

# %%
ridge_result = y_scaler.inverse_transform(ridge_y_pred)

# %%
ndf = [Reg_Models_Evaluation_Metrics(ridge_reg,x_train_scaler,y_train_scaler,x_test_scaler,y_test,ridge_result)]

ridge_score = pd.DataFrame(data = ndf, columns=['R2 Score','Adjusted R2 Score','Cross Validated R2 Score','RMSE', "MSE", "MAE"])
ridge_score.insert(0, 'Model', 'Ridge Regression')
ridge_score

# %% [markdown]
# XGBoost

# %%
from xgboost import XGBRegressor

xg_boost = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.8, colsample_bytree=0.8)
xg_boost.fit(x_train_scaler, y_train_scaler)

xg_boost_y_pred = xg_boost.predict(x_test_scaler)

# %%
print(xg_boost_y_pred)

# %%
xg_boost_result = y_scaler.inverse_transform(xg_boost_y_pred.reshape(-1,1))

# %%
ndf = [Reg_Models_Evaluation_Metrics(xg_boost,x_train_scaler,y_train_scaler,x_test_scaler,y_test,xg_boost_result)]

xg_score = pd.DataFrame(data = ndf, columns=['R2 Score','Adjusted R2 Score','Cross Validated R2 Score','RMSE', "MSE", "MAE"])
xg_score.insert(0, 'Model', 'XG Boost')
xg_score

# %% [markdown]
# Visual

# %%
predictions = pd.concat([lm_score, rf_score, dt_score, ridge_score, xg_score], ignore_index=True, sort=False)
predictions

# %%
f, axe = plt.subplots(1,1, figsize=(18,6))

predictions.sort_values(by=['Cross Validated R2 Score'], ascending=False, inplace=True)

sns.barplot(x='Cross Validated R2 Score', y='Model', data = predictions, ax = axe)
axe.set_xlabel('Cross Validated R2 Score', size=16)
axe.set_ylabel('Model')
axe.set_xlim(0,1.0)

axe.set(title='Model Performance')

plt.show()
