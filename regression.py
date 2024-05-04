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
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import mlflow

mlflow.set_tracking_uri("http://localhost:5001")

warnings.filterwarnings(
    action="ignore",
    message=".*Distutils was imported before Setuptools.*",
)

warnings.filterwarnings(
    action="ignore",
    message=".*Setuptools is replacing distutils.*",
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
df = pd.read_csv("data/output.csv")

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
        "SBA_Appv",
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
    "target": "GrAppv",
}

# %%
df.drop(columns=['ChgOffDate'], inplace=True)

# %%
#Encode Data
encoder = OrdinalEncoder()

df[variables["categorical"]] = encoder.fit_transform(df[variables["categorical"]])

# Print the updated dataset
df.head()

# %%
X = df.drop('GrAppv', axis=1)
y = df['GrAppv']

# %%
#Split data set
from sklearn.model_selection import train_test_split

# Split the data into train and test sets (85% train, 15% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Split the train set further into train and validation sets (70% train, 15% validation)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15 / 0.85, random_state=42)

print("Train set shape:", X_train.shape)
print("Validation set shape:", X_val.shape)
print("Test set shape:", X_test.shape)

validate_rmse_score = []
test_rmse_score = []

# %% [markdown]
# Modelling

# %%
validate_lm(X_train,y_train,X_test,y_test)

# %%
validate_random_forest(X_train,y_train,X_test,y_test)

# %%
validate_decision_tree(X_train,y_train,X_test,y_test)

# %%
validate_ridge_regression(X_train,y_train,X_test,y_test)

# %%
validate_xg_boost(X_train,y_train,X_test,y_test)

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
