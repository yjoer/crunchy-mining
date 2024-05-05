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
from src.regpipeline import (
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
df.drop(columns=['ChgOffDate'], inplace=True)

# %%
#Encode Data
encoder = OrdinalEncoder()

df[variables["categorical"]] = encoder.fit_transform(df[variables["categorical"]])

# Print the updated dataset
df.head()

# %%
X = df.drop('SBA_Appv', axis=1)
y = df['SBA_Appv']

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
