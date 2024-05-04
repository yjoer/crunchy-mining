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

experiment_name = "Bank Appv"
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
