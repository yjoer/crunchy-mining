# %%
import warnings

import altair as alt
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.pipeline import (
    get_variables,
    inspect_cv_split_size,
    inspect_holdout_split_size,
    validate_adaboost,
    validate_catboost,
    validate_decision_tree,
    validate_gaussian_nb,
    validate_knn,
    validate_lightgbm,
    validate_linear_svc,
    validate_logistic_regression,
    validate_random_forest,
    validate_xgboost,
)
from src.preprocessing.preprocessors import (
    PreprocessorV1,
    PreprocessorV2,
    PreprocessorV3,
    PreprocessorV4,
)

# %load_ext autoreload
# %autoreload 2

mlflow.set_tracking_uri("http://localhost:5001")

experiment_name = "Default"
mlflow.set_experiment(experiment_name)

warnings.filterwarnings(
    action="ignore",
    message=".*Distutils was imported before Setuptools.*",
)

warnings.filterwarnings(
    action="ignore",
    message=".*Setuptools is replacing distutils.*",
)

# %%
df = pd.read_parquet("data/output.parquet")

# %%
df.info()

# %% [markdown]
# ## Sample

# %%
counts = df["MIS_Status"].value_counts()
majority_class = counts.index[np.argmax(counts)]
minority_class = counts.index[np.argmin(counts)]
n_minority_class = np.min(counts)

# %%
df_sampled = pd.concat(
    [
        df[df["MIS_Status"] == majority_class].sample(
            n_minority_class,
            random_state=12345,
        ),
        df[df["MIS_Status"] == minority_class],
    ]
)

# %%
# Should the year be categorical or numerical?
# How to deal with dates?
variables = get_variables()

# %%
# Do we leak future information if we ignore the application date?
df_train, df_test = train_test_split(
    df_sampled,
    test_size=0.15,
    random_state=12345,
    stratify=df_sampled[variables["target"]],
)

# %%
# 1. Holdout method
df_train_sm, df_val = train_test_split(
    df_train,
    test_size=0.15 / 0.85,
    random_state=12345,
    stratify=df_train[variables["target"]],
)

# %%
inspect_holdout_split_size(df_train, df_train_sm, df_val, df_test, variables["target"])

# %%
# 2. Cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=12345)
skf_indices = list(skf.split(np.zeros(len(df_train)), df_train[variables["target"]]))

# %%
inspect_cv_split_size(df_train, skf_indices, variables["target"])

# %% [markdown]
# ## Modify

# %%
# What's the better way to handle categorical variables without one-hot encoding to
# avoid the curse of dimensionality?
# Do we need to scale the outputs of the ordinal encoder?

# %%
preprocessor = PreprocessorV4(experiment_name, variables)
preprocessor.fit(df_train_sm, df_val, name="validation")
preprocessor.fit(df_train, df_test, name="testing")

for i, (train_index, val_index) in enumerate(skf_indices):
    preprocessor.fit(
        df_train.iloc[train_index],
        df_train.iloc[val_index],
        name=f"fold_{i + 1}",
    )

# "name": (X_train, y_train, X_val, y_val)
preprocessor.save_train_val_sets()
train_val_sets = preprocessor.get_train_val_sets()

# %% [markdown]
# ## Model

# %%
validate_knn(train_val_sets)

# %%
validate_logistic_regression(train_val_sets)

# %%
validate_gaussian_nb(train_val_sets)

# %%
validate_linear_svc(train_val_sets)

# %%
validate_decision_tree(train_val_sets)

# %%
validate_random_forest(train_val_sets)

# %%
validate_adaboost(train_val_sets)

# %%
validate_xgboost(train_val_sets)

# %%
validate_lightgbm(train_val_sets)

# %%
validate_catboost(train_val_sets)

# %% [markdown]
# ## Assess

# %%
