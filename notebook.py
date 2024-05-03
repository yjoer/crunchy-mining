# %%
import warnings

import altair as alt
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from crunchy_mining.pipeline import get_variables
from crunchy_mining.pipeline import inspect_cv_split_size
from crunchy_mining.pipeline import inspect_holdout_split_size
from crunchy_mining.pipeline import intrinsic_catboost
from crunchy_mining.pipeline import intrinsic_lightgbm
from crunchy_mining.pipeline import intrinsic_linear
from crunchy_mining.pipeline import intrinsic_trees
from crunchy_mining.pipeline import intrinsic_xgboost
from crunchy_mining.pipeline import pdp
from crunchy_mining.pipeline import pimp
from crunchy_mining.pipeline import validate_adaboost
from crunchy_mining.pipeline import validate_catboost
from crunchy_mining.pipeline import validate_decision_tree
from crunchy_mining.pipeline import validate_gaussian_nb
from crunchy_mining.pipeline import validate_knn
from crunchy_mining.pipeline import validate_lightgbm
from crunchy_mining.pipeline import validate_linear_svc
from crunchy_mining.pipeline import validate_logistic_regression
from crunchy_mining.pipeline import validate_random_forest
from crunchy_mining.pipeline import validate_xgboost
from crunchy_mining.preprocessing.preprocessors import GenericPreprocessor
from crunchy_mining.preprocessing.preprocessors import PreprocessorV1
from crunchy_mining.preprocessing.preprocessors import PreprocessorV2
from crunchy_mining.preprocessing.preprocessors import PreprocessorV3
from crunchy_mining.preprocessing.preprocessors import PreprocessorV4

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

warnings.filterwarnings(
    action="ignore",
    message=".*Attempting to set identical low and high xlims.*",
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

# %% [markdown]
# ### Intrinsic Interpretation

# %%
feature_names = variables["categorical"] + variables["numerical"]

# %%
intrinsic_linear(
    train_val_sets,
    model_name="Logistic Regression",
    feature_names=feature_names,
)

# %%
intrinsic_linear(train_val_sets, model_name="Linear SVC", feature_names=feature_names)

# %%
intrinsic_trees(train_val_sets, model_name="Decision Tree", feature_names=feature_names)

# %%
intrinsic_trees(train_val_sets, model_name="Random Forest", feature_names=feature_names)

# %%
intrinsic_trees(train_val_sets, model_name="AdaBoost", feature_names=feature_names)

# %%
intrinsic_xgboost(train_val_sets, feature_names=feature_names)

# %%
intrinsic_lightgbm(train_val_sets, feature_names=feature_names)

# %%
intrinsic_catboost(train_val_sets, feature_names=feature_names)

# %% [markdown]
# ### Permutation Feature Importance

# %%
# It takes forever on KNN
# pimp(train_val_sets, model_name="KNN")

# %%
pimp(train_val_sets, model_name="Logistic Regression")

# %%
pimp(train_val_sets, model_name="Gaussian NB")

# %%
pimp(train_val_sets, model_name="Linear SVC")

# %%
pimp(train_val_sets, model_name="Decision Tree")

# %%
pimp(train_val_sets, model_name="Random Forest")

# %%
pimp(train_val_sets, model_name="AdaBoost")

# %%
pimp(train_val_sets, model_name="XGBoost")

# %%
pimp(train_val_sets, model_name="LightGBM")

# %%
pimp(train_val_sets, model_name="CatBoost")

# %% [markdown]
# ### Partial Dependence Plot

# %%
# It took about 5.5 hours/model for KNN, 10 minutes/model for Random Forest to CatBoost,
# and under or within a minute for the rest.
# pdp(train_val_sets, model_name="KNN", feature_names=feature_names)

# %%
pdp(train_val_sets, model_name="Logistic Regression", feature_names=feature_names)

# %%
pdp(train_val_sets, model_name="Gaussian NB", feature_names=feature_names)

# %%
pdp(train_val_sets, model_name="Linear SVC", feature_names=feature_names)

# %%
pdp(train_val_sets, model_name="Decision Tree", feature_names=feature_names)

# %%
pdp(train_val_sets, model_name="Random Forest", feature_names=feature_names)

# %%
pdp(train_val_sets, model_name="AdaBoost", feature_names=feature_names)

# %%
pdp(train_val_sets, model_name="XGBoost", feature_names=feature_names)

# %%
pdp(train_val_sets, model_name="LightGBM", feature_names=feature_names)

# %%
pdp(train_val_sets, model_name="CatBoost", feature_names=feature_names)

# %%
