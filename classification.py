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
from crunchy_mining.sampling.samplers import PostSamplerV1
from crunchy_mining.sampling.samplers import PostSamplerV2
from crunchy_mining.sampling.samplers import PostSamplerV3
from crunchy_mining.sampling.samplers import PostSamplerV4
from crunchy_mining.sampling.samplers import PostSamplerV5
from crunchy_mining.sampling.samplers import PostSamplerV6
from crunchy_mining.sampling.samplers import PostSamplerV7
from crunchy_mining.sampling.samplers import PostSamplerV8
from crunchy_mining.sampling.samplers import SamplerV1
from crunchy_mining.sampling.samplers import SamplerV2

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
# Should the year be categorical or numerical?
# How to deal with dates?
variables = get_variables()

# %%
# Do we leak future information if we ignore the application date?

# %%
sampler = SamplerV1(variables)
sampler.sample(df)
train_val_sets_raw = sampler.train_val_sets

# %%
inspect_holdout_split_size(train_val_sets_raw, variables["target"])

# %%
inspect_cv_split_size(train_val_sets_raw, variables["target"])

# %% [markdown]
# ## Modify

# %%
# What's the better way to handle categorical variables without one-hot encoding to
# avoid the curse of dimensionality?
# Do we need to scale the outputs of the ordinal encoder?

# %%
preprocessor = PreprocessorV4(experiment_name, variables)

for name, (df_train, df_val) in train_val_sets_raw.items():
    preprocessor.fit(df_train, df_val, name=name)

# "name": (X_train, y_train, X_val, y_val)
preprocessor.save_train_val_sets()
train_val_sets_pp = preprocessor.get_train_val_sets()

# %%
postsampler = PostSamplerV1()
postsampler.sample(train_val_sets_pp)
train_val_sets = postsampler.train_val_sets

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
