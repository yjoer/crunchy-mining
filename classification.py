# %%
import os
import warnings

import altair as alt
import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from hydra import compose
from hydra import initialize

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
from crunchy_mining.preprocessing.preprocessors import PreprocessorV5
from crunchy_mining.preprocessing.preprocessors import PreprocessorV6
from crunchy_mining.preprocessing.preprocessors import PreprocessorV7
from crunchy_mining.sampling.samplers import ResamplerV0
from crunchy_mining.sampling.samplers import ResamplerV1
from crunchy_mining.sampling.samplers import ResamplerV2
from crunchy_mining.sampling.samplers import ResamplerV3
from crunchy_mining.sampling.samplers import ResamplerV4
from crunchy_mining.sampling.samplers import ResamplerV5
from crunchy_mining.sampling.samplers import ResamplerV6
from crunchy_mining.sampling.samplers import ResamplerV7
from crunchy_mining.sampling.samplers import ResamplerV8
from crunchy_mining.sampling.samplers import SamplerV1
from crunchy_mining.sampling.samplers import SamplerV2

load_dotenv()

# %load_ext autoreload
# %autoreload 2

mlflow.set_tracking_uri("http://localhost:5001")

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
experiment = os.environ.get("CM_EXPERIMENT", "sampling_v1")

with initialize(version_base=None, config_path="conf"):
    cfg = compose(overrides=[f"+experiment={experiment}"])

# %%
cfg

# %%
mlflow.set_experiment(cfg.mlflow.experiment_name)

# %%
df = pd.read_parquet("data/output.parquet")

# %%
df.info()

# %% [markdown]
# ## Sample

# %%
# Should the year be categorical or numerical?
# How to deal with dates?

# %%
# Do we leak future information if we ignore the application date?

# %%
match cfg.sampling.variant:
    case 1:
        sampler = SamplerV1(cfg)
    case 2:
        sampler = SamplerV2(cfg)

sampler.sample(df)
train_val_sets_raw = sampler.train_val_sets

# %%
inspect_holdout_split_size(train_val_sets_raw, cfg.vars.stratify)

# %%
inspect_cv_split_size(train_val_sets_raw, cfg.vars.stratify)

# %% [markdown]
# ## Modify

# %%
# What's the better way to handle categorical variables without one-hot encoding to
# avoid the curse of dimensionality?
# Do we need to scale the outputs of the ordinal encoder?

# %% [markdown]
# ### Preprocessing

# %%
match cfg.preprocessing.variant:
    case 1:
        preprocessor = PreprocessorV1(cfg)
    case 2:
        preprocessor = PreprocessorV2(cfg)
    case 3:
        preprocessor = PreprocessorV3(cfg)
    case 4:
        preprocessor = PreprocessorV4(cfg)
    case 5:
        preprocessor = PreprocessorV5(cfg)
    case 6:
        preprocessor = PreprocessorV6(cfg)
    case 7:
        preprocessor = PreprocessorV7(cfg)

for name, (df_train, df_val) in train_val_sets_raw.items():
    preprocessor.fit(df_train, df_val, name=name)

# "name": (X_train, y_train, X_val, y_val)
preprocessor.save_train_val_sets()
preprocessor.save_encoders()
train_val_sets_pp = preprocessor.get_train_val_sets()

# %% [markdown]
# ### Resampling

# %%
match cfg.resampling.variant:
    case 0:
        resampler = ResamplerV0()
    case 1:
        resampler = ResamplerV1()
    case 2:
        resampler = ResamplerV2()
    case 3:
        resampler = ResamplerV3()
    case 4:
        resampler = ResamplerV4()
    case 5:
        resampler = ResamplerV5()
    case 6:
        resampler = ResamplerV6()
    case 7:
        resampler = ResamplerV7()
    case 8:
        resampler = ResamplerV8()

# "name": (X_train, y_train, X_val, y_val)
resampler.sample(train_val_sets_pp)
train_val_sets = resampler.train_val_sets

# %% [markdown]
# ## Model

# %% [markdown]
# ### Cross-Validation

# %%
if cfg.validation.models.knn:
    validate_knn(cfg, train_val_sets)

# %%
if cfg.validation.models.logistic_regression:
    validate_logistic_regression(cfg, train_val_sets)

# %%
if cfg.validation.models.gaussian_nb:
    validate_gaussian_nb(cfg, train_val_sets)

# %%
if cfg.validation.models.linear_svc:
    validate_linear_svc(cfg, train_val_sets)

# %%
if cfg.validation.models.decision_tree:
    validate_decision_tree(cfg, train_val_sets)

# %%
if cfg.validation.models.random_forest:
    validate_random_forest(cfg, train_val_sets)

# %%
if cfg.validation.models.adaboost:
    validate_adaboost(cfg, train_val_sets)

# %%
if cfg.validation.models.xgboost:
    validate_xgboost(cfg, train_val_sets)

# %%
if cfg.validation.models.lightgbm:
    validate_lightgbm(cfg, train_val_sets)

# %%
if cfg.validation.models.catboost:
    validate_catboost(cfg, train_val_sets)

# %% [markdown]
# ## Assess

# %% [markdown]
# ### Intrinsic Interpretation

# %%
if cfg.interpretation.intrinsic.models.logistic_regression:
    intrinsic_linear(cfg, train_val_sets, model_name="Logistic Regression")

# %%
if cfg.interpretation.intrinsic.models.linear_svc:
    intrinsic_linear(cfg, train_val_sets, model_name="Linear SVC")

# %%
if cfg.interpretation.intrinsic.models.decision_tree:
    intrinsic_trees(cfg, train_val_sets, model_name="Decision Tree")

# %%
if cfg.interpretation.intrinsic.models.random_forest:
    intrinsic_trees(cfg, train_val_sets, model_name="Random Forest")

# %%
if cfg.interpretation.intrinsic.models.adaboost:
    intrinsic_trees(cfg, train_val_sets, model_name="AdaBoost")

# %%
if cfg.interpretation.intrinsic.models.xgboost:
    intrinsic_xgboost(cfg, train_val_sets)

# %%
if cfg.interpretation.intrinsic.models.lightgbm:
    intrinsic_lightgbm(cfg, train_val_sets)

# %%
if cfg.interpretation.intrinsic.models.catboost:
    intrinsic_catboost(cfg, train_val_sets)

# %% [markdown]
# ### Permutation Feature Importance

# %%
# It takes forever on KNN
if cfg.interpretation.permutation_importance.models.knn:
    pimp(train_val_sets, model_name="KNN")

# %%
if cfg.interpretation.permutation_importance.models.logistic_regression:
    pimp(train_val_sets, model_name="Logistic Regression")

# %%
if cfg.interpretation.permutation_importance.models.gaussian_nb:
    pimp(train_val_sets, model_name="Gaussian NB")

# %%
if cfg.interpretation.permutation_importance.models.linear_svc:
    pimp(train_val_sets, model_name="Linear SVC")

# %%
if cfg.interpretation.permutation_importance.models.decision_tree:
    pimp(train_val_sets, model_name="Decision Tree")

# %%
if cfg.interpretation.permutation_importance.models.random_forest:
    pimp(train_val_sets, model_name="Random Forest")

# %%
if cfg.interpretation.permutation_importance.models.adaboost:
    pimp(train_val_sets, model_name="AdaBoost")

# %%
if cfg.interpretation.permutation_importance.models.xgboost:
    pimp(train_val_sets, model_name="XGBoost")

# %%
if cfg.interpretation.permutation_importance.models.lightgbm:
    pimp(train_val_sets, model_name="LightGBM")

# %%
if cfg.interpretation.permutation_importance.models.catboost:
    pimp(train_val_sets, model_name="CatBoost")

# %% [markdown]
# ### Partial Dependence Plot

# %%
# It took about 5.5 hours/model for KNN, 10 minutes/model for Random Forest to CatBoost,
# and under or within a minute for the rest.
if cfg.interpretation.partial_dependence.models.knn:
    pdp(cfg, train_val_sets, model_name="KNN")

# %%
if cfg.interpretation.partial_dependence.models.logistic_regression:
    pdp(cfg, train_val_sets, model_name="Logistic Regression")

# %%
if cfg.interpretation.partial_dependence.models.gaussian_nb:
    pdp(cfg, train_val_sets, model_name="Gaussian NB")

# %%
if cfg.interpretation.partial_dependence.models.linear_svc:
    pdp(cfg, train_val_sets, model_name="Linear SVC")

# %%
if cfg.interpretation.partial_dependence.models.decision_tree:
    pdp(cfg, train_val_sets, model_name="Decision Tree")

# %%
if cfg.interpretation.partial_dependence.models.random_forest:
    pdp(cfg, train_val_sets, model_name="Random Forest")

# %%
if cfg.interpretation.partial_dependence.models.adaboost:
    pdp(cfg, train_val_sets, model_name="AdaBoost")

# %%
if cfg.interpretation.partial_dependence.models.xgboost:
    pdp(cfg, train_val_sets, model_name="XGBoost")

# %%
if cfg.interpretation.partial_dependence.models.lightgbm:
    pdp(cfg, train_val_sets, model_name="LightGBM")

# %%
if cfg.interpretation.partial_dependence.models.catboost:
    pdp(cfg, train_val_sets, model_name="CatBoost")

# %%
